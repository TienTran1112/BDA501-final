"""Live reads from the serving layer: queries Q1 to Q5 plus the two precomputed dashboard collections.

Q1 to Q5 are deliberately not cached, so each panel's time is a real round trip to the database, including reading all rows.
"""
import math
import os
import time

import pandas as pd
import streamlit as st
from cassandra import ConsistencyLevel
from cassandra.cluster import EXEC_PROFILE_DEFAULT, Cluster, ExecutionProfile
from cassandra.policies import DCAwareRoundRobinPolicy, TokenAwarePolicy
from pymongo import ASCENDING, DESCENDING, MongoClient

CASSANDRA_HOSTS = os.environ.get("CASSANDRA_HOSTS", "cassandra1,cassandra2,cassandra3").split(",")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongo1:27017,mongo2:27017,mongo3:27017/?replicaSet=rs0")
DATABASE = "hotel_serving"
KM_PER_DEGREE = 111.32
EARTH_RADIUS_KM = 6378.1  # MongoDB's GeoJSON radius, so a displayed distance agrees with what $near filtered on.

Q1 = "SELECT snapshot_ts, price_night, currency, is_sold_out FROM price_history WHERE hotel_id=? AND checkin_date=?"
Q2 = "SELECT hotel_id, price_night FROM cheapest_by_city WHERE city=? AND checkin_date=? AND snap_date=? LIMIT 20"
PROFILE_COLUMNS = ["hotel_id", "stars", "review_score", "brand", "country_id", "destination_id",
                   "longitude", "latitude", "lowest_30d", "highest_30d", "price_rows_30d"]


@st.cache_resource(show_spinner="Connecting to Cassandra")
def cassandra():
    profile = ExecutionProfile(load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy("dc1")),
                               consistency_level=ConsistencyLevel.LOCAL_QUORUM, request_timeout=15)
    cluster = Cluster(CASSANDRA_HOSTS, execution_profiles={EXEC_PROFILE_DEFAULT: profile})
    session = cluster.connect("hotel_serving")
    return session, session.prepare(Q1), session.prepare(Q2)


@st.cache_resource(show_spinner="Connecting to MongoDB")
def mongo():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    return client[DATABASE]


def timed(fetch):
    started = time.perf_counter()
    rows = list(fetch())
    return rows, (time.perf_counter() - started) * 1000


def price_history(hotel, checkin):
    """Q1, Cassandra: every recorded price for one hotel and one check-in date."""
    session, statement, _ = cassandra()
    rows, ms = timed(lambda: session.execute(statement, (hotel, checkin)))
    frame = pd.DataFrame([(r.snapshot_ts, float(r.price_night)) for r in rows], columns=["snapshot_ts", "price"])
    return frame.sort_values("snapshot_ts", ignore_index=True), ms


def cheapest(destination, checkin, snapshot):
    """Q2, Cassandra: the 20 cheapest hotels in one destination, for one check-in and one snapshot date."""
    session, _, statement = cassandra()
    rows, ms = timed(lambda: session.execute(statement, (destination, checkin, snapshot)))
    return pd.DataFrame([(r.hotel_id, float(r.price_night)) for r in rows], columns=["hotel_id", "price"]), ms


def profile_row(doc):
    stats = doc.get("stats_30d", {})
    longitude, latitude = doc["location"]["coordinates"]
    return (doc["_id"], doc.get("star_rating"), doc.get("review", {}).get("score"), doc.get("is_brand"), doc.get("country_id"),
            doc.get("destination_id"), longitude, latitude, stats.get("min"), stats.get("max"), stats.get("rows"))


def profiles(hotels):
    """Q3, MongoDB: hotel profiles by _id."""
    docs, ms = timed(lambda: mongo().hotels.find({"_id": {"$in": list(hotels)}}))
    return pd.DataFrame([profile_row(d) for d in docs], columns=PROFILE_COLUMNS), ms


def distance_km(longitude, latitude, centre_longitude, centre_latitude):
    a1, a2 = math.radians(centre_latitude), math.radians(latitude)
    delta_a, delta_o = a2 - a1, math.radians(longitude - centre_longitude)
    h = math.sin(delta_a / 2) ** 2 + math.cos(a1) * math.cos(a2) * math.sin(delta_o / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def nearby(centre, radius_km, min_score, limit):
    """Q4, MongoDB: hotels within a radius of a point, with a minimum review score, nearest first."""
    near = {"$near": {"$geometry": {"type": "Point", "coordinates": list(centre)}, "$maxDistance": radius_km * 1000}}
    docs, ms = timed(lambda: mongo().hotels.find({"location": near, "review.score": {"$gte": min_score}}).limit(limit))
    frame = pd.DataFrame([profile_row(d) for d in docs], columns=PROFILE_COLUMNS)
    longitude, latitude = centre
    frame["east_km"] = (frame["longitude"] - longitude) * KM_PER_DEGREE * math.cos(math.radians(latitude))
    frame["north_km"] = (frame["latitude"] - latitude) * KM_PER_DEGREE
    frame["distance_km"] = [distance_km(o, a, longitude, latitude) for o, a in zip(frame["longitude"], frame["latitude"])]
    return frame, ms


def weekly(destination):
    """Q5, MongoDB: promotion rate and price per week for one destination."""
    docs, ms = timed(lambda: mongo().destination_weekly.find({"destination_id": destination}).sort("week_start", ASCENDING))
    columns = ["week_start", "searches_rows", "promotion_rate", "median_price", "mean_price"]
    return pd.DataFrame([[d.get(c) for c in columns] for d in docs], columns=columns), ms


ALERT_COLUMNS = ["week_start", "destination_id", "median_price", "trailing_median", "ratio", "searches_rows", "promotion_rate"]


def alerts_for(destination):
    docs = mongo().price_alerts.find({"destination_id": destination}).sort("week_start", ASCENDING)
    return pd.DataFrame([[d.get(c) for c in ALERT_COLUMNS] for d in docs], columns=ALERT_COLUMNS)


def top_alerts(limit=200):
    docs, ms = timed(lambda: mongo().price_alerts.find().sort("ratio", DESCENDING).limit(limit))
    return pd.DataFrame([[d.get(c) for c in ALERT_COLUMNS] for d in docs], columns=ALERT_COLUMNS), ms


def alert_totals():
    collection = mongo().price_alerts
    return collection.count_documents({}), len(collection.distinct("destination_id"))


def hotels_in_destination(destination, limit=200):
    docs = mongo().hotels.find({"destination_id": destination}, {"_id": 1, "location.coordinates": 1}).sort("_id", ASCENDING).limit(limit)
    return {d["_id"]: tuple(d["location"]["coordinates"]) for d in docs}


@st.cache_data(ttl="1h", show_spinner="Loading the destination list")
def catalog():
    """Precomputed by src/build_dashboard_tables.py. Busiest destination first."""
    return list(mongo().destination_catalog.find().sort("rows", DESCENDING))
