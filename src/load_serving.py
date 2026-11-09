"""Load the serving databases from the Silver layer, then compare what they return with the lake.

Run from the project folder, with the matching database profile started (see docker-compose.yml):
    docker compose exec spark spark-submit --master "local[6]" --driver-memory 2g src/load_serving.py cassandra
    docker compose exec spark spark-submit --master "local[6]" --driver-memory 2g src/load_serving.py verify-cassandra
    docker compose exec spark spark-submit --master "local[6]" --driver-memory 2g \
        --packages org.mongodb.spark:mongo-spark-connector_2.13:11.1.0 src/load_serving.py mongo
    (and verify-mongo, with the same --packages)
Add --fraction 0.0625 to any of them for a 1/16 pilot.

What differs from the proposal, because Expedia does not contain the fields:
  * city is Expedia's search destination_id. A hotel can appear under several destinations.
  * is_sold_out exists in price_history but stays empty, and there is no sold_out_rate: Expedia has no sold-out data.
  * Hotel GPS does not exist in Expedia. location is SYNTHETIC, generated from a hash, and every document says so in
    location_source. It exists only to exercise the 2dsphere index in the load test and has no analytic meaning.
  * review.score is on Expedia's 0-5 scale, and there is no review count.
  * Q5 is promotion rate and price by destination and week, in place of sold-out rate by district and week.

price_history adds search_id to the clustering key: 52 (hotel, check-in, timestamp) groups would otherwise overwrite each other.
"""
import argparse
import itertools
import json
import math
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

SILVER = "/workspace/data/lake/silver/expedia"
REPORTS = "/workspace/data/reports"
CONTACT_POINTS = ["cassandra1", "cassandra2", "cassandra3"]
KEYSPACE = "hotel_serving"
MONGO_URI = "mongodb://mongo1:27017,mongo2:27017,mongo3:27017/?replicaSet=rs0&w=majority"
DATABASE = "hotel_serving"
LAST_30_DAYS_START = "2013-06-01"
PARTITIONS, CHUNK_ROWS, CONCURRENCY, MAX_ATTEMPTS = 48, 5_000, 16, 8
VERIFY_KEYS = 3_000

SCHEMA = [
    f"CREATE KEYSPACE IF NOT EXISTS {KEYSPACE} WITH replication = {{'class': 'NetworkTopologyStrategy', 'dc1': 3}}",
    f"""CREATE TABLE IF NOT EXISTS {KEYSPACE}.price_history (
          hotel_id bigint, checkin_date date, snapshot_ts timestamp, search_id bigint,
          price_night decimal, currency text, is_sold_out boolean, has_promotion boolean,
          PRIMARY KEY ((hotel_id, checkin_date), snapshot_ts, search_id)
        ) WITH CLUSTERING ORDER BY (snapshot_ts DESC, search_id ASC)
          AND compaction = {{'class': 'TimeWindowCompactionStrategy',
                             'compaction_window_unit': 'DAYS', 'compaction_window_size': 7}}""",
    f"""CREATE TABLE IF NOT EXISTS {KEYSPACE}.cheapest_by_city (
          city int, checkin_date date, snap_date date, price_night decimal, hotel_id bigint,
          PRIMARY KEY ((city, checkin_date, snap_date), price_night, hotel_id)
        ) WITH CLUSTERING ORDER BY (price_night ASC, hotel_id ASC)""",
]
INSERT_HISTORY = (f"INSERT INTO {KEYSPACE}.price_history "
                  "(hotel_id, checkin_date, snapshot_ts, search_id, price_night, currency, has_promotion) "
                  "VALUES (?, ?, ?, ?, ?, ?, ?)")
INSERT_CHEAPEST = (f"INSERT INTO {KEYSPACE}.cheapest_by_city (city, checkin_date, snap_date, price_night, hotel_id) "
                   "VALUES (?, ?, ?, ?, ?)")


def cassandra_session(keyspace=None):
    from cassandra import ConsistencyLevel
    from cassandra.cluster import EXEC_PROFILE_DEFAULT, Cluster, ExecutionProfile
    from cassandra.policies import DCAwareRoundRobinPolicy, TokenAwarePolicy
    profile = ExecutionProfile(load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy("dc1")),
                               consistency_level=ConsistencyLevel.LOCAL_QUORUM, request_timeout=60)
    cluster = Cluster(CONTACT_POINTS, execution_profiles={EXEC_PROFILE_DEFAULT: profile})
    return cluster, cluster.connect(keyspace)


def writer(statement):
    """Return a mapPartitions function that writes rows and yields how many were acknowledged.

    Every statement is an idempotent upsert, so retrying after a timeout is safe.
    """
    def write(rows):
        from cassandra.concurrent import execute_concurrent_with_args
        cluster, session = cassandra_session(KEYSPACE)
        prepared = session.prepare(statement)
        written = 0
        rows = iter(rows)
        while chunk := [tuple(row) for row in itertools.islice(rows, CHUNK_ROWS)]:
            pending = chunk
            for attempt in range(MAX_ATTEMPTS):
                results = execute_concurrent_with_args(session, prepared, pending, concurrency=CONCURRENCY,
                                                       raise_on_first_error=False)
                pending = [args for args, (ok, _) in zip(pending, results) if not ok]
                if not pending:
                    break
                time.sleep(min(2 ** attempt, 30))
            if pending:
                raise RuntimeError(f"{len(pending)} rows still failing after {MAX_ATTEMPTS} attempts")
            written += len(chunk)
        cluster.shutdown()
        yield written
    return write


def silver_rows(spark, fraction):
    silver = spark.read.parquet(SILVER).filter(~F.col("price_outlier"))
    if fraction < 1:
        silver = silver.filter(F.pmod(F.hash("search_id"), F.lit(16)) < int(round(fraction * 16)))
    return silver


def history_frame(silver):
    return silver.select(F.col("id").alias("hotel_id"), "checkin_date", "snapshot_ts", "search_id",
                         F.col("price_usd").cast("decimal(12,2)").alias("price_night"), "currency", "has_promotion")


def cheapest_frame(silver):
    return (silver.groupBy(F.col("destination_id").alias("city"), "checkin_date", F.col("snapshot_date").alias("snap_date"),
                           F.col("id").alias("hotel_id"))
            .agg(F.min("price_usd").cast("decimal(12,2)").alias("price_night"))
            .select("city", "checkin_date", "snap_date", "price_night", "hotel_id"))


def save_report(fraction, name, entry):
    path = f"{REPORTS}/serving_load{'' if fraction >= 1 else '_pilot'}.json"
    report = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    report[name] = entry
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def load_table(fraction, name, frame, statement):
    expected = frame.count()
    start = time.time()
    written = frame.repartition(PARTITIONS).rdd.mapPartitions(writer(statement)).sum()
    seconds = time.time() - start
    if written != expected:
        raise RuntimeError(f"{name}: wrote {written}, expected {expected}")
    print(f"{name}: {written} rows in {seconds:.0f}s ({written / seconds:.0f} rows/s)", flush=True)
    save_report(fraction, name, {"rows": written, "seconds": round(seconds), "rows_per_s": round(written / seconds)})


def load_cassandra(spark, fraction):
    cluster, session = cassandra_session()
    for statement in SCHEMA:
        session.execute(statement, timeout=120)
    cluster.shutdown()
    silver = silver_rows(spark, fraction)  # Not cached: the Cassandra nodes leave Spark under 3 GB of memory.
    load_table(fraction, "price_history", history_frame(silver), INSERT_HISTORY)
    load_table(fraction, "cheapest_by_city", cheapest_frame(silver), INSERT_CHEAPEST)


def sample_keys(frame, columns, count):
    """A fixed pseudo-random set of partitions, plus the largest ones."""
    sizes = frame.groupBy(*columns).count().cache()
    uniform = sizes.withColumn("r", F.xxhash64(*columns)).orderBy("r").limit(count).select(*columns)
    largest = sizes.orderBy(F.desc("count"), *columns).limit(200).select(*columns)
    return uniform.union(largest).distinct()


def as_date(value):
    return value.date() if hasattr(value, "date") else value


def verify_cassandra(spark, fraction):
    silver = silver_rows(spark, fraction).cache()
    failures = 0
    cluster, session = cassandra_session(KEYSPACE)

    history = history_frame(silver)
    keys = sample_keys(history, ["hotel_id", "checkin_date"], VERIFY_KEYS)
    expected = {}
    for r in history.join(keys, ["hotel_id", "checkin_date"]).collect():
        expected.setdefault((r["hotel_id"], r["checkin_date"]), {})[(r["snapshot_ts"], r["search_id"])] = r["price_night"]
    query = session.prepare("SELECT snapshot_ts, search_id, price_night FROM price_history WHERE hotel_id=? AND checkin_date=?")
    for key, rows in expected.items():
        found = list(session.execute(query, key))
        timestamps = [r.snapshot_ts for r in found]
        if {(r.snapshot_ts, r.search_id): r.price_night for r in found} != rows or timestamps != sorted(timestamps, reverse=True):
            failures += 1
    print(f"price_history: {len(expected)} partitions, {sum(map(len, expected.values()))} rows compared, "
          f"{failures} differ", flush=True)

    cheapest_failures = 0
    cheapest = cheapest_frame(silver)
    columns = ["city", "checkin_date", "snap_date"]
    keys = sample_keys(cheapest, columns, VERIFY_KEYS)
    expected = {}
    for r in cheapest.join(keys, columns).collect():
        expected.setdefault((r["city"], r["checkin_date"], r["snap_date"]), {})[r["hotel_id"]] = r["price_night"]
    query = session.prepare("SELECT hotel_id, price_night FROM cheapest_by_city WHERE city=? AND checkin_date=? AND snap_date=?")
    for key, rows in expected.items():
        found = list(session.execute(query, key))
        prices = [r.price_night for r in found]
        if {r.hotel_id: r.price_night for r in found} != rows or prices != sorted(prices):
            cheapest_failures += 1
    print(f"cheapest_by_city: {len(expected)} partitions, {sum(map(len, expected.values()))} rows compared, "
          f"{cheapest_failures} differ", flush=True)
    cluster.shutdown()
    if failures or cheapest_failures:
        raise SystemExit("Cassandra verification failed")
    print("Cassandra verification passed", flush=True)


def start_replica_set():
    from pymongo import MongoClient
    from pymongo.errors import OperationFailure
    direct = MongoClient("mongodb://mongo1:27017/?directConnection=true", serverSelectionTimeoutMS=60_000)
    try:
        direct.admin.command("replSetInitiate", {"_id": "rs0", "members": [
            {"_id": i, "host": f"mongo{i + 1}:27017"} for i in range(3)]})
    except OperationFailure as error:
        if "already initialized" not in str(error):
            raise
    direct.close()
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=120_000)
    client.admin.command("ping")
    return client


def unit_hash(column, salt):
    return F.pmod(F.hash(column, F.lit(salt)), F.lit(1_000_000)) / 1_000_000.0


def hotels_frame(silver):
    """One document per hotel. Coordinates are synthetic: within 15 km of a made-up centre for its main destination."""
    ts = F.col("snapshot_ts")
    recent = F.col("snapshot_date") >= F.lit(LAST_30_DAYS_START).cast("date")
    latest_known = lambda c: F.max_by(c, F.when(F.col(c).isNotNull(), ts))  # noqa: E731
    base = silver.groupBy(F.col("id").alias("_id")).agg(
        F.first("country_id").alias("country_id"),
        F.mode("destination_id").alias("destination_id"),
        latest_known("star_rating").cast("double").alias("star_rating"),
        latest_known("review_score").cast("double").alias("score"),
        F.max_by("is_brand", ts).alias("is_brand"),
        F.min(F.when(recent, F.col("price_usd"))).alias("price_min"),
        F.max(F.when(recent, F.col("price_usd"))).alias("price_max"),
        F.sum(F.when(recent, 1).otherwise(0)).alias("price_rows"),
    )
    lat_c = F.lit(-55.0) + 110.0 * unit_hash("destination_id", "lat")
    lon_c = F.lit(-175.0) + 350.0 * unit_hash("destination_id", "lon")
    radius_km = 15.0 * F.sqrt(unit_hash("_id", "radius"))
    angle = 2 * math.pi * unit_hash("_id", "angle")
    lat = lat_c + radius_km * F.cos(angle) / 111.32
    lon = lon_c + radius_km * F.sin(angle) / (111.32 * F.cos(F.radians(lat_c)))
    stats = F.when(F.col("price_rows") > 0, F.struct(
        F.col("price_min").alias("min"), F.col("price_max").alias("max"), F.col("price_rows").alias("rows")))
    return base.select(
        "_id", "country_id", "destination_id", "star_rating", "is_brand",
        F.struct(F.lit("Point").alias("type"), F.array(lon, lat).alias("coordinates")).alias("location"),
        F.lit("synthetic").alias("location_source"),
        F.struct(F.col("score").alias("score"), F.lit(5).alias("scale")).alias("review"),
        stats.alias("stats_30d"))


WEEKLY_SQL = """
    SELECT concat(destination_id, '|', date_trunc('week', snapshot_date)) AS _id, destination_id,
           CAST(date_trunc('week', snapshot_date) AS date) AS week_start,
           count(*) AS searches_rows, avg(CAST(has_promotion AS int)) AS promotion_rate,
           percentile_approx(price_usd, 0.5, 10000) AS median_price, avg(price_usd) AS mean_price
    FROM silver GROUP BY destination_id, date_trunc('week', snapshot_date)
"""


def write_mongo(frame, collection):
    (frame.write.format("mongodb").mode("overwrite").option("connection.uri", MONGO_URI).option("database", DATABASE)
     .option("collection", collection).option("ignoreNullValues", "true").save())


def load_mongo(spark, fraction):
    client = start_replica_set()
    silver = silver_rows(spark, fraction).cache()
    silver.createOrReplaceTempView("silver")

    for name, frame in (("hotels", hotels_frame(silver)), ("destination_weekly", spark.sql(WEEKLY_SQL))):
        frame = frame.cache()
        rows = frame.count()
        start = time.time()
        write_mongo(frame, name)
        seconds = time.time() - start
        print(f"{name}: {rows} documents in {seconds:.0f}s ({rows / seconds:.0f} docs/s)", flush=True)
        save_report(fraction, name, {"rows": rows, "seconds": round(seconds), "rows_per_s": round(rows / seconds)})

    database = client[DATABASE]
    database.hotels.create_index([("location", "2dsphere"), ("review.score", 1)])
    database.destination_weekly.create_index([("destination_id", 1), ("week_start", 1)])
    print("indexes created", flush=True)


def verify_mongo(spark, fraction):
    client = start_replica_set()
    database = client[DATABASE]
    silver = silver_rows(spark, fraction).cache()
    silver.createOrReplaceTempView("silver")
    failures = []

    hotels_expected = silver.select("id").distinct().count()
    if database.hotels.count_documents({}) != hotels_expected:
        failures.append("hotel count")
    print(f"hotels: {database.hotels.count_documents({})} documents, lake has {hotels_expected}", flush=True)

    # Recompute a sample of hotels in plain Python from the raw rows: a different route from the Spark aggregation.
    ids = [r["id"] for r in silver.select("id").distinct().orderBy(F.xxhash64("id")).limit(VERIFY_KEYS).collect()]
    rows = (silver.filter(F.col("id").isin(ids))
            .select("id", "snapshot_ts", "snapshot_date", "star_rating", "review_score", "price_usd", "country_id").collect())
    by_hotel = {}
    for r in rows:
        by_hotel.setdefault(r["id"], []).append(r)
    limit = spark.sql(f"SELECT DATE'{LAST_30_DAYS_START}' AS d").first()["d"]
    bad = 0
    for hotel, group in by_hotel.items():
        doc = database.hotels.find_one({"_id": hotel})
        recent = [r["price_usd"] for r in group if r["snapshot_date"] >= limit]
        stars = [r for r in group if r["star_rating"] is not None]
        scores = [r for r in group if r["review_score"] is not None]
        ok = doc is not None and doc["country_id"] == group[0]["country_id"] and doc["location_source"] == "synthetic"
        if ok and recent:
            ok = doc.get("stats_30d", {}).get("rows") == len(recent) and doc["stats_30d"]["min"] == min(recent) \
                and doc["stats_30d"]["max"] == max(recent)
        elif ok:
            ok = "stats_30d" not in doc
        # A tie on timestamp may legitimately pick either row, so accept any value at the newest timestamp.
        if ok and stars:
            newest = max(r["snapshot_ts"] for r in stars)
            ok = doc["star_rating"] in {r["star_rating"] for r in stars if r["snapshot_ts"] == newest}
        if ok and scores:
            newest = max(r["snapshot_ts"] for r in scores)
            ok = doc["review"]["score"] in {r["review_score"] for r in scores if r["snapshot_ts"] == newest}
        bad += not ok
    print(f"hotels: {len(by_hotel)} recomputed from raw rows, {bad} differ", flush=True)
    if bad:
        failures.append("hotel contents")

    outside = database.hotels.count_documents({"$or": [
        {"location.coordinates.0": {"$not": {"$gte": -180, "$lte": 180}}},
        {"location.coordinates.1": {"$not": {"$gte": -90, "$lte": 90}}}]})
    if outside:
        failures.append("coordinates out of range")
    indexes = database.hotels.index_information()
    if not any(("location", "2dsphere") in i["key"] for i in indexes.values()):
        failures.append("2dsphere index")

    weekly = {r["k"]: r for r in silver.withColumn("week", F.date_trunc("week", "snapshot_date"))
              .groupBy("destination_id", "week")
              .agg(F.concat_ws("|", F.col("destination_id"), F.col("week")).alias("k0"), F.count(F.lit(1)).alias("n"),
                   F.avg(F.col("has_promotion").cast("int")).alias("promo"))
              .withColumnRenamed("k0", "k").collect()}
    docs = list(database.destination_weekly.find({}, {"searches_rows": 1, "promotion_rate": 1}))
    wrong = sum(1 for d in docs if d["_id"] not in weekly or weekly[d["_id"]]["n"] != d["searches_rows"]
                or abs(weekly[d["_id"]]["promo"] - d["promotion_rate"]) > 1e-9)
    print(f"destination_weekly: {len(docs)} documents, lake has {len(weekly)}, {wrong} differ", flush=True)
    if wrong or len(docs) != len(weekly):
        failures.append("weekly contents")

    members = client.admin.command("replSetGetStatus")["members"]
    print("replica set: " + ", ".join(f"{m['name']} {m['stateStr']}" for m in members), flush=True)
    if len(members) != 3:
        failures.append("replica set size")
    if failures:
        raise SystemExit(f"MongoDB verification failed: {failures}")
    print("MongoDB verification passed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["cassandra", "verify-cassandra", "mongo", "verify-mongo"])
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()
    spark = SparkSession.builder.appName(f"bda501-serving-{args.command}").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    {"cassandra": load_cassandra, "verify-cassandra": verify_cassandra,
     "mongo": load_mongo, "verify-mongo": verify_mongo}[args.command](spark, args.fraction)
    spark.stop()


if __name__ == "__main__":
    main()
