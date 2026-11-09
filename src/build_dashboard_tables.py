"""Precompute the two MongoDB collections the dashboard needs and cannot get from a live query.

Run from the project folder with the mongo profile started:
    docker compose exec spark spark-submit --master "local[6]" --driver-memory 3g \
        --packages org.mongodb.spark:mongo-spark-connector_2.13:11.1.0 src/build_dashboard_tables.py

destination_catalog
    One document per busy destination. Cassandra can only be read by full partition key, so the dashboard's pickers need
    a list of keys that exist: the (check-in, snapshot) pairs with the most hotels and the longest hotel price histories.
    It also holds the median price by star rating and by check-in week for proposal slide 15's price index.

price_alerts
    One document per destination and week whose median price is unusually high. Rule, computed from destination_weekly:
    the week and at least MIN_TRAILING of the 4 weeks before it each have at least MIN_ROWS rows, and the week's median is
    at least ALERT_RATIO times the median of those weekly medians.
    Limit: which hotels and travel dates people searched changes from week to week, so an alert can be a change in search mix
    and not a price rise.
"""
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from load_serving import DATABASE, MONGO_URI, REPORTS, silver_rows, start_replica_set, write_mongo

TOP_DESTINATIONS, TOP_COMBOS, TOP_HISTORIES = 300, 40, 15
MIN_COMBO_HOTELS, MIN_HISTORY_ROWS, MIN_GROUP_ROWS = 5, 3, 30
ALERT_RATIO, MIN_ROWS, MIN_TRAILING = 1.25, 200, 3


def alerts_frame(spark):
    weekly = (spark.read.format("mongodb").option("connection.uri", MONGO_URI).option("database", DATABASE)
              .option("collection", "destination_weekly").load())
    weekly.withColumn("week_start", F.to_date("week_start")).filter(F.col("searches_rows") >= MIN_ROWS).createOrReplaceTempView("eligible")
    return spark.sql(f"""
        SELECT _id, destination_id, week_start, searches_rows, promotion_rate, median_price, trailing_median, trailing_weeks,
               median_price / trailing_median AS ratio
        FROM (SELECT *, count(*) OVER w AS trailing_weeks, percentile(median_price, 0.5) OVER w AS trailing_median
              FROM eligible
              WINDOW w AS (PARTITION BY destination_id ORDER BY unix_date(week_start) RANGE BETWEEN 28 PRECEDING AND 7 PRECEDING))
        WHERE trailing_weeks >= {MIN_TRAILING} AND median_price >= {ALERT_RATIO} * trailing_median
    """)


def iso(column):
    return F.date_format(column, "yyyy-MM-dd")


def top_per_destination(frame, count, *order):
    window = Window.partitionBy("destination_id").orderBy(*order)
    return frame.withColumn("rank", F.row_number().over(window)).filter(F.col("rank") <= count).drop("rank")


def nested(frame, name, *fields):
    return frame.groupBy("destination_id").agg(F.sort_array(F.collect_list(F.struct(*fields))).alias(name))


def catalog_frame(silver, alert_destinations):
    busiest = (silver.groupBy("destination_id")
               .agg(F.mode("country_id").alias("country_id"), F.count(F.lit(1)).alias("rows"), F.countDistinct("id").alias("hotels")))
    ranked = busiest.withColumn("rank", F.row_number().over(Window.orderBy(F.desc("rows"), "destination_id")))
    wanted = ranked.join(alert_destinations.withColumn("alerted", F.lit(True)), "destination_id", "left") \
        .filter((F.col("rank") <= TOP_DESTINATIONS) | F.col("alerted").isNotNull()).drop("rank", "alerted").cache()
    rows = silver.join(wanted.select("destination_id"), "destination_id").cache()

    combos = (rows.groupBy("destination_id", "checkin_date", "snapshot_date").agg(F.countDistinct("id").alias("hotels"))
              .filter(F.col("hotels") >= MIN_COMBO_HOTELS))
    combos = top_per_destination(combos, TOP_COMBOS, F.desc("hotels"), F.desc("checkin_date"), F.desc("snapshot_date"))
    combos = nested(combos, "combos", iso("checkin_date").alias("checkin_date"), iso("snapshot_date").alias("snapshot_date"), "hotels")

    # A Cassandra history partition is (hotel, check-in) across every destination, so the count covers all of them.
    sizes = (silver.groupBy("id", "checkin_date").agg(F.count(F.lit(1)).alias("rows")).filter(F.col("rows") >= MIN_HISTORY_ROWS))
    hotels_here = rows.select("destination_id", "id", "checkin_date").distinct()
    histories = top_per_destination(sizes.join(hotels_here, ["id", "checkin_date"]), TOP_HISTORIES,
                                    F.desc("rows"), "id", F.desc("checkin_date"))
    histories = nested(histories, "long_histories", "rows", F.col("id").alias("hotel_id"), iso("checkin_date").alias("checkin_date"))

    median = F.percentile_approx("price_usd", 0.5, 10000).alias("median_price")
    by_star = (rows.filter(F.col("star_rating").isNotNull()).groupBy("destination_id", F.col("star_rating").cast("double").alias("star"))
               .agg(median, F.count(F.lit(1)).alias("rows")).filter(F.col("rows") >= MIN_GROUP_ROWS))
    by_star = nested(by_star, "by_star", "star", "median_price", "rows")
    by_checkin = (rows.groupBy("destination_id", F.date_trunc("week", "checkin_date").alias("week")).agg(median, F.count(F.lit(1)).alias("rows"))
                  .filter(F.col("rows") >= MIN_GROUP_ROWS))
    by_checkin = nested(by_checkin, "by_checkin_week", iso("week").alias("week"), "median_price", "rows")

    catalog = wanted.withColumnRenamed("destination_id", "_id")
    for part in (combos, histories, by_star, by_checkin):
        catalog = catalog.join(part.withColumnRenamed("destination_id", "_id"), "_id", "left")
    return catalog


def main():
    spark = SparkSession.builder.appName("bda501-dashboard-tables").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    client = start_replica_set()
    database = client[DATABASE]
    start = time.time()

    alerts = alerts_frame(spark).cache()
    alert_count = alerts.count()
    write_mongo(alerts, "price_alerts")
    database.price_alerts.create_index([("destination_id", 1), ("week_start", 1)])
    database.price_alerts.create_index([("ratio", -1)])
    print(f"price_alerts: {alert_count} documents", flush=True)

    columns = ("id", "destination_id", "country_id", "checkin_date", "snapshot_date", "star_rating", "price_usd")
    silver = silver_rows(spark, 1.0).select(*columns).cache()
    catalog = catalog_frame(silver, alerts.select("destination_id").distinct()).cache()
    catalog_count = catalog.count()
    write_mongo(catalog, "destination_catalog")
    database.destination_catalog.create_index([("rows", -1)])
    print(f"destination_catalog: {catalog_count} documents", flush=True)

    database.hotels.create_index([("destination_id", 1)])  # For the centre-hotel picker on the Nearby page.
    report = {"price_alerts": alert_count, "destination_catalog": catalog_count, "seconds": round(time.time() - start),
              "rule": {"ratio": ALERT_RATIO, "min_rows": MIN_ROWS, "min_trailing_weeks": MIN_TRAILING}}
    with open(f"{REPORTS}/dashboard_tables.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
