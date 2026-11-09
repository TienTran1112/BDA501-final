"""Build the Gold feature table for price modelling from the Silver layer, then verify it.

Run from the project folder:  docker compose exec spark spark-submit src/build_gold_expedia.py

Splits by search date: train = Nov 2012-Mar 2013, validation = Apr 2013, test = May-Jun 2013.

Leakage rules:
  * Rows flagged price_outlier are excluded.
  * Outcome columns (position, clicked, booked, gross_bookings_usd) are not carried into Gold.
  * country/destination/hotel average log prices come from the train window only.
  * On train rows those averages are out-of-fold: each search belongs to one of 5 folds and its rows
    only see averages from the other 4, so no row's own price feeds its own features.
  * Averages are shrunk towards the next level up (hotel -> destination -> country -> overall),
    so a hotel with no training history simply takes its destination's average.
"""
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

SILVER = "/workspace/data/lake/silver/expedia"
GOLD = "/workspace/data/lake/gold/expedia_features"
TRAIN_END = "2013-04-01"
VALIDATION_END = "2013-05-01"
FOLDS = 5
SMOOTHING = 10.0
SILVER_ROWS, OUTLIER_ROWS = 16_540_159, 9_540

KEEP = [
    "snapshot_date", "search_id", "id", "source_split", "checkin_date", "lead_time_days", "day_of_week", "nights",
    "price_usd", "country_id", "destination_id", "site_id", "star_rating", "review_score", "is_brand",
    "location_score1", "location_score2", "log_historical_price", "adults", "children", "rooms", "saturday_night",
    "has_promotion",
]
OUTCOMES = {"position", "clicked", "booked", "gross_bookings_usd"}

failures = []


def check(label, passed, detail=""):
    if not passed:
        failures.append(label)
    print(f"[{'PASS' if passed else 'FAIL'}] {label} {detail}".rstrip(), flush=True)


def add_average(df, stats, key, name, prior):
    """Add <name>_rows and a shrunk <name>_mean. Rows with a null fold (validation, test) see all of train."""
    def totals(keys, s, n):
        return stats.groupBy(*keys).agg(F.sum("label").alias(s), F.count(F.lit(1)).alias(n))

    joined = (df.join(totals([key], "_s", "_n"), key, "left")
              .join(totals([key, "fold"], "_fold_s", "_fold_n"), [key, "fold"], "left"))
    rows = F.coalesce(F.col("_n"), F.lit(0)) - F.coalesce(F.col("_fold_n"), F.lit(0))
    total = F.coalesce(F.col("_s"), F.lit(0.0)) - F.coalesce(F.col("_fold_s"), F.lit(0.0))
    return (joined
            .withColumn(f"{name}_rows", rows.cast("int"))
            .withColumn(f"{name}_mean", (total + SMOOTHING * prior) / (rows + SMOOTHING))
            .drop("_s", "_n", "_fold_s", "_fold_n"))


spark = SparkSession.builder.appName("bda501-build-gold").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
start = time.time()

train_end, validation_end = F.lit(TRAIN_END).cast("date"), F.lit(VALIDATION_END).cast("date")
split = (F.when(F.col("snapshot_date") < train_end, "train")
         .when(F.col("snapshot_date") < validation_end, "validation")
         .otherwise("test"))
base = (spark.read.parquet(SILVER)
        .filter(~F.col("price_outlier"))
        .select(*KEEP)
        .withColumn("label", F.log("price_usd"))
        .withColumn("split", split)
        .withColumn("fold", F.when(F.col("split") == "train", F.pmod(F.hash("search_id"), F.lit(FOLDS)))))

stats = base.filter(F.col("split") == "train").select("id", "destination_id", "country_id", "fold", "label").cache()
overall = stats.agg(F.avg("label")).first()[0]
print(f"Train rows: {stats.count()}, overall mean log price: {overall:.4f}", flush=True)

gold = add_average(base, stats, "country_id", "country", F.lit(overall))
gold = add_average(gold, stats, "destination_id", "destination", F.col("country_mean"))
gold = add_average(gold, stats, "id", "hotel", F.col("destination_mean"))
gold = gold.drop("fold")
gold.repartition("snapshot_date").write.mode("overwrite").partitionBy("snapshot_date").parquet(GOLD)
stats.unpersist()
print(f"Gold written: {time.time() - start:.0f}s\n", flush=True)

# Verify what is on disk ---------------------------------------------------------------------------
gold = spark.read.parquet(GOLD).cache()
by_split = {r["split"]: r for r in gold.groupBy("split").agg(
    F.count(F.lit(1)).alias("rows"), F.min("snapshot_date").alias("first"), F.max("snapshot_date").alias("last"),
    F.countDistinct("id").alias("hotels")).collect()}
total = sum(r["rows"] for r in by_split.values())
for name in ("train", "validation", "test"):
    r = by_split[name]
    print(f"{name:<11} rows={r['rows']:<9} hotels={r['hotels']:<7} {r['first']} to {r['last']}")

check("rows = Silver minus flagged outliers", total == SILVER_ROWS - OUTLIER_ROWS, f"{total}")
check("train ends before validation", str(by_split["train"]["last"]) < TRAIN_END, str(by_split["train"]["last"]))
check("validation ends before test", str(by_split["validation"]["last"]) < VALIDATION_END <= str(by_split["test"]["first"]))
check("(search_id, id) is unique", gold.select("search_id", "id").distinct().count() == total)
check("no outcome columns in Gold", not OUTCOMES & set(gold.columns))
nulls = gold.agg(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in
                   ("label", "country_mean", "destination_mean", "hotel_mean", "hotel_rows")]).first().asDict()
check("no nulls in label or averages", not any(nulls.values()), str(nulls))

train_counts = gold.filter(F.col("split") == "train").groupBy("id").agg(F.count(F.lit(1)).alias("true_rows"))
held_out = gold.filter(F.col("split") != "train").select("id", "hotel_rows").distinct().join(train_counts, "id", "left")
wrong = held_out.filter(F.coalesce(F.col("true_rows"), F.lit(0)) != F.col("hotel_rows")).count()
check("validation/test averages use exactly the train rows", wrong == 0, f"mismatches={wrong}")

train_rows = gold.filter(F.col("split") == "train").join(train_counts, "id")
own_fold_used = train_rows.filter(F.col("hotel_rows") >= F.col("true_rows")).count()
check("train rows never see their own fold", own_fold_used == 0, f"violations={own_fold_used}")
no_history = train_rows.filter((F.col("hotel_rows") == 0)
                               & (F.abs(F.col("hotel_mean") - F.col("destination_mean")) > 1e-9)).count()
check("no out-of-fold history means destination average", no_history == 0, f"violations={no_history}")

test = gold.filter(F.col("split") == "test")
unseen = test.filter(F.col("hotel_rows") == 0).count()
print(f"\nTest rows from hotels with no training history: {unseen} ({100 * unseen / by_split['test']['rows']:.2f}%)")
print(f"Total: {time.time() - start:.0f}s")

spark.stop()
if failures:
    sys.exit(f"Failed: {failures}")
print("All checks passed")
