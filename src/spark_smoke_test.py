"""Prove the Spark setup works on the Silver layer: read it, check it, write to the lake, train a small model.

Run from the project folder:  docker compose exec spark spark-submit src/spark_smoke_test.py

The model section only proves that Spark ML runs end to end on a 2% sample. Its numbers are not project results.
"""
import math
import os
import shutil
import sys
import time

import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

SILVER = "/workspace/data/lake/silver/expedia"
WRITE_TEST = "/workspace/data/lake/_smoke_test"
TRAIN_ROWS, TEST_ROWS = 9_917_530, 6_622_629
OUTLIER_ROWS = 9_540

failures = []


def check(label, passed, detail=""):
    if not passed:
        failures.append(label)
    print(f"[{'PASS' if passed else 'FAIL'}] {label} {detail}".rstrip(), flush=True)


def as_int(column):
    return F.sum(column.cast("int"))


spark = SparkSession.builder.appName("bda501-smoke-test").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print(f"Spark {spark.version}, master {spark.sparkContext.master}, "
      f"parallelism {spark.sparkContext.defaultParallelism}, "
      f"driver memory {spark.conf.get('spark.driver.memory')}", flush=True)

# 1. Read and verify ---------------------------------------------------------------------------
start = time.time()
df = spark.read.parquet(SILVER)
print("\nSchema as Spark sees it:")
df.printSchema()

s = df.agg(
    F.count(F.lit(1)).alias("rows"),
    F.sum("price_usd").alias("price_sum"),
    as_int(F.col("price_outlier")).alias("outliers"),
    F.countDistinct("snapshot_date").alias("days"),
    F.min("snapshot_ts").cast("string").alias("first_ts"),
    F.max("snapshot_ts").cast("string").alias("last_ts"),
    as_int(F.col("checkin_date") != F.date_add("snapshot_date", F.col("lead_time_days"))).alias("bad_checkin"),
    as_int(F.col("day_of_week") != F.weekday("checkin_date") + 1).alias("bad_weekday"),
).first()
splits = {r["source_split"]: r for r in df.groupBy("source_split").agg(
    F.count(F.lit(1)).alias("rows"), F.count("booked").alias("booked_known")).collect()}
distinct_keys = df.select("search_id", "id").distinct().count()

table = ds.dataset(SILVER, format="parquet", partitioning="hive").to_table(columns=["price_usd"])
arrow_sum = pc.sum(table["price_usd"]).as_py()

check("total rows", s["rows"] == TRAIN_ROWS + TEST_ROWS, f"{s['rows']}")
check("train rows", splits["train"]["rows"] == TRAIN_ROWS, f"{splits['train']['rows']}")
check("test rows", splits["test"]["rows"] == TEST_ROWS, f"{splits['test']['rows']}")
check("booked known on train only", splits["train"]["booked_known"] == TRAIN_ROWS and splits["test"]["booked_known"] == 0)
check("price total agrees with PyArrow", math.isclose(s["price_sum"], arrow_sum, rel_tol=1e-9), f"{s['price_sum']:.2f}")
check("price_outlier rows", s["outliers"] == OUTLIER_ROWS, f"{s['outliers']}")
check("snapshot days", s["days"] == 242, f"{s['days']}")
check("first timestamp", s["first_ts"] == "2012-11-01 00:01:37", s["first_ts"])
check("last timestamp", s["last_ts"] == "2013-06-30 23:58:24", s["last_ts"])
check("checkin_date = snapshot_date + lead_time_days", s["bad_checkin"] == 0, f"mismatches={s['bad_checkin']}")
check("day_of_week is ISO weekday", s["bad_weekday"] == 0, f"mismatches={s['bad_weekday']}")
check("(search_id, id) is unique", distinct_keys == s["rows"], f"{distinct_keys}")
print(f"Read and verify: {time.time() - start:.0f}s", flush=True)

# 2. Write back to the lake through the Windows bind mount --------------------------------------
start = time.time()
one_day = df.filter(F.col("snapshot_date") == "2013-01-15")
expected = one_day.count()
one_day.write.mode("overwrite").partitionBy("source_split").parquet(WRITE_TEST)
written = spark.read.parquet(WRITE_TEST).count()
check("Parquet write and read back", written == expected and expected > 0, f"{written}")
shutil.rmtree(WRITE_TEST)
check("write test removed", not os.path.exists(WRITE_TEST))
print(f"Write test: {time.time() - start:.0f}s", flush=True)

# 3. Spark ML end to end on a 2% sample, split by time -------------------------------------------
start = time.time()
sample = (df.filter(~F.col("price_outlier"))
          .sample(withReplacement=False, fraction=0.02, seed=42)
          .withColumn("label", F.log("price_usd"))
          .cache())
train = sample.filter(F.col("snapshot_date") < "2013-05-01")
test = sample.filter(F.col("snapshot_date") >= "2013-05-01")
train_rows, test_rows = train.count(), test.count()
print(f"\nSample: train {train_rows} rows (before May 2013), test {test_rows} rows (May-Jun 2013)", flush=True)

features = ["lead_time_days", "day_of_week", "nights", "star_rating_filled", "review_score_filled", "location_score1",
            "is_brand", "has_promotion", "adults", "rooms", "saturday_night"]
model = Pipeline(stages=[
    Imputer(strategy="median", inputCols=["star_rating", "review_score"],
            outputCols=["star_rating_filled", "review_score_filled"]),
    VectorAssembler(inputCols=features, outputCol="features"),
    GBTRegressor(featuresCol="features", labelCol="label", maxIter=20, maxDepth=5, seed=42),
]).fit(train)

median_price = train.approxQuantile("price_usd", [0.5], 0.001)[0]
errors = model.transform(test).withColumn("predicted", F.exp("prediction")).agg(
    F.avg(F.abs(F.col("predicted") - F.col("price_usd"))).alias("mae"),
    F.avg(F.abs(F.col("predicted") - F.col("price_usd")) / F.col("price_usd")).alias("mape"),
    F.avg(F.abs(F.lit(median_price) - F.col("price_usd"))).alias("baseline_mae"),
    F.avg(F.abs(F.lit(median_price) - F.col("price_usd")) / F.col("price_usd")).alias("baseline_mape"),
).first()
print(f"GBT:             MAE ${errors['mae']:.2f}   MAPE {100 * errors['mape']:.1f}%")
print(f"Median baseline: MAE ${errors['baseline_mae']:.2f}   MAPE {100 * errors['baseline_mape']:.1f}%")
check("model beats the median baseline", errors["mae"] < errors["baseline_mae"])
print(f"Spark ML: {time.time() - start:.0f}s", flush=True)

spark.stop()
if failures:
    sys.exit(f"Failed: {failures}")
print("\nAll checks passed")
