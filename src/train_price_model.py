"""Train the price model (GBT on log price) on the Gold layer and evaluate it against baselines.

Run from the project folder:
    docker compose exec spark spark-submit src/train_price_model.py                  (full data)
    docker compose exec spark spark-submit src/train_price_model.py --fraction 0.0625 (dry run, writes to /tmp only)

Tree depth is chosen on the April validation set using a sample of train. May-Jun test data is only touched once,
after the configuration is fixed. Full output: model, test/validation predictions, and a JSON report.
"""
import argparse
import json
import os
import time

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Imputer, OneHotEncoder, SQLTransformer, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

GOLD = "/workspace/data/lake/gold/expedia_features"
DEPTHS = (5, 7, 9)
TREES = 50
TUNE_FRACTION = 0.05

FILLED = ["star_rating", "review_score", "location_score2", "log_historical_price"]
FEATURES = [
    "lead_time_days", "nights", "day_of_week", "checkin_month", "adults", "children", "rooms",
    "saturday_night_num", "has_promotion_num", "is_brand_num", "location_score1",
    *[f"{c}_filled" for c in FILLED], *[f"{c}_missing" for c in FILLED],
    "country_mean", "destination_mean", "hotel_mean", "log_hotel_rows", "site_vec",
]


def preparation_pipeline():
    missing = ", ".join(f"CAST({c} IS NULL AS DOUBLE) AS {c}_missing" for c in FILLED)
    return Pipeline(stages=[
        SQLTransformer(statement=f"""
            SELECT *, month(checkin_date) AS checkin_month,
                   CAST(saturday_night AS DOUBLE) AS saturday_night_num,
                   CAST(has_promotion AS DOUBLE) AS has_promotion_num,
                   CAST(is_brand AS DOUBLE) AS is_brand_num,
                   log1p(hotel_rows) AS log_hotel_rows, {missing}
            FROM __THIS__"""),
        StringIndexer(inputCol="site_id", outputCol="site_idx", handleInvalid="keep"),
        OneHotEncoder(inputCol="site_idx", outputCol="site_vec", handleInvalid="keep"),
        Imputer(strategy="median", inputCols=FILLED, outputCols=[f"{c}_filled" for c in FILLED]),
        VectorAssembler(inputCols=FEATURES, outputCol="features"),
    ])


def gbt(depth):
    return GBTRegressor(featuresCol="features", labelCol="label", maxIter=TREES, maxDepth=depth, stepSize=0.1,
                        maxBins=32, maxMemoryInMB=512, seed=42)


def metric_columns(name, column):
    predicted = F.col(column)
    actual = F.col("price_usd")
    error = predicted - actual
    pct = F.abs(error) / actual
    log_error = F.log(predicted) - F.col("label")
    return [
        F.avg(F.abs(error)).alias(f"{name}|mae"),
        F.sqrt(F.avg(error * error)).alias(f"{name}|rmse"),
        F.avg(pct).alias(f"{name}|mape"),
        F.percentile_approx(pct, 0.5, 5000).alias(f"{name}|median_ape"),
        F.sqrt(F.avg(log_error * log_error)).alias(f"{name}|log_rmse"),
        (1 - F.avg(log_error * log_error) / F.var_pop("label")).alias(f"{name}|log_r2"),
    ]


def evaluate(df, models, group):
    """Return {group value: {model: {metric: value}}}."""
    exprs = [F.count(F.lit(1)).alias("rows")] + [e for name, col in models.items() for e in metric_columns(name, col)]
    result = {}
    for row in df.groupBy(group).agg(*exprs).collect():
        entry = {"rows": row["rows"]}
        for key, value in row.asDict().items():
            if "|" in key:
                name, metric = key.split("|")
                entry.setdefault(name, {})[metric] = value
        result[str(row[group])] = entry
    return result


def show(title, entry):
    print(f"\n{title} ({entry['rows']} rows)")
    print(f"{'model':<24}{'MAE $':>9}{'RMSE $':>10}{'MAPE %':>9}{'MdAPE %':>9}{'log RMSE':>10}{'log R2':>8}")
    for name, m in entry.items():
        if name != "rows":
            print(f"{name:<24}{m['mae']:>9.2f}{m['rmse']:>10.2f}{100 * m['mape']:>9.1f}"
                  f"{100 * m['median_ape']:>9.1f}{m['log_rmse']:>10.4f}{m['log_r2']:>8.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()
    root = "/workspace/data" if args.fraction >= 1 else "/tmp/dryrun"

    spark = SparkSession.builder.appName("bda501-price-model").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")
    started = time.time()
    report = {"fraction": args.fraction, "trees": TREES, "tuning": [], "timings_s": {}}

    gold = spark.read.parquet(GOLD)
    if args.fraction < 1:
        gold = gold.sample(withReplacement=False, fraction=args.fraction, seed=42)
    train = gold.filter(F.col("split") == "train")
    held_out = gold.filter(F.col("split") != "train")
    validation = gold.filter(F.col("split") == "validation")

    prep = preparation_pipeline().fit(train)
    train_vectors = prep.transform(train)
    feature_meta = train_vectors.schema["features"].metadata
    train_vectors = train_vectors.select("features", "label")
    validation_vectors = prep.transform(validation).select("features", "label", "price_usd")

    # Choose depth on April data only ----------------------------------------------------------------
    t = time.time()
    tune_sample = train_vectors.sample(withReplacement=False, fraction=TUNE_FRACTION, seed=7).cache()
    print(f"Tuning on {tune_sample.count()} train rows", flush=True)
    best_depth, best_mae = None, float("inf")
    for depth in DEPTHS:
        t_depth = time.time()
        scored = gbt(depth).fit(tune_sample).transform(validation_vectors)
        mae = scored.agg(F.avg(F.abs(F.exp("prediction") - F.col("price_usd")))).first()[0]
        report["tuning"].append({"depth": depth, "validation_mae": mae, "seconds": round(time.time() - t_depth)})
        print(f"  depth {depth}: validation MAE ${mae:.2f} ({time.time() - t_depth:.0f}s)", flush=True)
        if mae < best_mae:
            best_depth, best_mae = depth, mae
    tune_sample.unpersist()
    report["best_depth"] = best_depth
    report["timings_s"]["tuning"] = round(time.time() - t)
    print(f"Chosen depth: {best_depth}", flush=True)

    # Final fit on the whole training window ----------------------------------------------------------
    t = time.time()
    fitted = gbt(best_depth).fit(train_vectors)
    report["timings_s"]["final_fit"] = round(time.time() - t)
    print(f"Final fit: {report['timings_s']['final_fit']}s", flush=True)
    model = PipelineModel(stages=[*prep.stages, fitted])
    model.write().overwrite().save(f"{root}/lake/models/expedia_price_gbt")

    # Score once, store, then compute every metric from the stored predictions --------------------------
    t = time.time()
    predictions_path = f"{root}/lake/gold/expedia_price_predictions"
    (model.transform(held_out)
     .select("snapshot_date", "split", "search_id", "id", "checkin_date", "lead_time_days", "hotel_rows",
             "price_usd", "label", F.exp("prediction").alias("predicted_price"),
             F.exp("hotel_mean").alias("hotel_average_price"),
             F.exp("destination_mean").alias("destination_average_price"))
     .repartition("snapshot_date").write.mode("overwrite").partitionBy("snapshot_date").parquet(predictions_path))
    report["timings_s"]["scoring"] = round(time.time() - t)

    median_price = train.agg(F.percentile_approx("price_usd", 0.5, 10000)).first()[0]
    report["train_median_price"] = median_price
    predictions = spark.read.parquet(predictions_path).withColumn("overall_median", F.lit(float(median_price))).cache()
    models = {
        "GBT model": "predicted_price",
        "hotel average": "hotel_average_price",
        "destination average": "destination_average_price",
        "overall median": "overall_median",
    }

    overall = evaluate(predictions, models, "split")
    report["metrics"] = overall
    show("VALIDATION, April 2013", overall["validation"])
    show("TEST, May-June 2013", overall["test"])

    test = predictions.filter(F.col("split") == "test")
    segment = test.withColumn("seen", F.when(F.col("hotel_rows") > 0, "seen in training").otherwise("new hotel"))
    report["test_by_history"] = evaluate(segment, models, "seen")
    for name, entry in sorted(report["test_by_history"].items()):
        show(f"TEST, {name}", entry)

    bucket = F.when(F.col("lead_time_days") <= 1, "1) 0-1 days").when(F.col("lead_time_days") <= 7, "2) 2-7 days") \
        .when(F.col("lead_time_days") <= 30, "3) 8-30 days").when(F.col("lead_time_days") <= 90, "4) 31-90 days") \
        .otherwise("5) 91+ days")
    report["test_by_lead_time"] = evaluate(test.withColumn("lead", bucket), models, "lead")
    for name, entry in sorted(report["test_by_lead_time"].items()):
        show(f"TEST, lead time {name}", entry)

    # Feature importance, with the one-hot site columns added together ---------------------------------
    names = {a["idx"]: a["name"] for group in feature_meta["ml_attr"]["attrs"].values() for a in group}
    totals = {}
    for index, value in enumerate(fitted.featureImportances.toArray().tolist()):
        name = "site_id" if names[index].startswith("site_vec") else names[index]
        totals[name] = totals.get(name, 0.0) + value
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    report["feature_importance"] = dict(ranked)
    print("\nFeature importance:")
    for name, value in ranked[:15]:
        print(f"  {name:<28}{value:.4f}")

    report["timings_s"]["total"] = round(time.time() - started)
    os.makedirs(f"{root}/reports", exist_ok=True)
    with open(f"{root}/reports/expedia_price_model.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nTotal: {report['timings_s']['total']}s")
    print("Finished")
    spark.stop()


if __name__ == "__main__":
    main()
