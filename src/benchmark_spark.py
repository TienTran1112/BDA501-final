"""Spark performance test from the proposal: run time and records/second at 1x, 4x and 16x data,
while changing the number of partitions and executors.

Run from the project folder:
    docker compose exec spark python src/benchmark_spark.py                  (every test)
    docker compose exec spark python src/benchmark_spark.py --configs scale_1x --repeats 1

16x is the whole Gold table. 1x and 4x are 1/16 and 4/16 of its searches, and 1x is contained in 4x.
The partition and executor comparisons use 4x data, because one executor on the full table would take hours.
Each run is its own spark-submit, so runs share no JVM cache.

Each run measures three stages:
    etl    read Parquet, out-of-fold hotel averages (group, join), write Parquet by date
    fit    fit the feature pipeline and a small GBT (10 trees, depth 5) on the train rows
    score  apply the model to the validation and test rows and compute MAE

The GBT here is deliberately smaller than the final model, so timings are not the cost of the final model.

Executors are separate JVMs (Spark local-cluster mode) on this one machine: 2 cores each and 4 GB in total
however many there are, so only parallelism changes. There is no network between them, so this is not a cluster test.
AQE partition coalescing is off, otherwise Spark would undo the partition setting being tested.
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request

GOLD = "/workspace/data/lake/gold/expedia_features"
WORK = "/tmp/benchmark"
REPORT = "/workspace/data/reports/spark_scaling.json"
GOLD_ROWS = 16_530_619
TREES, DEPTH = 10, 5
TOTAL_EXECUTOR_MB, DRIVER_MEMORY = 4096, "1500m"
BASE_PARTITIONS, BASE_EXECUTORS, CORES_PER_EXECUTOR = 48, 4, 2
EXPERIMENT_SCALE = 4  # Partition and executor tests run at 4x; scale_4x is their 48-partition, 4-executor baseline.


def config(name, experiment, scale=EXPERIMENT_SCALE, partitions=BASE_PARTITIONS, executors=BASE_EXECUTORS):
    return dict(name=name, experiment=experiment, scale=scale, partitions=partitions, executors=executors)


CONFIGS = [
    config("scale_1x", "data size", scale=1),
    config("scale_4x", "data size", scale=4),
    config("scale_16x", "data size", scale=16),
    config("partitions_12", "partitions", partitions=12),
    config("partitions_192", "partitions", partitions=192),
    config("executors_1", "executors", executors=1),
    config("executors_2", "executors", executors=2),
]


def session(name):
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName(name).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def prepare():
    """Write the 1x and 4x subsets and a copy of the full table to the container's own disk."""
    from pyspark.sql import functions as F
    spark = session("bda501-benchmark-prepare")
    gold = spark.read.parquet(GOLD)
    for scale in (1, 4, 16):
        subset = gold.filter(F.pmod(F.hash("search_id"), F.lit(16)) < scale)
        subset.repartition("snapshot_date").write.mode("overwrite").partitionBy("snapshot_date") \
            .parquet(f"{WORK}/gold_{scale}x")
        print(f"subset {scale}x: {spark.read.parquet(f'{WORK}/gold_{scale}x').count()} rows", flush=True)
    spark.stop()


def spill_totals(spark):
    """Sum shuffle and spill bytes over all stages from Spark's own monitoring API."""
    context = spark.sparkContext
    url = f"{context.uiWebUrl}/api/v1/applications/{context.applicationId}/stages"
    with urllib.request.urlopen(url, timeout=60) as response:
        stages = json.load(response)
    return {
        "shuffle_write_gb": sum(s["shuffleWriteBytes"] for s in stages) / 1e9,
        "spill_memory_gb": sum(s["memoryBytesSpilled"] for s in stages) / 1e9,
        "spill_disk_gb": sum(s["diskBytesSpilled"] for s in stages) / 1e9,
    }


def run_once(scale, partitions):
    from pyspark.sql import functions as F
    from train_price_model import gbt, preparation_pipeline

    spark = session(f"bda501-benchmark-{scale}x-{partitions}p")
    spark.conf.set("spark.sql.shuffle.partitions", partitions)
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")
    df = spark.read.parquet(f"{WORK}/gold_{scale}x")
    rows = df.count()
    train = df.filter(F.col("split") == "train")
    held_out = df.filter(F.col("split") != "train")
    train_rows, held_out_rows = train.count(), held_out.count()
    output = f"{WORK}/output"
    result = {"rows": rows, "train_rows": train_rows, "held_out_rows": held_out_rows}

    t = time.time()
    folded = df.withColumn("fold", F.pmod(F.hash("search_id"), F.lit(5)))
    train_part = folded.filter(F.col("split") == "train")
    totals = train_part.groupBy("id").agg(F.sum("label").alias("s"), F.count(F.lit(1)).alias("n"))
    by_fold = train_part.groupBy("id", "fold").agg(F.sum("label").alias("fs"), F.count(F.lit(1)).alias("fn"))
    used_rows = F.coalesce(F.col("n"), F.lit(0)) - F.coalesce(F.col("fn"), F.lit(0))
    used_sum = F.coalesce(F.col("s"), F.lit(0.0)) - F.coalesce(F.col("fs"), F.lit(0.0))
    out = (folded.join(totals, "id", "left").join(by_fold, ["id", "fold"], "left")
           .withColumn("hotel_oof_mean", (used_sum + 10 * F.col("destination_mean")) / (used_rows + 10))
           .drop("s", "n", "fs", "fn"))
    out.repartition("snapshot_date").write.mode("overwrite").partitionBy("snapshot_date").parquet(output)
    result["etl_s"] = time.time() - t
    written = spark.read.parquet(output).count()
    if written != rows:
        raise RuntimeError(f"ETL wrote {written} rows, expected {rows}")

    t = time.time()
    prep = preparation_pipeline().fit(train)
    vectors = prep.transform(train).select("features", "label").repartition(partitions)
    model = gbt(DEPTH).setMaxIter(TREES).fit(vectors)
    result["fit_s"] = time.time() - t

    t = time.time()
    scored = model.transform(prep.transform(held_out).select("features", "price_usd").repartition(partitions))
    result["mae"] = scored.agg(F.avg(F.abs(F.exp("prediction") - F.col("price_usd")))).first()[0]
    result["score_s"] = time.time() - t

    result["total_s"] = result["etl_s"] + result["fit_s"] + result["score_s"]
    result["etl_rows_per_s"] = rows / result["etl_s"]
    result["fit_rows_per_s"] = train_rows / result["fit_s"]
    result["score_rows_per_s"] = held_out_rows / result["score_s"]
    result["input_partitions"] = df.rdd.getNumPartitions()
    result["master"] = spark.sparkContext.master
    result.update(spill_totals(spark))
    print("RESULT " + json.dumps(result), flush=True)
    spark.stop()
    shutil.rmtree(output, ignore_errors=True)


def submit(args_after_script, master, driver_memory, extra=()):
    command = ["spark-submit", "--master", master, "--driver-memory", driver_memory, *extra, __file__, *args_after_script]
    return subprocess.run(command, capture_output=True, text=True)


def run_config(c):
    memory = TOTAL_EXECUTOR_MB // c["executors"]
    master = f"local-cluster[{c['executors']},{CORES_PER_EXECUTOR},{memory}]"
    started = time.time()
    done = submit(["--worker", "--scale", str(c["scale"]), "--partitions", str(c["partitions"])], master, DRIVER_MEMORY,
                  ["--conf", f"spark.executor.memory={memory}m"])
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    if done.returncode != 0 or not lines:
        print(done.stderr[-3000:], flush=True)
        return {"failed": True, "return_code": done.returncode, "wall_s": time.time() - started}
    return {**json.loads(lines[-1][len("RESULT "):]), "wall_s": time.time() - started}


def summarise(c, runs):
    good = [r for r in runs if not r.get("failed")]
    summary = {**c, "cores": c["executors"] * CORES_PER_EXECUTOR, "executor_mb": TOTAL_EXECUTOR_MB // c["executors"],
               "repeats": len(runs), "failed_runs": len(runs) - len(good), "runs": runs}
    for key in ("rows", "train_rows", "held_out_rows", "etl_s", "fit_s", "score_s", "total_s", "etl_rows_per_s",
                "fit_rows_per_s", "score_rows_per_s", "mae", "spill_disk_gb", "shuffle_write_gb"):
        if good:
            summary[key] = statistics.median(r[key] for r in good)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--partitions", type=int, default=BASE_PARTITIONS)
    parser.add_argument("--configs", nargs="*")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    if args.worker:
        run_once(args.scale, args.partitions)
        return
    if args.prepare:
        prepare()
        return

    if not os.path.exists(f"{WORK}/gold_16x/_SUCCESS"):
        done = submit(["--prepare"], "local[*]", "5g")
        print(done.stdout, flush=True)
        if done.returncode != 0:
            sys.exit(done.stderr[-3000:])

    chosen = [c for c in CONFIGS if not args.configs or c["name"] in args.configs]
    report_path = REPORT if not args.configs else f"{WORK}/partial_report.json"
    summaries = []
    for c in chosen:
        runs = []
        for repeat in range(args.repeats):
            result = run_config(c)
            runs.append(result)
            print(f"{c['name']} run {repeat + 1}: " + ("FAILED" if result.get("failed") else
                  f"etl {result['etl_s']:.0f}s, fit {result['fit_s']:.0f}s, score {result['score_s']:.0f}s, "
                  f"spill {result['spill_disk_gb']:.2f} GB, MAE ${result['mae']:.2f}"), flush=True)
        summaries.append(summarise(c, runs))
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"trees": TREES, "depth": DEPTH, "total_executor_mb": TOTAL_EXECUTOR_MB, "driver_memory": DRIVER_MEMORY,
                       "results": summaries}, f, indent=2)

    if not args.configs:
        shutil.rmtree(WORK, ignore_errors=True)
    print("Finished", flush=True)


if __name__ == "__main__":
    main()
