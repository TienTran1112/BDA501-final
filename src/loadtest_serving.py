"""Load test for serving queries Q1 to Q5: p50, p95, p99 latency and throughput at 1, 4 and 16 concurrent clients.

Run from the project folder, with one database profile started:
    docker compose exec spark spark-submit --master "local[4]" --driver-memory 2g src/loadtest_serving.py keys
    docker compose exec spark python src/loadtest_serving.py cassandra
    docker compose exec spark python src/loadtest_serving.py mongo
Add --fraction 0.0625 to all three after a 1/16 pilot load.

Queries
    Q1  Cassandra  full price history of one hotel for one check-in date
    Q2  Cassandra  20 cheapest hotels in one destination, for one check-in and one snapshot date
    Q3  MongoDB    one hotel profile by _id
    Q4  MongoDB    hotels within 2 km and review score at least 4 of 5 (the proposal's 8 of 10), nearest 50
    Q5  MongoDB    weekly promotion rate and price for one destination (substitute for sold-out rate, see load_serving.py)

Method
    Closed loop: each client is its own process, sends one request, waits for all rows, then sends the next.
    Q1, Q2, Q3 and Q5 keys are drawn from random Silver rows, so busy hotels and destinations come up more often.
    Q4 centres are the coordinates of random hotels, and those coordinates are synthetic.
    Cassandra reads at LOCAL_QUORUM and MongoDB reads from the primary.
    Each run has a 10 s warm-up that is not measured, then 45 s that is.
    An empty result counts as an error except on Q4, where finding no hotel is a valid answer.

Limits: the clients and all database nodes share one machine and one disk, so clients compete with the servers for CPU,
and there is no real network. Latency includes Python client time.
"""
import argparse
import datetime
import itertools
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KEY_FILE = "/tmp/serving_keys.json"
REPORTS = "/workspace/data/reports"
KEY_COUNT = 20_000
WARMUP_S, MEASURE_S = 10, 45
CONCURRENCY = (1, 4, 16)
QUERIES = {"cassandra": ("Q1", "Q2"), "mongo": ("Q3", "Q4", "Q5")}
MIN_REVIEW_SCORE, RADIUS_M, NEAREST = 4, 2_000, 50


def make_keys(fraction):
    from pyspark.sql import SparkSession
    from load_serving import silver_rows
    spark = SparkSession.builder.appName("bda501-serving-keys").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    silver = silver_rows(spark, fraction).select("id", "checkin_date", "destination_id", "snapshot_date").cache()
    rows = silver.sample(fraction=min(1.0, KEY_COUNT / silver.count()), seed=501).collect()
    keys = {"Q1": [[r["id"], r["checkin_date"].isoformat()] for r in rows],
            "Q2": [[r["destination_id"], r["checkin_date"].isoformat(), r["snapshot_date"].isoformat()] for r in rows],
            "Q3": [r["id"] for r in rows],
            "Q5": [r["destination_id"] for r in rows]}
    with open(KEY_FILE, "w", encoding="utf-8") as f:
        json.dump(keys, f)
    print(f"{len(rows)} keys written", flush=True)
    spark.stop()


def cassandra_query(name):
    from load_serving import KEYSPACE, cassandra_session
    _, session = cassandra_session(KEYSPACE)
    day = datetime.date.fromisoformat
    if name == "Q1":
        statement = session.prepare("SELECT snapshot_ts, price_night, currency, is_sold_out FROM price_history "
                                    "WHERE hotel_id=? AND checkin_date=?")
        return lambda key: len(list(session.execute(statement, (key[0], day(key[1])))))
    statement = session.prepare("SELECT hotel_id, price_night FROM cheapest_by_city "
                                "WHERE city=? AND checkin_date=? AND snap_date=? LIMIT 20")
    return lambda key: len(list(session.execute(statement, (key[0], day(key[1]), day(key[2])))))


def mongo_query(name):
    from pymongo import MongoClient
    from load_serving import DATABASE, MONGO_URI
    database = MongoClient(MONGO_URI)[DATABASE]
    if name == "Q3":
        return lambda key: int(database.hotels.find_one({"_id": key}) is not None)
    if name == "Q4":
        def nearby(key):
            near = {"$near": {"$geometry": {"type": "Point", "coordinates": key}, "$maxDistance": RADIUS_M}}
            return len(list(database.hotels.find({"location": near, "review.score": {"$gte": MIN_REVIEW_SCORE}}).limit(NEAREST)))
        return nearby
    return lambda key: len(list(database.destination_weekly.find({"destination_id": key}).sort("week_start", 1)))


def client(database, query, keys, seed, barrier, results):
    """One closed-loop client. Sends back only requests started inside the measured window."""
    try:
        run = {"cassandra": cassandra_query, "mongo": mongo_query}[database](query)
        keys = list(keys)
        random.Random(seed).shuffle(keys)
        barrier.wait(300)
        begin = time.perf_counter() + WARMUP_S
        end = begin + MEASURE_S
        latencies, rows, errors, empty, first_error = [], 0, 0, 0, None
        for key in itertools.cycle(keys):
            started = time.perf_counter()
            if started >= end:
                break
            try:
                found, failed = run(key), False
            except Exception as error:
                found, failed = 0, True
                first_error = first_error or repr(error)[:300]
            finished = time.perf_counter()
            if started >= begin:
                errors += failed
                if not failed:
                    latencies.append(finished - started)
                    rows += found
                    empty += found == 0
        results.put({"latencies": latencies, "rows": rows, "errors": errors, "empty": empty, "first_error": first_error})
    except Exception as error:
        results.put({"crashed": repr(error)[:500]})


def run_load(database, query, keys, clients):
    context = mp.get_context("spawn")
    barrier, results = context.Barrier(clients), context.Queue()
    workers = [context.Process(target=client, args=(database, query, keys, 501 + i, barrier, results)) for i in range(clients)]
    for worker in workers:
        worker.start()
    parts = [results.get(timeout=600) for _ in workers]
    for worker in workers:
        worker.join()
    crashed = [p["crashed"] for p in parts if "crashed" in p]
    if crashed:
        raise RuntimeError(f"{query}: {crashed[0]}")

    latencies = np.concatenate([np.array(p["latencies"], dtype=float) for p in parts]) * 1000
    ok, rows = len(latencies), sum(p["rows"] for p in parts)
    empty, errors = sum(p["empty"] for p in parts), sum(p["errors"] for p in parts)
    result = {"clients": clients, "requests": ok, "throughput_per_s": round(ok / MEASURE_S, 1), "errors": errors,
              "empty": empty, "empty_share": round(empty / ok, 4) if ok else None,
              "rows_per_request": round(rows / ok, 2) if ok else None,
              "first_error": next((p["first_error"] for p in parts if p["first_error"]), None)}
    for label, q in (("p50_ms", 50), ("p95_ms", 95), ("p99_ms", 99)):
        result[label] = round(float(np.percentile(latencies, q)), 2) if ok else None
    print(f"{query} {clients:>2} clients: {result['throughput_per_s']:>8.1f}/s  p50 {result['p50_ms']} ms  "
          f"p95 {result['p95_ms']} ms  p99 {result['p99_ms']} ms  rows/request {result['rows_per_request']}  "
          f"errors {errors}  empty {empty}", flush=True)
    return result


def q4_centres():
    from pymongo import MongoClient
    from load_serving import DATABASE, MONGO_URI
    connection = MongoClient(MONGO_URI)
    documents = connection[DATABASE].hotels.aggregate([{"$sample": {"size": KEY_COUNT}}, {"$project": {"location.coordinates": 1}}])
    centres = [d["location"]["coordinates"] for d in documents]
    connection.close()
    return centres


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["keys", "cassandra", "mongo"])
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--queries", nargs="*", help="run only these queries, for example Q1")
    parser.add_argument("--suffix", default="", help="added to the report name, so a repeat keeps the first report")
    args = parser.parse_args()
    if args.command == "keys":
        make_keys(args.fraction)
        return

    with open(KEY_FILE, encoding="utf-8") as f:
        keys = json.load(f)
    if args.command == "mongo":
        keys["Q4"] = q4_centres()

    path = f"{REPORTS}/serving_loadtest{'' if args.fraction >= 1 else '_pilot'}{args.suffix}.json"
    report = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    report["settings"] = {"fraction": args.fraction, "warmup_s": WARMUP_S, "measure_s": MEASURE_S, "key_count": KEY_COUNT}
    for query in args.queries or QUERIES[args.command]:
        report[query] = [run_load(args.command, query, keys[query], clients) for clients in CONCURRENCY]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print("Finished", flush=True)


if __name__ == "__main__":
    main()
