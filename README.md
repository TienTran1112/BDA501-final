# Booking Hotels: Prices & Availability

BDA501 final project, Group 5: Nguyen Ngoc Chien, Huynh Hieu Nhan, Tran Thuy Tien.

A batch big data pipeline for hotel prices. It turns 16.5 million Expedia search-result rows into a Parquet lake, trains a price model with Spark ML, serves the results from Cassandra and MongoDB, and shows them on a Streamlit dashboard. Everything except the first conversion step runs in Docker on one machine.

The proposal (`Group 5 - Booking Hotels - Proposal.pptx`) targets live Booking.com data for five Vietnamese cities, collected through an Apify actor. This build uses the public Expedia ICDM 2013 dataset (November 2012 to June 2013) instead, so it is a historical test of the pipeline and not a view of current prices.

## Status

| Proposal item | State |
|---|---|
| Raw data to Parquet (Silver) | Done, Expedia only |
| Gold features and Spark ML price model | Done |
| Spark scaling tests (1x, 4x, 16x, partitions, executors) | Done |
| Cassandra (3 nodes, RF=3) and MongoDB (3-member replica set) | Done, loaded and verified against the lake |
| Serving load test (Q1 to Q5 at 1, 4, 16 clients) | Done |
| Dashboard | Done, see the gaps in [Data limits](#data-limits) |
| Apify Booking.com collection | Not run |
| Kafka and Airflow | Not built |
| Sold-out model, "book now or wait" hint, real GPS price map | Not built; the first and last need data Expedia does not have |

## Architecture

```text
data/raw/{train,test}.csv              raw copy, 16,540,159 rows
        |  convert_expedia.py (PyArrow, on the host)
        v
data/lake/silver/expedia               Parquet by snapshot_date, 9,540 rows flagged as price outliers
        |  build_gold_expedia.py (Spark)
        v
data/lake/gold/expedia_features        16,530,619 rows with leakage-safe features
        |  train_price_model.py (Spark ML)
        v
data/lake/models/expedia_price_gbt, data/lake/gold/expedia_price_predictions

Silver --> load_serving.py --> Cassandra: price_history, cheapest_by_city
                           --> MongoDB:   hotels, destination_weekly
       --> build_dashboard_tables.py --> MongoDB: destination_catalog, price_alerts

Cassandra + MongoDB --> Streamlit dashboard at http://localhost:8550
```

The dashboard reads only from the two databases, never from the lake.

## Project layout

```text
docker-compose.yml            all services; Cassandra, MongoDB and the dashboard are Compose profiles
docker/spark/                 Python 3.11, Java 17, PySpark 4.2.0, plus Spark defaults in conf/
docker/dashboard/             Streamlit 1.65.0 image
src/convert_expedia.py        CSV to Silver
src/spark_smoke_test.py       checks the Spark setup end to end
src/build_gold_expedia.py     Gold features, then verification
src/train_price_model.py      GBT on log price, compared with baselines
src/benchmark_spark.py        scaling tests
src/load_serving.py           loads and verifies Cassandra and MongoDB
src/loadtest_serving.py       latency and throughput for Q1 to Q5
src/build_dashboard_tables.py precomputed MongoDB collections for the dashboard
src/dashboard/                Streamlit app: streamlit_app.py, serving.py (queries), charts.py (Altair), app_pages/
data/raw/                     the Expedia CSVs from Kaggle
data/lake/                    Silver, Gold, model
data/reports/                 JSON and log output of every run below
```

## Requirements

- Docker Desktop with about 7.6 GB of memory for containers, and about 10 GB of disk for the raw files, the lake and the database volumes
- Docker subnet `172.22.0.0/16` free, because Compose uses it for this project's network
- Python 3 with `numpy` and `pyarrow` on the host, only for the CSV conversion
- The Expedia data from the Kaggle competition [expedia-personalized-sort](https://www.kaggle.com/c/expedia-personalized-sort/data), placed in `data/raw/`. Kaggle requires accepting the competition rules, and those rules apply to sharing the files.
- Internet access the first time images are built and the first time a Spark job uses `--packages`, to download the MongoDB connector

## Services

| Service | Containers | Memory limit | Access |
|---|---|---|---|
| Cassandra 5.0.9, RF=3 | `bda501-booking-cassandra1` to `3` | 1600 MB each, heap 1024 MB | Compose network only |
| MongoDB 8.0, replica set `rs0` | `bda501-booking-mongo1` to `3` | 700 MB each, WiredTiger cache 0.25 GB | Compose network only |
| Spark 4.2.0 (`local[*]`, driver 5 GB, 48 shuffle partitions) | `bda501-booking-spark` | 7 GB | Spark UI at http://localhost:4050 while a job runs |
| Streamlit dashboard | `bda501-booking-dashboard` | 600 MB | http://localhost:8550, this machine only |

The Spark-Cassandra connector stops at Spark 3.5, so `load_serving.py` writes to Cassandra with the Python driver. MongoDB uses the Spark connector.

The Compose project name is fixed (`bda501-booking`), so the database volumes survive moving the project folder. Run `up -d` from the new location to remount the code and data. The three Cassandra nodes have fixed IP addresses because each node's address is recorded in the ring, and a node will not start on an address recorded for another node.

## Run the dashboard

Run every command from the `BDA501-final` folder. If the databases are already loaded on this machine (the data lives in Docker volumes):

```bash
docker compose --profile cassandra --profile mongo --profile dashboard up -d --wait
```

The command returns when the containers are healthy. Then open http://localhost:8550. It also starts the idle Spark container. The first dashboard query after a start is slower while Cassandra warms up.

| Page | Contents |
|---|---|
| Price monitor | Pick a destination, check-in date and date the price was seen. Shows the 20 cheapest hotels with stars, review score, chain flag and June 2013 price range. Selecting a row shows that hotel's full price history. |
| Trends and alerts | Weekly median price and promotion rate for a destination, with alert weeks marked; median by star rating and by check-in week; the 200 price alerts across all destinations, where selecting a row opens that destination. |
| Nearby hotels | Hotels around a chosen hotel, within a radius (1 to 15 km, default 10) and above a review score (default 4 of 5), on synthetic coordinates. |

Each page shows the measured time of its Q1 to Q5 queries, which are not cached. Only the destination list is cached, for one hour. If the databases cannot be reached, the dashboard shows the command to start them. The footer on every page lists what the data can and cannot show.

Python changes in `src/dashboard/app_pages/` reload on their own. After changing `serving.py` or `charts.py`, restart the container:

```bash
docker compose --profile dashboard restart dashboard
```

To stop everything and keep the data:

```bash
docker compose --profile cassandra --profile mongo --profile dashboard down
```

Do not add `-v` to that command. It deletes the database volumes, and loading them again takes about 2.5 hours.

## Rebuild from the raw data

Run these in order. Spark jobs need most of the memory, so keep the other services stopped unless a step needs them.

**1. Convert the CSVs to Silver (on the host).** The source files stay untouched. Rows are not deleted: prices of 0 or less and above 5,000 USD are flagged with `price_outlier`.

```bash
python src/convert_expedia.py data/raw data/lake/silver/expedia
```

**2. Start Spark and check it.**

```bash
docker compose up -d spark
```

```bash
docker compose exec spark spark-submit src/spark_smoke_test.py
```

**3. Build Gold and train the model.** Training took about 27 minutes on the full data. Add `--fraction 0.0625` to `train_price_model.py` for a dry run that writes only to `/tmp`.

```bash
docker compose exec spark spark-submit src/build_gold_expedia.py
```

```bash
docker compose exec spark spark-submit src/train_price_model.py
```

**4. Optional: run the scaling tests.** Allow one to two hours. Use `--configs scale_1x --repeats 1` for a single short run.

```bash
docker compose exec spark python src/benchmark_spark.py
```

**5. Load Cassandra.** The two tables took about 2.5 hours together. The script creates the keyspace and tables. Add `--fraction 0.0625` to the load and verify steps for a 1/16 pilot.

```bash
docker compose --profile cassandra up -d --wait
```

```bash
docker compose exec spark spark-submit --master "local[6]" --driver-memory 2g src/load_serving.py cassandra
```

```bash
docker compose exec spark spark-submit --master "local[6]" --driver-memory 2g src/load_serving.py verify-cassandra
```

**6. Stop Cassandra, then load MongoDB and build the dashboard tables.** The script initiates the replica set on first use.

```bash
docker compose --profile cassandra stop cassandra1 cassandra2 cassandra3
```

```bash
docker compose --profile mongo up -d --wait
```

```bash
docker compose exec spark spark-submit --master "local[6]" --driver-memory 2g --packages org.mongodb.spark:mongo-spark-connector_2.13:11.1.0 src/load_serving.py mongo
```

```bash
docker compose exec spark spark-submit --master "local[6]" --driver-memory 2g --packages org.mongodb.spark:mongo-spark-connector_2.13:11.1.0 src/load_serving.py verify-mongo
```

```bash
docker compose exec spark spark-submit --master "local[6]" --driver-memory 3g --packages org.mongodb.spark:mongo-spark-connector_2.13:11.1.0 src/build_dashboard_tables.py
```

**7. Optional: run the load test,** with one database profile started at a time. The `keys` step samples the query keys from Silver. `--queries Q1` limits a run to some queries, and `--suffix _x` adds to the report name so a repeat keeps the first report.

```bash
docker compose exec spark spark-submit --master "local[4]" --driver-memory 2g src/loadtest_serving.py keys
```

```bash
docker compose exec spark python src/loadtest_serving.py cassandra
```

```bash
docker compose exec spark python src/loadtest_serving.py mongo
```

Cassandra is limited to 1600 MB in Compose so bulk loading fits in memory. For read tests, raise it to leave room for the file cache. The change is lost when the containers are recreated, and the limit is back at 1600 MB until you run it again.

```bash
docker update --memory 2000m --memory-swap 4000m bda501-booking-cassandra1 bda501-booking-cassandra2 bda501-booking-cassandra3
```

## Storage

| Data | Size | Used by |
|---|---|---|
| `data/raw/` | 4.1 GB: `train.csv` 2.36 GB, `test.csv` 1.50 GB, three benchmark CSVs 93 MB each | CSV conversion only |
| `data/lake/silver/expedia` | 270 MB, 242 days | Gold build, serving load, dashboard tables, smoke test, load test keys |
| `data/lake/gold/expedia_features` | 497 MB, 242 days | Training, scaling tests |
| `data/lake/gold/expedia_price_predictions` | 182 MB, 6,941,849 rows | Written by training, read by nothing afterwards |
| `data/lake/models/expedia_price_gbt` | 2.2 MB | The trained model |
| Cassandra volumes | About 1.2 GB per node | Dashboard |
| MongoDB volumes | About 315 MB per member | Dashboard |

The dashboard keeps working if `data/lake` is removed, but the rebuild steps need it. To free space, delete a whole dataset rather than some of its `snapshot_date` folders, because a partial Silver or Gold set gives wrong results without any error. The predictions are the safest to delete, and they come back when the model is retrained.

## Price model

Gradient-boosted trees on log price: 50 trees, depth 9 chosen on the April validation set (MAE 35.06 at depth 5, 34.39 at 7, 34.32 at 9). The data is split by search date: train November to March, validation April, test May and June. Destination, country and hotel averages come from the train window only, and on train rows they are out-of-fold over 5 folds, so no row's own price feeds its own features. Outliers and outcome columns are excluded.

| Test set (4,790,852 rows) | MAE (USD) | RMSE (USD) | Median absolute % error | R² on log price |
|---|---|---|---|---|
| GBT model | 38.74 | 93.25 | 16.8% | 0.676 |
| Hotel's average price | 40.35 | | 17.6% | 0.648 |
| Destination's average price | 60.56 | | 30.1% | 0.264 |
| Overall median | 70.79 | | 35.7% | -0.023 |

The model improves on the hotel's own average by only about 4% in MAE, and that average is its strongest feature (importance 0.59). Hotels with no training history are much harder: MAE is 58.78 on 55,928 rows, against 38.50 for hotels seen in training. Error also grows with lead time, from 32.48 at 0 to 1 days to 45.11 at 91 days or more.

## Spark scaling

Four executors, 8 cores, 48 partitions, average of 2 runs. The benchmark model is deliberately small (10 trees, depth 5).

| Data | Rows | ETL | Train | Score | Total |
|---|---|---|---|---|---|
| 1x | 1.04M | 29 s | 74 s | 4.5 s | 108 s |
| 4x | 4.13M | 57 s | 115 s | 11 s | 183 s |
| 16x | 16.53M | 109 s | 584 s | 28 s | 721 s |

| Setting (4x data) | Total |
|---|---|
| 12 partitions | 160 s |
| 48 partitions | 170 s |
| 192 partitions | 239 s |
| 1 executor, 2 cores | 158 s |
| 2 executors, 4 cores | 144 s |
| 4 executors, 8 cores | 170 s |

Training is the bottleneck: from 4x to 16x it took 5.1 times as long for 4 times the data, and only 16x spilled to disk (about 1.6 GB). Adding executors gave no real gain. The same 4x baseline measured 183 s in the data-size runs and 170 s in the partition and executor runs, and individual 4x runs varied from 167 s to 198 s, so small differences are not meaningful.

All executors share one machine with no network between them and 4 GB of memory in total, so this is not a real cluster test.

## Serving model

| Query | Store | Question | Key or index |
|---|---|---|---|
| Q1 | Cassandra `price_history` | Every recorded price for one hotel and check-in date | `((hotel_id, checkin_date), snapshot_ts DESC, search_id)` |
| Q2 | Cassandra `cheapest_by_city` | 20 cheapest hotels for a destination, check-in and snapshot date | `((city, checkin_date, snap_date), price_night, hotel_id)` |
| Q3 | MongoDB `hotels` | Hotel profile | `_id` |
| Q4 | MongoDB `hotels` | Hotels within a radius with a minimum review score | 2dsphere on `location`, plus `review.score` |
| Q5 | MongoDB `destination_weekly` | Promotion rate and price per destination and week | `destination_id`, `week_start` |

`city` in Cassandra holds Expedia's anonymised `destination_id`. `search_id` is in the `price_history` key so that rows with the same timestamp do not overwrite each other. `cheapest_by_city` keeps each hotel's lowest price per key. Q5 replaces the proposal's sold-out rate by district, because Expedia records neither. Cassandra reads use `LOCAL_QUORUM`, and MongoDB reads go to the primary.

| Table or collection | Rows | Load time | Check against the lake |
|---|---|---|---|
| `price_history` | 16,530,619 | 4,785 s (3,455 rows/s) | 3,200 partitions, 17,944 rows, 0 differ |
| `cheapest_by_city` | 16,064,292 | 4,393 s (3,657 rows/s) | 3,199 partitions, 94,906 rows, 0 differ |
| `hotels` | 140,812 | 29 s | 3,000 recomputed from raw rows, 0 differ |
| `destination_weekly` | 192,476 | 24 s | All documents, 0 differ |
| `destination_catalog` | 339 | 136 s, together with `price_alerts` | Not checked separately |
| `price_alerts` | 200 | See above | Not checked separately |

Cassandra can only be read by full partition key, so `destination_catalog` lists keys that exist for the dashboard's pickers, along with the star and check-in medians. It covers the busiest destinations and every destination with an alert, so the pickers offer 339 destinations and not all of them. `price_history` uses `TimeWindowCompactionStrategy` with 7-day windows. MongoDB also has indexes on `hotels.destination_id`, `price_alerts (destination_id, week_start)`, `price_alerts.ratio` and `destination_catalog.rows`.

## Serving load test

Closed loop from separate client processes, 10 s warm-up, 45 s measured, 20,000 sampled keys, no errors. Cassandra was at 2000 MB per node.

| Query | p95 at 1 client | p95 at 4 clients | p95 at 16 clients | Requests/s at 16 clients |
|---|---|---|---|---|
| Q1 | 6.88 ms | 11.1 ms | 24.99 ms | 1,132 |
| Q2 | 5.91 ms | 9.56 ms | 19.81 ms | 1,399 |
| Q3 | 1.25 ms | 1.85 ms | 10.94 ms | 4,235 |
| Q4 | 2.75 ms | 4.65 ms | 14.72 ms | 2,417 |
| Q5 | 1.44 ms | 2.72 ms | 9.00 ms | 3,868 |

Memory matters for Cassandra. At the Compose limit of 1600 MB, the file cache held only about 17 MB:

| Query at 1600 MB | p95 at 1 client | p95 at 4 clients | p95 at 16 clients |
|---|---|---|---|
| Q1 | 52.88 ms | 134.69 ms | 297.55 ms |
| Q2 | 8.72 ms | 13.19 ms | 39.46 ms |

Clients and all database nodes share one machine and disk. The load test's Q4 uses 2 km and a review score of at least 4 of 5, on synthetic coordinates, and about 35.5% of those searches found no hotel. An empty Q4 result is not counted as an error.

## Reports

Everything is in `data/reports/`. Files with `_pilot` in the name are 1/16 trial runs.

| Topic | File | Note |
|---|---|---|
| Price model | `expedia_price_model.json`, `train_price_model.log` | |
| Scaling by data size | `spark_scaling_matched.log` | Use this one |
| Scaling by partitions and executors | `spark_scaling_recheck.log` | Use this one |
| Older scaling pass | `spark_scaling.json`, `spark_scaling.log` | First pass, which ran slower |
| Loading and verification | `serving_load.json`, `serving_load_{cassandra,mongo}.log`, `serving_verify_{cassandra,mongo}.log` | |
| Dashboard tables | `dashboard_tables.json`, `dashboard_tables.log` | |
| Q1 and Q2 at 2000 MB | `serving_loadtest_q{1,2}_2000mb.json` and `.log` | |
| Q3 to Q5, and Q1 and Q2 at 1600 MB | `serving_loadtest.json`, `serving_loadtest_{cassandra,mongo}.log` | |

## Data limits

- **Price basis:** Expedia's displayed `price_usd`. On 276,370 booked rows it behaves as per room per night, usually before taxes and fees, but Expedia notes the display basis can differ by country.
- **Implausible prices remain:** only prices of 0 or less and above 5,000 USD are flagged. Single prices under 1 USD can appear as a hotel's lowest price, and some June maximums are inflated.
- **No hotel locations:** coordinates are synthetic and marked `location_source: "synthetic"`. The Nearby page shows that the geospatial query works, and its distances mean nothing. It defaults to 10 km where the proposal uses 2 km, because the positions are spread thinly.
- **No sold-out data:** `is_sold_out` stays empty, and promotion rate is shown instead.
- **No names:** destinations are anonymised numbers, and there are no hotel, city or district names.
- **Review score:** Expedia's scale is 0 to 5, so the proposal's 8 of 10 becomes 4 of 5. A score of 0 means unknown and is stored as null, as are star rating and historical price.
- **Weekly and star medians are approximate:** Spark's `percentile_approx` returns an actual value, for example 59.27 where the exact median is 59.555.
- **Alerts need care:** an alert is a week whose median is at least 25% above the median of the previous four weekly medians, where that week and at least three of the four each have 200 or more price rows. There are 200 alert weeks in 135 destinations. Search mix changes from week to week, so an alert may not be a real price rise. The largest, +630% at destination 4862 ($67 to $489), returned to about $54 the next week.
