"""Convert the Expedia ICDM 2013 CSVs into the Silver layer: Parquet partitioned by snapshot_date.

Usage: python convert_expedia.py <raw_dir> <silver_dir>

The source CSVs stay untouched and act as the raw copy. Silver differs from them as follows:
  * columns are renamed towards the project schema (id, snapshot_date, checkin_date, lead_time_days, ...)
  * train and test are combined; source_split records the origin, test has null position/click/booking
  * no rows are deleted; price_outlier marks price_usd <= 0 or > MAX_PLAUSIBLE_PRICE
  * star_rating, review_score and log_historical_price: 0 means "unknown", so it becomes null
  * dropped as mostly empty: visitor_hist_*, srch_query_affinity_score, comp1..8_*

price_usd is per room per night, usually before taxes and fees: on 276,370 booked rows,
gross_bookings_usd / (price_usd * rooms) has a median of 1.14 at 1 night and 3.43 at 3 nights.
Duplicates are defined by (search_id, id). The same hotel can legitimately appear several times per
(id, checkin_date, snapshot_date), once for each separate search.
"""
import math
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.dataset as ds

MAX_PLAUSIBLE_PRICE = 5000.0
EXPECTED_ROWS = {"train": 9_917_530, "test": 6_622_629}

SILVER = pa.schema([
    ("snapshot_date", pa.date32()),
    ("snapshot_ts", pa.timestamp("s")),
    ("search_id", pa.int64()),
    ("id", pa.int64()),
    ("source_split", pa.string()),
    ("checkin_date", pa.date32()),
    ("lead_time_days", pa.int32()),
    ("day_of_week", pa.int8()),  # check-in day, ISO: 1 = Monday, 7 = Sunday
    ("nights", pa.int32()),
    ("price_usd", pa.float64()),
    ("price_outlier", pa.bool_()),
    ("currency", pa.string()),
    ("has_promotion", pa.bool_()),
    ("country_id", pa.int32()),
    ("destination_id", pa.int32()),
    ("star_rating", pa.float32()),
    ("review_score", pa.float32()),
    ("is_brand", pa.bool_()),
    ("location_score1", pa.float32()),
    ("location_score2", pa.float32()),
    ("log_historical_price", pa.float32()),
    ("adults", pa.int16()),
    ("children", pa.int16()),
    ("rooms", pa.int16()),
    ("saturday_night", pa.bool_()),
    ("orig_destination_distance", pa.float64()),
    ("site_id", pa.int32()),
    ("visitor_country_id", pa.int32()),
    ("is_random_order", pa.bool_()),
    ("position", pa.int32()),
    ("clicked", pa.bool_()),
    ("booked", pa.bool_()),
    ("gross_bookings_usd", pa.float64()),
])


def zero_to_null(arr, target):
    return pc.if_else(pc.equal(arr, 0), pa.scalar(None, arr.type), arr).cast(target)


def as_bool(arr):
    return arr.cast(pa.int8()).cast(pa.bool_())


def to_silver(batch, split):
    n = batch.num_rows
    c = batch.column
    names = set(batch.schema.names)

    def optional(name, convert):
        return convert(c(name)) if name in names else None

    ts = c("date_time").cast(pa.timestamp("s"))
    snapshot = pc.cast(ts, pa.date32(), safe=False)
    lead = c("srch_booking_window").cast(pa.int32())
    checkin = pc.add(snapshot.cast(pa.int32()), lead).cast(pa.date32())
    price = c("price_usd")
    columns = {
        "snapshot_date": snapshot,
        "snapshot_ts": ts,
        "search_id": c("srch_id").cast(pa.int64()),
        "id": c("prop_id").cast(pa.int64()),
        "source_split": pa.repeat(split, n),
        "checkin_date": checkin,
        "lead_time_days": lead,
        "day_of_week": pc.day_of_week(checkin, count_from_zero=False, week_start=1).cast(pa.int8()),
        "nights": c("srch_length_of_stay").cast(pa.int32()),
        "price_usd": price,
        "price_outlier": pc.or_(pc.less_equal(price, 0), pc.greater(price, MAX_PLAUSIBLE_PRICE)),
        "currency": pa.repeat("USD", n),
        "has_promotion": as_bool(c("promotion_flag")),
        "country_id": c("prop_country_id").cast(pa.int32()),
        "destination_id": c("srch_destination_id").cast(pa.int32()),
        "star_rating": zero_to_null(c("prop_starrating"), pa.float32()),
        "review_score": zero_to_null(c("prop_review_score"), pa.float32()),
        "is_brand": as_bool(c("prop_brand_bool")),
        "location_score1": c("prop_location_score1").cast(pa.float32()),
        "location_score2": c("prop_location_score2").cast(pa.float32()),
        "log_historical_price": zero_to_null(c("prop_log_historical_price"), pa.float32()),
        "adults": c("srch_adults_count").cast(pa.int16()),
        "children": c("srch_children_count").cast(pa.int16()),
        "rooms": c("srch_room_count").cast(pa.int16()),
        "saturday_night": as_bool(c("srch_saturday_night_bool")),
        "orig_destination_distance": c("orig_destination_distance"),
        "site_id": c("site_id").cast(pa.int32()),
        "visitor_country_id": c("visitor_location_country_id").cast(pa.int32()),
        "is_random_order": as_bool(c("random_bool")),
        "position": optional("position", lambda a: a.cast(pa.int32())),
        "clicked": optional("click_bool", as_bool),
        "booked": optional("booking_bool", as_bool),
        "gross_bookings_usd": optional("gross_bookings_usd", lambda a: a),
    }
    arrays = [columns[f.name] if columns[f.name] is not None else pa.nulls(n, f.type) for f in SILVER]
    return pa.record_batch(arrays, schema=SILVER)


def csv_batches(path, split, totals):
    with open(path, encoding="utf-8") as f:
        cols = f.readline().strip().split(",")
    types = {name: pa.float64() for name in cols}
    types["date_time"] = pa.string()
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024),
        convert_options=pacsv.ConvertOptions(column_types=types, null_values=["NULL", ""], strings_can_be_null=True),
    )
    for batch in reader:
        totals["rows"] += batch.num_rows
        totals["price_sum"] += pc.sum(batch.column("price_usd")).as_py()
        yield to_silver(batch, split)


def write_split(raw_dir, out_dir, split):
    totals = {"rows": 0, "price_sum": 0.0}
    ds.write_dataset(
        csv_batches(os.path.join(raw_dir, f"{split}.csv"), split, totals),
        out_dir,
        schema=SILVER,
        format="parquet",
        partitioning=ds.partitioning(pa.schema([("snapshot_date", pa.date32())]), flavor="hive"),
        basename_template=f"{split}-{{i}}.parquet",
        file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
        existing_data_behavior="overwrite_or_ignore",
        min_rows_per_group=20_000,
        max_rows_per_group=100_000,
        max_partitions=400,
    )
    print(f"written {split}: rows={totals['rows']}", flush=True)
    return totals


def verify(out_dir, source):
    silver = ds.dataset(
        out_dir, format="parquet",
        partitioning=ds.partitioning(pa.schema([("snapshot_date", pa.date32())]), flavor="hive"),
    )
    table = silver.to_table(columns=[
        "snapshot_date", "checkin_date", "lead_time_days", "search_id", "id",
        "source_split", "price_usd", "price_outlier", "day_of_week",
    ])
    ok = True

    def check(label, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {label} {detail}".rstrip())

    split = table["source_split"].to_numpy(zero_copy_only=False).astype(str)
    for name, expected in EXPECTED_ROWS.items():
        rows = int((split == name).sum())
        check(f"{name} rows match published count", rows == expected == source[name]["rows"], f"{rows}")

    price = table["price_usd"].to_numpy()
    expected_sum = source["train"]["price_sum"] + source["test"]["price_sum"]
    check("price_usd sum matches CSVs", math.isclose(float(np.nansum(price)), expected_sum, rel_tol=1e-9))
    check("price_usd has no nulls", table["price_usd"].null_count == 0)

    search = table["search_id"].to_numpy()
    prop = table["id"].to_numpy()
    key = search * 1_000_000 + prop
    check("no duplicates on (search_id, id)", len(np.unique(key)) == len(key))
    overlap = np.intersect1d(np.unique(search[split == "train"]), np.unique(search[split == "test"]))
    check("no search_id shared by train and test", len(overlap) == 0, f"shared={len(overlap)}")

    snapshot = table["snapshot_date"].cast(pa.int32()).to_numpy()
    checkin = table["checkin_date"].cast(pa.int32()).to_numpy()
    lead = table["lead_time_days"].to_numpy()
    check("checkin_date = snapshot_date + lead_time_days", bool(np.array_equal(checkin - snapshot, lead)))
    check("snapshot partitions cover 242 days", len(np.unique(snapshot)) == 242, f"days={len(np.unique(snapshot))}")
    # 1970-01-01, day 0 of date32, was a Thursday (ISO 4).
    iso_weekday = (checkin + 3) % 7 + 1
    check("day_of_week is the ISO weekday of checkin_date", bool(np.array_equal(table["day_of_week"].to_numpy(), iso_weekday)))

    outliers = int(table["price_outlier"].to_numpy(zero_copy_only=False).astype(bool).sum())
    print(f"rows flagged price_outlier: {outliers} ({100 * outliers / len(price):.3f}%)")
    files = silver.files
    print(f"columns: {len(silver.schema.names)}  files: {len(files)}  "
          f"size GB: {sum(os.path.getsize(f) for f in files) / 1024 ** 3:.2f}")
    return ok


def main():
    raw_dir, out_dir = sys.argv[1], sys.argv[2]
    source = {split: write_split(raw_dir, out_dir, split) for split in EXPECTED_ROWS}
    if not verify(out_dir, source):
        sys.exit("Verification failed")
    print("All checks passed")


if __name__ == "__main__":
    main()
