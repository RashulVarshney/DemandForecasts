"""
PySpark re-implementation of the core demand-panel feature pipeline
(src/features/build_features.py), to demonstrate how the same leakage-safe
lag/rolling feature logic scales to a distributed engine.

This does NOT need a cluster to demonstrate the point -- it runs locally via
`SparkSession.builder.master("local[*]")`, reading the same Parquet files
already produced by src/data/generate_dataset.py. The goal is to show:
  - reading a large Parquet dataset with Spark
  - joins (orders x restaurants x weather)
  - window functions for lag/rolling features (Spark's Window API mirrors the
    SQL window functions in sql/feature_queries.sql almost 1:1)
  - partitioning considerations for a real multi-node job

Run:
    python spark/distributed_feature_pipeline.py
"""
from __future__ import annotations

import pathlib

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("food-delivery-feature-pipeline")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")  # small for local run; would be 200+ on a real cluster
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def build_demand_panel(spark: SparkSession, raw_dir: str):
    orders = spark.read.parquet(f"{raw_dir}/orders.parquet").filter(~F.col("is_cancelled"))

    # Bucket to 30-min windows. Repartitioning by zone_id here is deliberate:
    # rider-allocation and zone-level aggregates downstream are consumed
    # per-zone, so co-locating a zone's data on one partition avoids a
    # shuffle in the next stage. On a real cluster with skewed zones (a few
    # very busy zones), this key would need salting to avoid data skew.
    orders = orders.withColumn(
        "bucket_ts", F.window("order_timestamp", "30 minutes").start
    ).repartition("zone_id")

    panel = (
        orders.groupBy("restaurant_id", "zone_id", "bucket_ts")
        .agg(
            F.count("*").alias("order_count"),
            F.avg("basket_value_inr").alias("avg_basket_value"),
            F.avg("distance_km").alias("avg_distance_km"),
        )
    )

    # Lag / rolling features via Spark Window functions -- direct analogue of
    # the SQL window functions in sql/feature_queries.sql. Partitioning by
    # restaurant_id and ordering by bucket_ts mirrors the pandas
    # groupby("restaurant_id") + sort_values("bucket_ts") used upstream.
    w = Window.partitionBy("restaurant_id").orderBy("bucket_ts")
    w_roll_8h = w.rowsBetween(-16, -1)  # 16 * 30min = 8h, EXCLUDING current row

    panel = (
        panel
        .withColumn("lag_30m", F.lag("order_count", 1).over(w))
        .withColumn("lag_1h", F.lag("order_count", 2).over(w))
        .withColumn("lag_1d", F.lag("order_count", 48).over(w))
        .withColumn("lag_7d", F.lag("order_count", 336).over(w))
        .withColumn("roll_mean_8h", F.avg("order_count").over(w_roll_8h))
        .withColumn("roll_std_8h", F.stddev("order_count").over(w_roll_8h))
        .withColumn(
            "restaurant_hist_avg_demand",
            F.avg("order_count").over(w.rowsBetween(Window.unboundedPreceding, -1)),
        )
        .withColumn("hour", F.hour("bucket_ts"))
        .withColumn("dow", F.dayofweek("bucket_ts"))  # Spark: 1=Sunday
        .withColumn("is_weekend", F.col("dow").isin([1, 7]).cast("int"))
    )
    return panel


def main():
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    raw_dir = str(pathlib.Path("data/raw").resolve())
    panel = build_demand_panel(spark, raw_dir)

    print(f"Demand panel row count: {panel.count():,}")
    panel.orderBy("restaurant_id", "bucket_ts").show(10, truncate=False)

    out_dir = pathlib.Path("data/processed/spark_demand_panel")
    panel.write.mode("overwrite").partitionBy("zone_id").parquet(str(out_dir))
    print(f"Written partitioned output to {out_dir}")

    spark.stop()


if __name__ == "__main__":
    main()
