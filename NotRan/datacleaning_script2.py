from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    sqrt,
    pow,
    to_timestamp,
    when,
    abs,
    first,
    last,
    count,
    isnull,
)
from pyspark.sql.window import Window

# Set Up
spark = SparkSession.builder.appName("askdlakls;da;sjdka;sl").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
data_traffic = spark.read.option("header", True).csv("us_congestion_2016_2022.csv")
# Turn Times into timestamp
data_traffic = data_traffic.withColumn(
    "StartTime", to_timestamp(col("StartTime"), "yyyy-MM-dd'T'HH:mm:ss.SSSXXX")
)
data_weather = data_weather.withColumn(
    "StartTime(UTC)", to_timestamp(col("StartTime(UTC)"), "yyyy-MM-dd HH:mm:ss")
)
# Remove rows with NULL's
data_traffic = data_traffic.filter(col("State").isNotNull())
data_traffic = data_traffic.filter(col("StartTime").isNotNull())
# Ensure data is sorted
data_traffic = data_traffic.orderBy("StartTime")

# Forward and Backward fill
window_spec = Window.partitionBy("State").orderBy("StartTime")
data_traffic = data_traffic.withColumn(
    "Weather_Conditions_Forward",
    when(
        col("Weather_Conditions").isNull(),
        last("Weather_Conditions", ignorenulls=True).over(window_spec),
    ).otherwise(col("Weather_Conditions")),
)
data_traffic = data_traffic.withColumn(
    "Weather_Conditions_Backward",
    when(
        col("Weather_Conditions").isNull(),
        first("Weather_Conditions", ignorenulls=True).over(window_spec),
    ).otherwise(col("Weather_Conditions_Forward")),
)

# Drop temporary columns and rename
data_traffic = data_traffic.drop("Weather_Conditions")
data_traffic = data_traffic.withColumnRenamed("Weather_Conditions_Backward", "Weather_Conditions")
data_traffic = data_traffic.drop("Weather_Conditions_Forward")

# Save the file
data_traffic.write.parquet("filled_traffic", mode="overwrite")
