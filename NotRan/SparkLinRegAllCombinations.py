from typing import List, Dict
import os
import pandas as pd
import itertools
from weather_conditions_map import good_weather, neutral_weather, bad_weather


from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, StringType, StructType, StructField
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.stat import Correlation
from time import sleep
from pyspark.ml.feature import StandardScaler
from datetime import datetime


# MOVING TO TOP OF FILE
# CHANGE PARAMS HERE
#
MAX_NUMBER_PREDICTOR = 2

# Numeric weather measurements
WEATHER_NUMERIC = [
    "Temperature(F)",
    "WindChill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "WindSpeed(mph)",
    "Precipitation(in)",
]

# we cant create dummies in spark so we need to create them manually
WEATHER_CATEGORIES = ["good", "bad", "neutral", "unknown"]
WEATHER_DUMMY_COLS = [f"weather_{cat}" for cat in WEATHER_CATEGORIES]

# All weather variables (numeric + dummies)
WEATHER_VARS = WEATHER_NUMERIC + WEATHER_DUMMY_COLS

# Traffic variables
# Remove speed_category as continuous variable is much better than discrete category
TRAFFIC_VARS = [
    "Severity",
    "Traffic Jam Duration",
    "Distance(mi)",
    "DelayFromTypicalTraffic(mins)",
    "DelayFromFreeFlowSpeed(mins)",
    "speed_numeric",
]
TRAFFIC_VARS = list(reversed(TRAFFIC_VARS))
WEATHER_VARS = list(reversed(WEATHER_VARS))

# required columns for final output
REQUIRED_COLUMNS = WEATHER_NUMERIC + TRAFFIC_VARS + WEATHER_DUMMY_COLS + ["weather_rating"]


def create_spark_session(app_name="SparkLinRegAllCombinations"):
    """Create or get existing Spark session"""
    spark = SparkSession.builder.appName("SparkLinRegAllCombinations").getOrCreate()
    return spark


def extract_speed_info(speed_str: str) -> tuple:
    if not speed_str:
        return ("Unknown", float("nan"))

    clean_str = speed_str.strip().lower()
    category_map = {
        "fast": ("fast", 60.0),
        "moderate": ("moderate", 40.0),
        "slow": ("slow", 20.0),
    }

    for key, value in category_map.items():
        if key in clean_str:
            return value

    return ("Unknown", float("nan"))


def process_file(parquet_path: str):
    print(f"Processing {parquet_path}...")

    spark = create_spark_session()
    df = spark.read.parquet(parquet_path)

    print("\n!!!!!!!!!!!!!!!!!!!!!! ORIGINAL DATA SCHEME !!!!!!!!!!!!!!!!!!!")
    df.printSchema()

    # Process timestamps
    # Calculate duration in minutes
    df = (
        df.withColumn("StartTime", F.to_timestamp("StartTime"))
        .withColumn("EndTime", F.to_timestamp("EndTime"))
        .withColumn(
            "Traffic Jam Duration",
            (F.unix_timestamp("EndTime") - F.unix_timestamp("StartTime")) / 60,
        )
    )

    df = df.filter(
        (F.col("Traffic Jam Duration") > 0) & (F.col("Traffic Jam Duration") <= 1440)
    )  # filter 24h

    # create tuple like struc
    speed_info_udf = F.udf(
        extract_speed_info,
        StructType(
            [
                StructField("category", StringType(), True),
                StructField("numeric", DoubleType(), True),
            ]
        ),
    )
    # load speed into tuple move into new columns then drop tuple col
    df = (
        df.withColumn("speed_info", speed_info_udf(F.col("Congestion_Speed")))
        .withColumn("speed_category", F.col("speed_info.category"))
        .withColumn("speed_numeric", F.col("speed_info.numeric"))
        .drop("speed_info")
    )
    categorize_weather_udf = F.udf(categorize_weather, StringType())
    df = df.withColumn("weather_rating", categorize_weather_udf(F.col("Weather_Conditions")))

    # unlike pandas we dont have dummies so we need to create them
    weather_categories = ["good", "bad", "neutral", "unknown"]
    for category in weather_categories:
        df = df.withColumn(
            f"weather_{category}", F.when(F.col("weather_rating") == category, 1).otherwise(0)
        )
    # unlike pandas we dont have dummies so we need to create them
    weather_categories = ["good", "bad", "neutral", "unknown"]
    for category in weather_categories:
        df = df.withColumn(
            f"weather_{category}", F.when(F.col("weather_rating") == category, 1).otherwise(0)
        )

    # Batch convert vars (aka numeric columns)
    numeric_columns = WEATHER_NUMERIC + TRAFFIC_VARS
    for col_name in numeric_columns:
        df = df.withColumn(col_name, F.col(col_name).cast("double"))

    # include both original and dummy columns
    df = df.select([F.col(c) for c in REQUIRED_COLUMNS])

    # check missing
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"missing required columns: {missing_cols}")

    print("\n!!!!!!!!!!!!!!!!!!!!!! NEW DATA SCHEME !!!!!!!!!!!!!!!!!!! :")
    df.printSchema()
    df.show(5)

    return df


def categorize_weather(condition: str) -> str:
    """Categorize weather condition into good, bad, or neutral"""

    if not condition or condition.strip() == "":
        return "unknown"

    condition_lower = condition.lower()

    if any(good in condition_lower for good in good_weather):
        return "good"

    if any(bad in condition_lower for bad in bad_weather):
        return "bad"

    if any(neutral in condition_lower for neutral in neutral_weather):
        return "neutral"

    return "unknown"


def train_linear_regression_model(df, features: List[str], target: str):
    """Train linear regression model using PySpark"""

    #drop null columns:
    checkcolumns = features + [target]
    df.dropna(subset=checkcolumns)

    # Assemble feature columns into a vector
    assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip")
    df = assembler.transform(df)
    # standardize for scales
    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features", withStd=True, withMean=True
    )
    scaler_df = scaler.fit(df).transform(df)

    lr = LinearRegression(featuresCol="scaled_features", labelCol=target,standardization=False)
    lr_model = lr.fit(scaler_df)

    # extract r2 value
    r2_value = lr_model.summary.r2
    # get the number of significant features
    coefficients = lr_model.coefficients
    # dont use zero here
    significant_features = [
        (features[i], coeff) for i, coeff in enumerate(coefficients) if abs(coeff > 1e-8)
    ]
    num_significant_features = len(significant_features)

    print(
        f"Target: {target} | R²: {r2_value:.4f} | Number of Significant Features: {num_significant_features} | Features: {features}"
    )

    return {
        "r2": r2_value,
        "num_significant_features": num_significant_features,
        "features": features,
    }


def analyze_all_models(df, max_predictors: int = 2, output_file=None):
    results = []

    # Create a file with header if doesnt extist
    if not os.path.exists(output_file):
        pd.DataFrame(
            columns=["target", "r2", "num_significant_features", "significant_features", "formula"]
        ).to_csv(output_file, index=False)
    # Weather -> Traffic models
    for traffic_target in TRAFFIC_VARS:
        for n in range(1, max_predictors + 1):
            for predictors in itertools.combinations(WEATHER_VARS, n):
                features = list(predictors)
                features.append("Traffic Jam Duration")  # Add essential feature

                df = df.cache()  # Cache DataFrame for performance
                try:
                    # Train the model and get results
                    model_results = train_linear_regression_model(df, features, traffic_target)

                    # Append results to the list
                    result_row = {
                        "target": traffic_target,
                        "r2": model_results["r2"],
                        "num_significant_features": model_results["num_significant_features"],
                        "significant_features": model_results["significant_features"],
                        "formula": f"{traffic_target} ~ {' + '.join(features)}",
                    }

                    pd.DataFrame([result_row]).to_csv(
                        output_file, mode="a", header=False, index=False
                    )
                except Exception as e:
                    print(
                        f"Failed to train model for {traffic_target} with predictors {predictors}: {e}"
                    )
                    exit(1)


def main():
    try:
        # TODO: hdfs location

        parquet_path = (
            "/user/s3521281/filled_traffic"  # on hdfs cluster use: /user/s3521281/filled_traffic
        )
        # parquet_path = "./PARQUETFILLED"

        df = process_file(parquet_path)
        timestamp = datetime.now().strftime("%m%d%H%M")
        output_file = f"{timestamp}_sparklinreg.csv"

        print(f"Results logging to to: {output_file}")

        analyze_all_models(df, max_predictors=MAX_NUMBER_PREDICTOR, output_file=output_file)
        results_df = analyze_all_models(df, max_predictors=2)

        print("\n Most Significant influence:  ")
        print(results_df.sort_values("adj_r2", ascending=False).head(100).to_string())

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        print(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()
