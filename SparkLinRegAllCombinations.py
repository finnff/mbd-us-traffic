import re
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import itertools
from dataclasses import dataclass
from weather_conditions_map import good_weather, neutral_weather, bad_weather


from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, DoubleType, StringType, StructType, StructField
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.stat import Correlation
from time import sleep


# MOVING TO TOP OF FILE
# CHANGE PARAMS HERE
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

# required columns for final output
REQUIRED_COLUMNS = WEATHER_NUMERIC + TRAFFIC_VARS + WEATHER_DUMMY_COLS + ["weather_rating"]


def create_spark_session(app_name="SparkLinRegAllCombinations"):
    """Create or get existing Spark session"""
    return SparkSession.builder.appName(app_name).getOrCreate()


def extract_speed_info(speed_str: str) -> tuple:
    if pd.isna(speed_str):
        return (np.nan, "Unknown")
    clean_str = str(speed_str).strip().lower()
    category_map = {
        "fast": ("fast", 60.0),
        "moderate": ("moderate", 40.0),
        "slow": ("slow", 20.0),
    }
    for key in category_map:
        if key in clean_str:
            return category_map[key]
    return (np.nan, "Unknown")


def process_file(parquet_path: str) -> pd.DataFrame:
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
    if pd.isna(condition) or str(condition).strip() == "":
        return "unknown"

    condition_lower = str(condition).lower()

    for good in good_weather:
        if good.lower() in condition_lower:
            return "good"

    for bad in bad_weather:
        if bad.lower() in condition_lower:
            return "bad"

    for neutral in neutral_weather:
        if neutral.lower() in condition_lower:
            return "neutral"

    return "unknown"


def generate_formula(target: str, predictors: list) -> str:
    # create formula strings with main effects and pairwise interactions."""
    def quote_var(var: str) -> str:
        if re.search(r"[ ()]", var):
            return f'Q("{var}")'
        return var

    quoted_target = quote_var(target)
    # we hit recusion depth if we have target in predictors
    quoted_predictors = [quote_var(p) for p in predictors if p != target]
    formula_terms = " + ".join(quoted_predictors)
    return f"{quoted_target} ~ ({formula_terms})**2"


def analyze_all_models(df: pd.DataFrame, max_predictors: int = 2):
    results = []
    # Weather -> Traffic models
    for traffic_target in TRAFFIC_VARS:
        for n in range(1, max_predictors + 1):
            for predictors in itertools.combinations(WEATHER_VARS, n):
                formula = generate_formula(traffic_target, list(predictors))
                try:
                    model = smf.ols(formula, data=df).fit(cov_type="HC3")
                    result = {
                        "model": f"weather_to_{traffic_target}_{'_'.join(predictors)}",
                        "adj_r2": model.rsquared_adj,
                        "significant_vars": sum(model.pvalues < 0.05),
                        "formula": model.model.formula,
                    }
                    results.append(result)
                    print(f"{len(results)} {formula} - Adj R²: {model.rsquared_adj:.3f}")
                except Exception as e:
                    print(f"Failed for {traffic_target} with predictors {predictors}: {str(e)}")

    results_df = pd.DataFrame(results)
    print("\nAnalysis Summary:")
    print(f"Total models analyzed: {len(results_df)}")
    print(f"Average Adj R²: {results_df['adj_r2'].mean():.3f}")
    print(f"Best Adj R²: {results_df['adj_r2'].max():.3f}")
    return pd.DataFrame(results)


def compile_results(models: dict) -> pd.DataFrame:
    # get info from models for printing
    results = []
    for name, model in models.items():
        if not hasattr(model, "rsquared_adj"):
            continue  # Skip failed models

        results.append(
            {
                "model": name,
                "adj_r2": model.rsquared_adj,
                "significant_vars": sum(model.pvalues < 0.05),
                "formula": model.model.formula,
            }
        )
    return pd.DataFrame(results)


def main():
    try:
        # TODO: hdfs location
        parquet_path = "./PARQUETFILLED"
        df = process_file(parquet_path)
        # convert to pandas df
        # df = df.toPandas()
        # print first 5 rows
        df = df.toPandas()
        
        # Print schema and first few rows for verification
        print(df.info())
        print(df.head())

        # Test the formula generation
        target = "speed_numeric"
        predictors = WEATHER_VARS  # Full list of weather variables
        
        formula = generate_formula(target, predictors)
        print("Generated Formula:", formula)
        sleep(5)
        results_df = analyze_all_models(df, 2)
        results_df.to_csv("automated_model_results.csv", index=False)

        print("\n Most Significant influence:  ")
        print(results_df.sort_values("adj_r2", ascending=False).head(100).to_string())

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    main()
