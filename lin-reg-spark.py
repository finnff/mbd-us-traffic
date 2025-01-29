from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import functions as F
from itertools import combinations
from typing import List
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, DoubleType
import csv
import re



good_weather = ["Fair", "Clear", "Mostly Cloudy", "Partly Cloudy", "Scattered Clouds",
                "Fair / Windy", "Mostly Cloudy / Windy","Partly Cloudy / Windy",]

neutral_weather = ["Cloudy", "Overcast", "NULL/MISSING", "Haze", "Fog", "Cloudy / Windy", "Smoke", "N/A Precipitation", "Mist",
                   "Fog / Windy", "Patches of Fog", "Shallow Fog","Haze / Windy", "Smoke / Windy", "Partial Fog", "Light Haze",
                   "Mist / Windy", "Light Fog", "Patches of Fog / Windy", "Shallow Fog / Windy", "Heavy Fog", "Heavy Smoke", "Light Smoke",]

bad_weather = ["Light Snow", "Light Rain", "Snow", "Rain", "Heavy Snow", "Heavy Rain", "Light Drizzle", "Light Snow / Windy", "Wintry Mix",
               "T-Storm", "Thunder in the Vicinity", "Light Rain with Thunder", "Thunder", "Light Freezing Rain", "Heavy T-Storm", 
               "Light Thunderstorms and Rain", "Thunderstorm", "Light Rain / Windy", "Heavy Thunderstorms and Rain", "Snow / Windy",
               "Light Freezing Fog", "Thunderstorms and Rain", "Drizzle", "Heavy Snow / Windy", "Blowing Snow", "Heavy T-Storm / Windy",
               "Light Freezing Drizzle", "Rain / Windy", "Snow and Sleet", "T-Storm / Windy", "Showers in the Vicinity", "Heavy Rain / Windy", 
               "Light Ice Pellets", "Thunder / Windy", "Light Snow and Sleet", "Blowing Snow / Windy", "Wintry Mix / Windy", "Sleet", 
               "Ice Pellets", "Light Sleet", "Heavy Drizzle", "Snow and Sleet / Windy", "Freezing Rain", "Drizzle and Fog", "Light Rain Showers",
               "Light Drizzle / Windy", "Light Snow with Thunder", "Widespread Dust", "Light Freezing Rain / Windy", "Light Rain Shower",
               "Light Snow and Sleet / Windy", "Light Snow Showers", "Blowing Dust / Windy", "Heavy Sleet", "Rain Showers",
               "Light Thunderstorms and Snow", "Freezing Rain / Windy", "Sleet / Windy", "Heavy Blowing Snow", "Thunder / Wintry Mix", 
               "Squalls", "Heavy Snow with Thunder", "Heavy Ice Pellets", "Small Hail", "Blowing Dust", "Heavy Thunderstorms and Snow", 
               "Light Sleet / Windy", "Sleet and Thunder", "Rain Shower", "Hail", "Snow Showers", "Heavy Freezing Drizzle", "Snow and Thunder",
               "Sand / Dust Whirlwinds", "Light Snow Shower", "Freezing Drizzle", "Heavy Freezing Rain", "Funnel Cloud", "Squalls / Windy",
               "Snow and Thunder / Windy", "Heavy Sleet and Thunder", "Thunderstorms and Snow", "Heavy Sleet / Windy", "Low Drifting Snow", 
               "Light Blowing Snow", "Drizzle / Windy", "Light Hail", "Thunder and Hail", "Thunderstorms and Ice Pellets", "Light Sleet and Thunder",
               "Sand", "Light Snow Grains", "Heavy Rain Showers", "Light Rain Shower / Windy", "Heavy Rain Shower", "Heavy Thunderstorms with Ice Pellets",
               "Snow Shower", "Volcanic Ash", "Thunder and Hail / Windy", "Sandstorm", "Drifting Snow", "Light Snow Shower / Windy", "Widespread Dust / Windy",
               "Heavy Thunderstorms with Hail", "Thunder / Wintry Mix / Windy", "Rain / Freezing Rain", "Tornado", "Sand / Dust Whirls Nearby",
               "Light Thunderstorm", "Light Snow Grains / Windy", "Heavy Snow Showers", "Light Thunderstorms and Ice Pellets", "Light Ice Pellet Showers",
               "Heavy Thunderstorm", "Heavy Thunderstorms with Small Hail", "Dust Whirls", "Rain Shower / Windy", "Rain and Sleet",
               "Blowing Snow Nearby / Windy", "Snow Grains", "Thunder and Small Hail", "Blowing Snow Nearby", "Light Thunderstorms with Small Hail",
               "Rain and Snow / Windy", "Blowing Sand", "Sand / Dust Whirlwinds / Windy", "Sleet and Thunder / Windy", "Rain and Snow",
               "Drizzle and Fog / Windy", "Duststorm / Windy", "Hail / Windy", "Heavy Hail", "Small Hail / Windy", "Heavy Drizzle / Windy",
               "Ice Crystals", "Thunderstorms with Hail", "Light Thunderstorms with Hail", "Low Drifting Sand / Windy", "Drifting Snow / Windy",
               "Snow Shower / Windy", "Blowing Sand / Windy", "Thunderstorms with Small Hail", "Low Drifting Dust / Windy",]



def process_file(file_path: str):
    # Read data
    traffic_df = spark.read.parquet(file_path)




    # Compute TrafficJamDuration(mins) from EndTime and StartTime
    traffic_df = traffic_df.withColumn("EndTime", F.to_timestamp("EndTime", "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"))
    traffic_df = traffic_df.withColumn(
        "TrafficJamDuration(mins)",
        (F.unix_timestamp(col("EndTime")) - F.unix_timestamp(col("StartTime"))) / 60
    )
    traffic_df = traffic_df.filter((col("TrafficJamDuration(mins)") > 0) & (col("TrafficJamDuration(mins)") <= 1440)) # Filter valid TrafficJamDuration values (greater than 0 and max 24 hours)




    # Map Congestion_Speed strings to numeric values using category_map
    category_map = {
        "fast": 60.0,
        "moderate": 40.0,
        "slow": 20.0
    }
    # Create a UDF to map the Congestion_Speed category or set as null for invalid values
    map_congestion_speed = F.udf(lambda speed: category_map.get(speed.lower(), None), "double")
    traffic_df = traffic_df.withColumn("SpeedNumeric", map_congestion_speed(F.col("Congestion_Speed")))


    # UDF for weather condition mapping
    def weather_category(condition: str, category: set) -> int:
        if condition is None:
            return 0
        return int(condition.strip() in category)

    # Register UDFs
    good_weather_udf = F.udf(lambda x: weather_category(x, good_weather), "int")
    neutral_weather_udf = F.udf(lambda x: weather_category(x, neutral_weather), "int")
    bad_weather_udf = F.udf(lambda x: weather_category(x, bad_weather), "int")

    # Apply the transformations
    traffic_df = traffic_df.withColumn("GoodWeather", good_weather_udf(F.col("Weather_Conditions"))) \
                           .withColumn("NeutralWeather", neutral_weather_udf(F.col("Weather_Conditions"))) \
                           .withColumn("BadWeather", bad_weather_udf(F.col("Weather_Conditions")))


    for var in weather_vars + traffic_vars:
        traffic_df = traffic_df.withColumn(var, F.col(var).cast("double"))
    
    # Select required features and drop rows with missing values
    filtered_data = traffic_df.select(weather_vars + traffic_vars).na.drop()
    print("CP: Data loaded")

    return filtered_data


def apply_log_transform(df, target_col):
    # Apply log transformation to the target variable to handle heteroskedasticity
    df = df.withColumn(target_col, F.log(F.col(target_col)))
    return df




spark = (
    SparkSession.builder
    .appName("Traffic")
    .getOrCreate()
)

# Define the weather and traffic variables
weather_vars = ["Temperature(F)", "WindChill(F)", "Pressure(in)",  "Humidity(%)", "WindSpeed(mph)", "Visibility(mi)", "GoodWeather", "BadWeather", "NeutralWeather"]
traffic_vars = ["Distance(mi)", "Severity", "DelayFromTypicalTraffic(mins)", "TrafficJamDuration(mins)", "SpeedNumeric"]

max_predictors = 3  # Define the max number of predictors
results = []
data = '/user/s3521281/filled_traffic' # on hdfs cluster use: /user/s3521281/filled_traffic
output_file = 'regression_results_log.csv'

# Load existing results from the CSV file
existing_experiments = set()
try:
    with open(output_file, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            existing_experiments.add((row[0], row[6]))  # target and formula uniquely identify an experiment
except FileNotFoundError:
    print(f"Output file {output_file} not found. Creating a new one.")

df = process_file(data)
df.describe().show()






# Analyze models for weather → traffic
for traffic_target in traffic_vars:
    for n in range(1, max_predictors + 1):
        for predictors in combinations(weather_vars, n):
            predictors_list = list(predictors)
            formula_str = f"{traffic_target} ~ ({' + '.join(predictors_list)})"
                
            # Skip if experiment already done
            if (traffic_target, formula_str) in existing_experiments:
                print(f"CP: Skipping existing model: {formula_str}")
                continue
            
            transformed_df = df.withColumn("log_target", log1p(col(traffic_target)))

            # Assemble feature vector
            assembler = VectorAssembler(inputCols=predictors_list, outputCol="features")
            assembled_df = assembler.transform(transformed_df).select("features", "log_target")
            
            # Skip if insufficient data
            if assembled_df.count() < 2:
                print(f"CP: Insufficient data for {traffic_target} with predictors {predictors_list}")
                continue
            
            #  Apply log transformation to the target variable to handle heteroskedasticity
            # transformed_df = apply_log_transform(assembled_df, traffic_target)

            # Perform linear regression
            lr = LinearRegression(featuresCol="features", labelCol="log_target")
            try:
                lr_model = lr.fit(assembled_df)
                print("CP: LR Model fit")
                summary = lr_model.summary

                significant_vars = sum(p < 0.05 for p in summary.pValues if p is not None)  # Count significant predictors (p-value < 0.05)
                adj_r2 = summary.r2adj
                rmse = summary.rootMeanSquaredError
                r2 = summary.r2

                # print("CP: calculating count now")
                # # Manual calculation of adjusted R²
                # n_obs = assembled_df.count()  # Number of observations
                # p_predictors = len(predictors_list)  # Number of predictors
                # # Avoid division by zero (if p_predictors + 1 >= n_obs)
                # if n_obs > p_predictors + 1:
                #     manual_adj_r2 = 1 - ((1 - r2) * (n_obs - 1) / (n_obs - p_predictors - 1))
                # else:
                #     manual_adj_r2 = None
                # print("CP: calculated count")

                # Format the statistics to 4 decimals
                adj_r2_formatted = f"{adj_r2:.4f}"
                rmse_formatted = f"{rmse:.4f}"
                r2_formatted = f"{r2:.4f}"
                # manual_adj_r2_formatted = f"{manual_adj_r2:.4f}" if manual_adj_r2 is not None else "N/A"

                result_row = [
                    traffic_target,
                    predictors_list,
                    rmse_formatted,
                    r2_formatted,
                    adj_r2_formatted,
                    # manual_adj_r2_formatted,
                    significant_vars,
                    formula_str
                ]

                # Append results to CSV after each successful model
                with open(output_file, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(result_row)

                print(f"CP: Model: {formula_str} | RMSE: {rmse_formatted} | R²: {r2_formatted} | "
                      f"Adj R²: {adj_r2_formatted} | "
                      f"Significant Vars: {significant_vars}")
            except Exception as e:
                print(f"CP: Failed for {traffic_target} with predictors {predictors_list}: {str(e)}")