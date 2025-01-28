from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.stat import Correlation


good_weather = [
    "Fair",
    "Clear",
    "Mostly Cloudy",
    "Partly Cloudy",
    "Scattered Clouds",
    "Fair / Windy",
    "Mostly Cloudy / Windy",
    "Partly Cloudy / Windy",
]

neutral_weather = [
    "Cloudy",
    "Overcast",
    "NULL/MISSING",
    "Haze",
    "Fog",
    "Cloudy / Windy",
    "Smoke",
    "N/A Precipitation",
    "Mist",
    "Fog / Windy",
    "Patches of Fog",
    "Shallow Fog",
    "Haze / Windy",
    "Smoke / Windy",
    "Partial Fog",
    "Light Haze",
    "Mist / Windy",
    "Light Fog",
    "Patches of Fog / Windy",
    "Shallow Fog / Windy",
    "Heavy Fog",
    "Heavy Smoke",
    "Light Smoke",
]

bad_weather = [
    "Light Snow",
    "Light Rain",
    "Snow",
    "Rain",
    "Heavy Snow",
    "Heavy Rain",
    "Light Drizzle",
    "Light Snow / Windy",
    "Wintry Mix",
    "T-Storm",
    "Thunder in the Vicinity",
    "Light Rain with Thunder",
    "Thunder",
    "Light Freezing Rain",
    "Heavy T-Storm",
    "Light Thunderstorms and Rain",
    "Thunderstorm",
    "Light Rain / Windy",
    "Heavy Thunderstorms and Rain",
    "Snow / Windy",
    "Light Freezing Fog",
    "Thunderstorms and Rain",
    "Drizzle",
    "Heavy Snow / Windy",
    "Blowing Snow",
    "Heavy T-Storm / Windy",
    "Light Freezing Drizzle",
    "Rain / Windy",
    "Snow and Sleet",
    "T-Storm / Windy",
    "Showers in the Vicinity",
    "Heavy Rain / Windy",
    "Light Ice Pellets",
    "Thunder / Windy",
    "Light Snow and Sleet",
    "Blowing Snow / Windy",
    "Wintry Mix / Windy",
    "Sleet",
    "Ice Pellets",
    "Light Sleet",
    "Heavy Drizzle",
    "Snow and Sleet / Windy",
    "Freezing Rain",
    "Drizzle and Fog",
    "Light Rain Showers",
    "Light Drizzle / Windy",
    "Light Snow with Thunder",
    "Widespread Dust",
    "Light Freezing Rain / Windy",
    "Light Rain Shower",
    "Light Snow and Sleet / Windy",
    "Light Snow Showers",
    "Blowing Dust / Windy",
    "Heavy Sleet",
    "Rain Showers",
    "Light Thunderstorms and Snow",
    "Freezing Rain / Windy",
    "Sleet / Windy",
    "Heavy Blowing Snow",
    "Thunder / Wintry Mix",
    "Squalls",
    "Heavy Snow with Thunder",
    "Heavy Ice Pellets",
    "Small Hail",
    "Blowing Dust",
    "Heavy Thunderstorms and Snow",
    "Light Sleet / Windy",
    "Sleet and Thunder",
    "Rain Shower",
    "Hail",
    "Snow Showers",
    "Heavy Freezing Drizzle",
    "Snow and Thunder",
    "Sand / Dust Whirlwinds",
    "Light Snow Shower",
    "Freezing Drizzle",
    "Heavy Freezing Rain",
    "Funnel Cloud",
    "Squalls / Windy",
    "Snow and Thunder / Windy",
    "Heavy Sleet and Thunder",
    "Thunderstorms and Snow",
    "Heavy Sleet / Windy",
    "Low Drifting Snow",
    "Light Blowing Snow",
    "Drizzle / Windy",
    "Light Hail",
    "Thunder and Hail",
    "Thunderstorms and Ice Pellets",
    "Light Sleet and Thunder",
    "Sand",
    "Light Snow Grains",
    "Heavy Rain Showers",
    "Light Rain Shower / Windy",
    "Heavy Rain Shower",
    "Heavy Thunderstorms with Ice Pellets",
    "Snow Shower",
    "Volcanic Ash",
    "Thunder and Hail / Windy",
    "Sandstorm",
    "Drifting Snow",
    "Light Snow Shower / Windy",
    "Widespread Dust / Windy",
    "Heavy Thunderstorms with Hail",
    "Thunder / Wintry Mix / Windy",
    "Rain / Freezing Rain",
    "Tornado",
    "Sand / Dust Whirls Nearby",
    "Light Thunderstorm",
    "Light Snow Grains / Windy",
    "Heavy Snow Showers",
    "Light Thunderstorms and Ice Pellets",
    "Light Ice Pellet Showers",
    "Heavy Thunderstorm",
    "Heavy Thunderstorms with Small Hail",
    "Dust Whirls",
    "Rain Shower / Windy",
    "Rain and Sleet",
    "Blowing Snow Nearby / Windy",
    "Snow Grains",
    "Thunder and Small Hail",
    "Blowing Snow Nearby",
    "Light Thunderstorms with Small Hail",
    "Rain and Snow / Windy",
    "Blowing Sand",
    "Sand / Dust Whirlwinds / Windy",
    "Sleet and Thunder / Windy",
    "Rain and Snow",
    "Drizzle and Fog / Windy",
    "Duststorm / Windy",
    "Hail / Windy",
    "Heavy Hail",
    "Small Hail / Windy",
    "Heavy Drizzle / Windy",
    "Ice Crystals",
    "Thunderstorms with Hail",
    "Light Thunderstorms with Hail",
    "Low Drifting Sand / Windy",
    "Drifting Snow / Windy",
    "Snow Shower / Windy",
    "Blowing Sand / Windy",
    "Thunderstorms with Small Hail",
    "Low Drifting Dust / Windy",
]


def categorize_weather(condition):
    if condition is None or condition.strip() == "":
        return 0
    condition_lower = condition.lower()

    if any(good.lower() in condition_lower for good in good_weather):
        return 1
    if any(bad.lower() in condition_lower for bad in bad_weather):
        return -1
    if any(neutral.lower() in condition_lower for neutral in neutral_weather):
        return 0

    return 0


def categorize_congestion_speed(speed):
    if speed is None or speed.strip() == "":
        return 0
    speed_lower = speed.lower()
    if speed_lower == "fast":
        return 1
    elif speed_lower == "moderate":
        return 0
    elif speed_lower == "slow":
        return -1
    else:
        return 0


# Set Up
spark = SparkSession.builder.appName("LinReg").getOrCreate()
# data_traffic = spark.read.option("header", True).csv("/user/s2899078/projectdata/us_congestion_2016_2022.csv.gz")
data_traffic = spark.read.parquet("data/filled_traffic")  # Filled in null values

categorize_weather_udf = udf(categorize_weather, IntegerType())
categorize_congestion_speed_udf = udf(categorize_congestion_speed, IntegerType())

var1 = "DelayFromTypicalTraffic(mins)"
var2 = "Weather_Conditions"
var3 = "Congestion_Speed"
var4 = "Severity"
var5 = "DelayFromFreeFlowSpeed(mins)"

# Drop NULL values, categorize weather and congestion speed, and cast to numeric types
# data_traffic = data_traffic.withColumn("weather_rating", categorize_weather_udf(col(var2)))
# data_traffic = data_traffic.withColumn("congestion_speed_rating", categorize_congestion_speed_udf(col(var3)))
# data_traffic = data_traffic.withColumn(var1, col(var1).cast("double"))
# data_traffic = data_traffic.withColumn(var4, col(var4).cast("double"))
# data_traffic = data_traffic.withColumn(var5, col(var5).cast("double"))
# filtered_data = data_traffic.select(col(var1),col(var4),col(var5), col("weather_rating"), col("congestion_speed_rating")).na.drop()

# pearson_corr_weather = filtered_data.stat.corr(var4, "weather_rating", method="pearson")
# print(f"Pearson correlation coefficient between {var4} and weather_rating: {pearson_corr_weather}")
# pearson_corr_weather = filtered_data.stat.corr(var5, "weather_rating", method="pearson")
# print(f"Pearson correlation coefficient between {var5} and weather_rating: {pearson_corr_weather}")
# pearson_corr_weather = filtered_data.stat.corr(var1, "weather_rating", method="pearson")
# print(f"Pearson correlation coefficient between {var1} and weather_rating: {pearson_corr_weather}")
# pearson_corr_weather = filtered_data.stat.corr("congestion_speed_rating", "weather_rating", method="pearson")
# print(f"Pearson correlation coefficient between congestion_speed_rating and weather_rating: {pearson_corr_weather}")


var_weather = "DelayFromTypicalTraffic(mins)"
var_traffic = "Precipitation(in)"

filtered_data = data_traffic.select(col(var_weather), col(var_traffic)).na.drop()

# TODO Can be split using
# train_data,test_data = filtered_data.randomSplit([0.7,0.3])

assembler = VectorAssembler(inputCols=[var_traffic], outputCol="features")

assembled_df = assembler.transform(filtered_data).select("features", var_weather)


# Perform linear regression
lr = LinearRegression(featuresCol="features", labelCol=var_weather)

lr_model = lr.fit(assembled_df)

# Print the model summary
summary = lr_model.summary


print(f"R²: {summary.r2}")
print(f"RMSE: {summary.rootMeanSquaredError}")
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

# Display residuals
summary.residuals.show()
