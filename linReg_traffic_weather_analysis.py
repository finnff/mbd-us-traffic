import re
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import itertools
from dataclasses import dataclass
from weather_conditions_map import good_weather, neutral_weather, bad_weather


@dataclass
class FeatureConfig:
    """Class to manage feature configurations for modeling"""

    name: str
    enabled: bool = True
    coefficient: float = 1.0


class TrafficWeatherFeatures:
    """Container for traffic and weather feature configurations"""

    def __init__(self):
        self.features = {
            # Weather features
            "Temperature(F)": FeatureConfig("Temperature(F)"),
            "WindChill(F)": FeatureConfig("WindChill(F)"),
            "Humidity(%)": FeatureConfig("Humidity(%)"),
            "Pressure(in)": FeatureConfig("Pressure(in)"),
            "WindSpeed(mph)": FeatureConfig("WindSpeed(mph)"),
            "Precipitation(in)": FeatureConfig("Precipitation(in)"),
            "weather_rating": FeatureConfig("weather_rating"),
            # Traffic features
            "Severity": FeatureConfig("Severity"),
            "Traffic Jam Duration": FeatureConfig("Traffic Jam Duration"),
            "Distance(mi)": FeatureConfig("Distance(mi)"),
            "DelayFromTypicalTraffic(mins)": FeatureConfig("DelayFromTypicalTraffic(mins)"),
            "DelayFromFreeFlowSpeed(mins)": FeatureConfig("DelayFromFreeFlowSpeed(mins)"),
            "speed_numeric": FeatureConfig("speed_numeric"),
            "speed_category": FeatureConfig("speed_category"),
        }

    def get_active_features(self) -> List[str]:
        """Return list of enabled feature names"""
        return [name for name, cfg in self.features.items() if cfg.enabled]

    def set_coefficient(self, feature_name: str, value: float):
        """Set coefficient for a specific feature"""
        if feature_name in self.features:
            self.features[feature_name].coefficient = value
        else:
            raise ValueError(f"Feature {feature_name} not found")

    def toggle_feature(self, feature_name: str, enable: bool):
        """Enable/disable a specific feature"""
        if feature_name in self.features:
            self.features[feature_name].enabled = enable
        else:
            raise ValueError(f"Feature {feature_name} not found")


def process_file(filename: str) -> pd.DataFrame:
    """Read and process the CSV file with enhanced feature handling"""
    print(f"Processing {filename}...")

    # Read the CSV file
    df = pd.read_csv(filename, low_memory=False)

    df["StartTime"] = pd.to_datetime(df["StartTime"], errors="coerce", utc=True)
    df["EndTime"] = pd.to_datetime(df["EndTime"], errors="coerce", utc=True)

    # Calculate duration only if both times are valid
    df["Traffic Jam Duration"] = (
        df["EndTime"] - df["StartTime"]
    ).dt.total_seconds() / 60  # in minutes

    # Apply the speed conversion
    speed_info = df["Congestion_Speed"].apply(extract_speed_info)
    df["speed_numeric"] = speed_info.apply(lambda x: x[1])
    df["speed_category"] = speed_info.apply(lambda x: x[0])

    # Clean invalid speed values
    df = df[df["speed_numeric"].notna()]

    df["weather_rating"] = df["Weather_Conditions"].apply(categorize_weather)

    # 1. Validate temporal data
    df = df.dropna(subset=["StartTime", "EndTime"])

    # 2. Calculate duration properly
    df["Traffic Jam Duration"] = (df["EndTime"] - df["StartTime"]).dt.total_seconds() / 60

    # 3. Filter invalid durations
    print(f"Before duration filtering: {df.shape}")
    df = df[(df["Traffic Jam Duration"] > 0) & (df["Traffic Jam Duration"] <= 1440)]  # Max 24h
    print(f"After duration filtering: {df.shape}")

    # create dummys here
    df = weather_dummy_creation(df)
    return df


def weather_dummy_creation(df: pd.DataFrame) -> pd.DataFrame:
    df["weather_rating"] = pd.Categorical(
        df["weather_rating"],
        categories=["good", "bad", "neutral", "unknown"],
        ordered=False,
    )

    # dont dropt orignal weather rating by passing axis1
    dummies = pd.get_dummies(df["weather_rating"], prefix="weather", drop_first=False)
    df = pd.concat([df, dummies], axis=1)

    # Ensure all dummy columns exist
    for col in ["weather_good", "weather_bad", "weather_neutral", "weather_unknown"]:
        if col not in df.columns:
            df[col] = 0

    # validate critical columns exist
    required_columns = [
        "Severity",
        "Distance(mi)",
        "DelayFromTypicalTraffic(mins)",
        "DelayFromFreeFlowSpeed(mins)",
        "speed_category",
        "speed_numeric",
        "Temperature(F)",
        "WindSpeed(mph)",
        "Humidity(%)",
        "Precipitation(in)",
        "weather_rating",
        "Traffic Jam Duration",
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
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


def extract_speed_info(speed_str: str) -> tuple:
    # Handle missing values first
    if pd.isna(speed_str):
        return np.nan, "Unknown"

    # Clean and standardize the input
    clean_str = str(speed_str).strip().lower()

    # Map qualitative values to numeric equivalents
    category_map = {
        "fast": ("fast", 60.0),
        "moderate": ("moderate", 40.0),
        "slow": ("slow", 20.0),
    }

    # Find first matching category
    for key in category_map:
        if key in clean_str:
            return category_map[key]

    # Default case for unknown values
    return np.nan, "Unknown"


weather_vars = [
    "Temperature(F)",
    "WindChill(F)",
    "Humidity(%)",
    "Pressure(in)",
    "WindSpeed(mph)",
    "Precipitation(in)",
    "weather_good",
    "weather_bad",
    "weather_neutral",
    "weather_unknown",
]

traffic_vars = [
    "Severity",
    "Traffic Jam Duration",
    "Distance(mi)",
    "DelayFromTypicalTraffic(mins)",
    "DelayFromFreeFlowSpeed(mins)",
    "speed_numeric",
    "speed_category",
]


def generate_formula(target: str, predictors: list) -> str:
    # create formula strings with main effects and pairwise interactions."""
    def quote_var(var: str) -> str:
        if re.search(r"[ ()]", var):
            return f'Q("{var}")'
        return var

    quoted_target = quote_var(target)
    quoted_predictors = [quote_var(p) for p in predictors]
    formula_terms = " + ".join(quoted_predictors)
    return f"{quoted_target} ~ ({formula_terms})**2"


def analyze_all_models(df: pd.DataFrame, max_predictors: int = 2):
    results = []
    # Weather -> Traffic models
    for traffic_target in traffic_vars:
        for n in range(1, max_predictors + 1):
            for predictors in itertools.combinations(weather_vars, n):
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

    # Uncomment this block if you want Traffic -> Weather models
    # Traffic -> Weather models
    # for weather_target in ["Temperature(F)", "WindSpeed(mph)", "Humidity(%)", "Precipitation(in)"]:
    #     for n in range(1, max_predictors + 1):
    #         for predictors in itertools.permutations(traffic_vars, n):
    #             formula = generate_formula(weather_target, list(predictors))
    #             try:
    #                 model = smf.ols(formula, data=df).fit(cov_type="HC3")
    #                 result = {
    #                     "model": f"traffic_to_{weather_target}_{'_'.join(predictors)}",
    #                     "adj_r2": model.rsquared_adj,
    #                     "significant_vars": sum(model.pvalues < 0.05),
    #                     "formula": model.model.formula,
    #                 }
    #                 results.append(result)
    #                 print(f"{len(results)} {formula} - Adj R²: {model.rsquared_adj:.3f}")
    #             except Exception as e:
    #                 print(f"Failed for {weather_target} with predictors {predictors}: {str(e)}")

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
        df = process_file("us_congestion_2016_2022SMALL.csv")
        # all_models = analyze_all_models(df)
        results_df = analyze_all_models(df, 3)
        results_df.to_csv("automated_model_results.csv", index=False)

        print("\n Most Significant influence:  ")
        print(results_df.sort_values("adj_r2", ascending=False).head(100).to_string())

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    main()
