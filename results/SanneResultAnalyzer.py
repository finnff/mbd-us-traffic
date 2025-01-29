import pandas as pd
import itertools
import re
from collections import Counter

# List of weather variables and traffic variables
weather_vars = [
    "Temperature(F)",
    "WindChill(F)",
    "Pressure(in)",
    "Humidity(%)",
    "WindSpeed(mph)",
    "Visibility(mi)",
    "GoodWeather",
    "BadWeather",
    "NeutralWeather",
]

traffic_vars = [
    "Distance(mi)",
    "Severity",
    "DelayFromTypicalTraffic(mins)",
    "TrafficJamDuration(mins)",
    "SpeedNumeric",
]

# Define column names
columns = [
    "traffic_target",
    "predictors_list",
    "rmse_formatted",
    "r2_formatted",
    "adj_r2_formatted",
    "significant_vars",
    "formula_str",
    "valid_permutation",
    "RMSE",
    "R2",
    "Adj_R2",
]



def load_and_check_permutations(file_path):
    """Load CSV file and check if all permutations match column 0 and 1."""
    df = pd.read_csv(file_path, header=None, names=columns)
    df["predictors_list"] = df["predictors_list"].apply(
        lambda x: eval(x) if isinstance(x, str) else x
    )  # Safe eval

    def is_valid_permutation(row):
        sorted_list = sorted(row["predictors_list"])
        return sorted_list == sorted_list  

    df["valid_permutation"] = df.apply(is_valid_permutation, axis=1)
    invalid_rows = df[~df["valid_permutation"]]

    valid_rows = df[df["valid_permutation"]]
    print("Valid permutations:")
    print(valid_rows[["traffic_target", "predictors_list"]])

    if not invalid_rows.empty:
        print("Invalid permutations found:")
        print(invalid_rows[["traffic_target", "predictors_list"]])
    else:
        print("All rows have correct permutations.")

    return df


def generate_combinations(max_predictors=3):
    """Generate all possible combinations of traffic and weather variables."""
    experiment_count = 0
    possible_combinations = []

    for traffic_target in traffic_vars:
        for n in range(1, max_predictors + 1):
            for predictors in itertools.combinations(weather_vars, n):
                features = list(predictors) + ["Traffic Jam Duration"]
                experiment_count += 1
                possible_combinations.append((f'{traffic_target},"{str(features)}"'))
    print(f"\nTotal number of experiments: {experiment_count}")
    return possible_combinations




# Initialize global counter for weather variables
global_weather_var_count = Counter()

# Convert the relevant columns to numeric values for analysis

df = load_and_check_permutations("FINAL_RESULTS_3_PREDICT.csv")
df['RMSE'] = pd.to_numeric(df['rmse_formatted'], errors='coerce')
df['R2'] = pd.to_numeric(df['r2_formatted'], errors='coerce')
df['Adj_R2'] = pd.to_numeric(df['adj_r2_formatted'], errors='coerce')

# Loop through each traffic target variable in the CSV
for traffic_target in traffic_vars:
    print(f"\nAnalyzing weather variables for traffic target: {traffic_target}")
    
    # Filter the rows corresponding to the current traffic target
    target_df = df[df['traffic_target'] == traffic_target]
    
    # Sort by R2 (descending)
    sorted_target_df = target_df.sort_values(by='R2', ascending=False)
    
    df_columns = list(df)
    sorted_target_df = sorted_target_df[df_columns]
    
    # Select the top 10 and bottom 5 models based on R2
    top_models = sorted_target_df.head(10)
    bottom_models = sorted_target_df.tail(5)
    
    # Combine with separator row
    separator = pd.DataFrame([['...'] * len(df_columns)], columns=df_columns)
    combined_df = pd.concat([top_models, separator, bottom_models])
    
    # Drop unnecessary columns
    combined_df = combined_df.drop(columns=['rmse_formatted', 'r2_formatted', 'adj_r2_formatted', 'valid_permutation', 'formula_str'])
    
    # Clean up the predictors_list column format - removing all quotes and replacing brackets
    combined_df['predictors_list'] = combined_df['predictors_list'].astype(str)\
        .str.replace("'", '', regex=False)\
        .str.replace('"', '', regex=False)
    
    # Rename columns by removing underscores
    new_columns = {col: col.replace('_', '') for col in combined_df.columns}
    combined_df = combined_df.rename(columns=new_columns)
    
    print(combined_df)
    
    # Save combined_df to CSV
    combined_df.to_csv(f'{traffic_target}_models.csv', index=False,sep=';')
    
    # Process influential variables
    predictors_list = top_models['predictors_list'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    flat_list = [item for sublist in predictors_list for item in sublist]
    weather_var_count = Counter(flat_list)
    global_weather_var_count.update(flat_list)
    sorted_weather_var_count = dict(sorted(weather_var_count.items(), key=lambda item: item[1], reverse=True))
    
    # Create DataFrame for most influential variables and clean up format
    influential_df = pd.DataFrame(sorted_weather_var_count.items(), columns=['WeatherVariable', 'Frequency'])
    influential_df['WeatherVariable'] = influential_df['WeatherVariable'].str.replace('"', '', regex=False)
    
    # Save influential variables to CSV
    influential_df.to_csv(f'{traffic_target}_mostinfluential.csv', index=False, sep=';')

    
    # Display the most frequently appearing weather variables
    print(f"\nMost Influential Weather Variables Based on Frequency in Top 10 Models for {traffic_target}:")
    for weather_var, count in sorted_weather_var_count.items():
        print(f"{weather_var}: {count} times")

# After all traffic variables are processed, display global influence
print("\nGLOBAL INFLUENCE - Most Influential Weather Variables Across All top 10 Traffic Targets:")
sorted_global_count = dict(sorted(global_weather_var_count.items(), key=lambda item: item[1], reverse=True))
for weather_var, count in sorted_global_count.items():
    print(f"{weather_var}: {count} times")
