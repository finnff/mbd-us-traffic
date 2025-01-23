#!/usr/bin/env python3
import re
import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm

# Weather condition mappings
good_weather = [
    "Fair",
    "Clear",
    "Mostly Cloudy",
    "Partly Cloudy",
    "Scattered Clouds",
    "Fair / Windy",
    "Mostly Cloudy / Windy",
    "Partly Cloudy / Windy"
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
    "Light Smoke"
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
    "Low Drifting Dust / Windy"
    ]

def categorize_weather(condition: str) -> str:
    """Categorize weather condition into good, bad, or neutral"""
    if pd.isna(condition) or str(condition).strip() == '':
        return 'unknown'
    
    condition_lower = str(condition).lower()
    
    for good in good_weather:
        if good.lower() in condition_lower:
            return 'good'
            
    for bad in bad_weather:
        if bad.lower() in condition_lower:
            return 'bad'
            
    for neutral in neutral_weather:
        if neutral.lower() in condition_lower:
            return 'neutral'
            
    return 'unknown'

def extract_speed_info(speed_str: str) -> tuple:
    """Extract both qualitative and quantitative speed information"""
    if pd.isna(speed_str):
        return np.nan, 'Unknown'
    
    speed_str = str(speed_str)
    
    # Extract qualitative speed (Fast/Moderate/Slow)
    if 'Fast' in speed_str:
        category = 'Fast'
    elif 'Moderate' in speed_str:
        category = 'Moderate'
    elif 'Slow' in speed_str:
        category = 'Slow'
    else:
        category = 'Unknown'
    
    # Extract numeric value
    try:
        numeric = float(''.join(filter(lambda x: x.isdigit() or x == '.', speed_str)))
    except:
        numeric = np.nan
        
    return numeric, category

def process_file(filename: str):
    """Read and process the CSV file"""
    print(f"Processing {filename}...")
    
    # Read the CSV file
    df = pd.read_csv(filename, low_memory=False)
    
    # Process Congestion_Speed column
    speed_info = df['Congestion_Speed'].apply(extract_speed_info)
    df['speed_numeric'] = speed_info.apply(lambda x: x[0])
    df['speed_category'] = speed_info.apply(lambda x: x[1])
    
    # Process delay columns
    df['DelayFromTypicalTraffic(mins)'] = pd.to_numeric(df['DelayFromTypicalTraffic(mins)'], errors='coerce')
    df['DelayFromFreeFlowSpeed(mins)'] = pd.to_numeric(df['DelayFromFreeFlowSpeed(mins)'], errors='coerce')
    
    # Add weather rating
    df['weather_rating'] = df['Weather_Conditions'].apply(categorize_weather)
    
    return df

def analyze_weather_impact(df: pd.DataFrame):
    """Analyze the impact of weather on traffic"""
    results = []
    
    for rating in ['good', 'bad', 'neutral', 'unknown']:
        mask = df['weather_rating'] == rating
        subset = df[mask]
        
        if len(subset) > 0:
            # Calculate speed category distribution
            speed_dist = subset['speed_category'].value_counts()
            speed_pct = (speed_dist / len(subset) * 100).round(2)
            
            result = {
                'weather_rating': rating,
                'total_incidents': len(subset),
                'avg_speed': subset['speed_numeric'].mean(),
                'avg_delay_typical': subset['DelayFromTypicalTraffic(mins)'].mean(),
                'avg_delay_freeflow': subset['DelayFromFreeFlowSpeed(mins)'].mean(),
                'pct_fast': speed_pct.get('Fast', 0),
                'pct_moderate': speed_pct.get('Moderate', 0),
                'pct_slow': speed_pct.get('Slow', 0),
            }
            results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Add percentage of total incidents
    total_records = results_df['total_incidents'].sum()
    results_df['percentage_of_total'] = (results_df['total_incidents'] / total_records * 100).round(2)
    
    return results_df.round(2)

def main():
    try:
        # Process the file
        df = process_file('us_congestion_2016_2022SMALL.csv')
        
        # Save processed data
        print("\nSaving processed data...")
        df.to_csv('weather_rated_traffic.csv', index=False)
        
        # Analyze weather impact
        print("Analyzing weather impact...")
        analysis = analyze_weather_impact(df)
        
        # Save analysis
        analysis.to_csv('weather_analysis.csv', index=False)
        
        # Print summary
        print("\nWeather Impact Analysis:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(analysis.to_string())
        
        # Print speed category distribution by weather
        print("\nDetailed Speed Category Distribution by Weather:")
        speed_weather_dist = pd.crosstab(
            df['weather_rating'], 
            df['speed_category'], 
            normalize='index'
        ) * 100
        print(speed_weather_dist.round(2))
        
        print("\nFiles saved:")
        print("1. weather_rated_traffic.csv - Full dataset with weather ratings")
        print("2. weather_analysis.csv - Weather impact analysis")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()




"""
Weather Impact Analysis:
  weather_rating  total_incidents  avg_speed  avg_delay_typical  avg_delay_freeflow  pct_fast  pct_moderate  pct_slow  percentage_of_total
0           good         27784047        NaN               2.72                4.01     33.33         41.27     25.40                83.43
1            bad          4570965        NaN               3.34                4.10     50.96         32.40     16.64                13.72
2        unknown           949187        NaN               3.10                4.04     38.85         36.29     24.86                 2.85

Detailed Speed Category Distribution by Weather:
speed_category   Fast  Moderate   Slow
weather_rating
bad             50.96     32.40  16.64
good            33.33     41.27  25.40
unknown         38.85     36.29  24.86


Weather Impact Analysis:
  weather_rating  total_incidents  avg_speed  avg_delay_typical  avg_delay_freeflow  pct_fast  pct_moderate  pct_slow  percentage_of_total
0           good         21876833        NaN               2.69                3.99     32.61         41.87     25.52                65.69
1            bad          4570965        NaN               3.34                4.10     50.96         32.40     16.64                13.72
2        neutral          5907214        NaN               2.83                4.07     35.98         39.05     24.97                17.74
3        unknown           949187        NaN               3.10                4.04     38.85         36.29     24.86                 2.85

Detailed Speed Category Distribution by Weather:
speed_category   Fast  Moderate   Slow
weather_rating
bad             50.96     32.40  16.64
good            32.61     41.87  25.52
neutral         35.98     39.05  24.97
unknown         38.85     36.29  24.86
"""
