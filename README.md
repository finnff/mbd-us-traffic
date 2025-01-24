Dir structure is :


```
total 16G
drwxr-xr-x. 1 sga sga  738 Jan 23 15:04 .
drwxr-xr-x. 1 sga sga  152 Jan 23 15:03 ..
-rwxr-xr-x. 1 sga sga 5.3K Jan 17 17:50 analyzer.py
-rwxr-xr-x. 1 sga sga 3.0K Jan 17 17:21 downsizer.py
drwxr-xr-x. 1 sga sga  116 Jan 23 15:04 .git
-rw-r--r--. 1 sga sga   33 Jan 23 15:04 .gitignore
-rw-r--r--. 1 sga sga  194 Jan 23 15:04 pyproject.toml
-rwxr-xr-x. 1 sga sga  21K Jan 17 21:51 linReg_traffic_weather_analysis.py
-rw-r--r--. 1 sga sga  12G Dec  9  2023 us_congestion_2016_2022.csv
-rw-r--r--. 1 sga sga 2.1G Dec  9  2023 us_congestion_2016_2022.csv.gz
-rw-r--r--. 1 sga sga  62M Jan 17 17:23 us_congestion_2016_2022SMALL.csv
-rwxr-xr-x. 1 sga sga 4.3K Jan 17 20:11 weather_conditions_categorised.py

```
# Tools

### 1. analyzer.py
**Purpose**: Fast CSV data profiling script for analyzing column statistics

**Parameters**:
- `chunk_size` (int): Rows to process at once (default: 100,000)
- `top_n` (int): Number of frequent values to show (default: 20)

**Usage**:
```bash
python analyzer.py large_dataset.csv analysis_report
```

### 2. downsizer.py
**Purpose**: Create smaller representative data sets from from large CSV files fir testing

**Usage**:
```bash
python downsizer.py us_congestion_2016_2022.csv us_congestion_2016_2022tiny.csv
# Original: 12G us_congestion_2016_2022.csv
# Result:   62M us_congestion_2016_2022tiny.csv
```


### 3. weather_conditions_categorized.py
**Purpose**: Maps weather conditions to good/bad/neutral categories as defined in Excel sheet

### 4. linReg_traffic_weather_analysis.py
**Purpose**: Traffic and weather impact analysis using linear regression

**Dependencies**: pandas, numpy, statsmodels, tqdm, dataclasses

**Usage**:
```bash
python linReg_traffic_weather_analysis.py 
```
