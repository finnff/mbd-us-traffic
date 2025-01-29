
# MAX 3 PREDICTORS RESULTS 

| traffic_target | predictors_list |
|---------------|----------------|
| Distance(mi) | [Temperature(F)] |
| Distance(mi) | [WindChill(F)] |
| Distance(mi) | [Pressure(in)] |
| Distance(mi) | [Humidity(%)] |
| Distance(mi) | [WindSpeed(mph)] |
| ... | ... |
| SpeedNumeric | [WindSpeed(mph), GoodWeather, NeutralWeather] |
| SpeedNumeric | [WindSpeed(mph), BadWeather, NeutralWeather] |
| SpeedNumeric | [Visibility(mi), GoodWeather, BadWeather] |
| SpeedNumeric | [Visibility(mi), GoodWeather, NeutralWeather] |
| SpeedNumeric | [Visibility(mi), BadWeather, NeutralWeather] |

Total: **645 rows**  
✅ All rows have correct permutations.

---

## Analyzing Weather Variables for Traffic Target: **Distance(mi)**

| traffictarget | predictorslist | significantvars | RMSE | R² | AdjR² |
|--------------|-----------------|-----------------|------|------|-------|
| Distance(mi) | [WindChill(F), Pressure(in), Visibility(mi)] | 4 | 0.5307 | 0.0506 | 0.0506 |
| Distance(mi) | [Temperature(F), Pressure(in), Visibility(mi)] | 4 | 0.5309 | 0.0497 | 0.0497 |
| Distance(mi) | [Pressure(in), Visibility(mi), BadWeather] | 4 | 0.5309 | 0.0497 | 0.0497 |
| Distance(mi) | [Pressure(in), Humidity(%), Visibility(mi)] | 4 | 0.5316 | 0.0474 | 0.0474 |
| Distance(mi) | [Pressure(in), Visibility(mi), GoodWeather] | 4 | 0.5316 | 0.0473 | 0.0473 |
| Distance(mi) | [Pressure(in), WindSpeed(mph), Visibility(mi)] | 4 | 0.5317 | 0.0471 | 0.0471 |
| Distance(mi) | [Pressure(in), Visibility(mi), NeutralWeather] | 4 | 0.5319 | 0.0464 | 0.0464 |
| Distance(mi) | [Pressure(in), Visibility(mi)] | 3 | 0.5319 | 0.0463 | 0.0463 |
| Distance(mi) | [WindChill(F), Pressure(in), BadWeather] | 4 | 0.5319 | 0.0461 | 0.0461 |
| Distance(mi) | [Temperature(F), WindChill(F), Visibility(mi)] | 4 | 0.532 | 0.046 | 0.046 |

### **Most Influential Weather Variables for Distance(mi)**
- **Pressure(in)**: 9 times
- **Visibility(mi)**: 9 times
- **WindChill(F)**: 3 times
- **Temperature(F)**: 2 times
- **BadWeather**: 2 times
- **Humidity(%)**: 1 time
- **GoodWeather**: 1 time
- **WindSpeed(mph)**: 1 time
- **NeutralWeather**: 1 time

---

## Analyzing Weather Variables for Traffic Target: **Severity**

| traffictarget | predictorslist | significantvars | RMSE | R² | AdjR² |
|--------------|-----------------|-----------------|------|------|-------|
| Severity | [Pressure(in), Humidity(%), BadWeather] | 4 | 0.4307 | 0.0081 | 0.0081 |
| Severity | [Pressure(in), Humidity(%), Visibility(mi)] | 4 | 0.4307 | 0.0081 | 0.0081 |
| Severity | [WindChill(F), Pressure(in), Humidity(%)] | 4 | 0.4307 | 0.008 | 0.008 |
| Severity | [Pressure(in), Humidity(%), NeutralWeather] | 4 | 0.4307 | 0.0079 | 0.0079 |
| Severity | [Temperature(F), Pressure(in), Humidity(%)] | 4 | 0.4307 | 0.0079 | 0.0079 |

### **Most Influential Weather Variables for Severity**
- **Pressure(in)**: 10 times
- **Humidity(%)**: 8 times
- **WindChill(F)**: 3 times
- **Visibility(mi)**: 2 times
- **Temperature(F)**: 2 times
- **BadWeather**: 1 time
- **NeutralWeather**: 1 time
- **GoodWeather**: 1 time
- **WindSpeed(mph)**: 1 time

---

## **Global Influence: Most Influential Weather Variables Across All Targets**
| Weather Variable | Frequency |
|-----------------|-----------|
| **Pressure(in)** | **47 times** |
| **Visibility(mi)** | **24 times** |
| **Humidity(%)** | **20 times** |
| **WindChill(F)** | **19 times** |
| **Temperature(F)** | **10 times** |
| **BadWeather** | **9 times** |
| **WindSpeed(mph)** | **6 times** |
| **GoodWeather** | **5 times** |
| **NeutralWeather** | **5 times** |


--------------------------------------------------------


# MAX 2 PREDICTORS RESULTS 


| traffic_target | predictors_list |
|---------------|----------------|
| SpeedNumeric | [Visibility(mi)] |
| SpeedNumeric | [WindSpeed(mph)] |
| SpeedNumeric | [Humidity(%)] |
| SpeedNumeric | [Pressure(in)] |
| SpeedNumeric | [WindChill(F)] |
| ... | ... |
| Distance(mi) | [Humidity(%), WindChill(F)] |
| Distance(mi) | [Humidity(%), Temperature(F)] |
| Distance(mi) | [Pressure(in), WindChill(F)] |
| Distance(mi) | [Pressure(in), Temperature(F)] |
| Distance(mi) | [WindChill(F), Temperature(F)] |

Total: **105 rows**  
✅ All rows have correct permutations.

---

## Analyzing Weather Variables for Traffic Target: **Distance(mi)**

| traffictarget | predictorslist | significantvars | R² |
|--------------|-----------------|-----------------|----|
| Distance(mi) | [Visibility(mi), Pressure(in)] | 2 | 0.0463 |
| Distance(mi) | [Visibility(mi), WindChill(F)] | 2 | 0.0429 |
| Distance(mi) | [Visibility(mi), Temperature(F)] | 2 | 0.0419 |
| Distance(mi) | [Visibility(mi), WindSpeed(mph)] | 2 | 0.0391 |
| Distance(mi) | [Visibility(mi), Humidity(%)] | 2 | 0.0388 |
| Distance(mi) | [Visibility(mi)] | 1 | 0.0382 |
| Distance(mi) | [Pressure(in), WindChill(F)] | 2 | 0.0303 |
| Distance(mi) | [Pressure(in), Temperature(F)] | 2 | 0.0279 |
| Distance(mi) | [Humidity(%), Pressure(in)] | 2 | 0.0266 |
| Distance(mi) | [WindChill(F), Temperature(F)] | 2 | 0.0266 |

### **Most Influential Weather Variables for Distance(mi)**
- **Visibility(mi)**: 6 times
- **Pressure(in)**: 4 times
- **WindChill(F)**: 3 times
- **Temperature(F)**: 3 times
- **Humidity(%)**: 2 times
- **WindSpeed(mph)**: 1 time

---

## Analyzing Weather Variables for Traffic Target: **Severity**

| traffictarget | predictorslist | significantvars | R² |
|--------------|-----------------|-----------------|----|
| Severity | [Humidity(%), Pressure(in)] | 2 | 0.0079 |
| Severity | [Visibility(mi), Pressure(in)] | 2 | 0.0066 |
| Severity | [Pressure(in), WindChill(F)] | 2 | 0.0061 |
| Severity | [Pressure(in), Temperature(F)] | 2 | 0.0059 |
| Severity | [Pressure(in)] | 1 | 0.0051 |

### **Most Influential Weather Variables for Severity**
- **Pressure(in)**: 6 times
- **Humidity(%)**: 4 times
- **WindChill(F)**: 3 times
- **Temperature(F)**: 3 times
- **Visibility(mi)**: 2 times
- **WindSpeed(mph)**: 1 time

---

## Analyzing Weather Variables for Traffic Target: **DelayFromTypicalTraffic(mins)**

| traffictarget | predictorslist | significantvars | R² |
|--------------|-----------------|-----------------|----|
| DelayFromTypicalTraffic(mins) | [Humidity(%), Pressure(in)] | 2 | 0.0296 |
| DelayFromTypicalTraffic(mins) | [Visibility(mi), Pressure(in)] | 2 | 0.0232 |
| DelayFromTypicalTraffic(mins) | [Visibility(mi), Humidity(%)] | 2 | 0.0169 |
| DelayFromTypicalTraffic(mins) | [Humidity(%), Temperature(F)] | 2 | 0.0156 |
| DelayFromTypicalTraffic(mins) | [Humidity(%), WindChill(F)] | 2 | 0.0151 |

### **Most Influential Weather Variables for DelayFromTypicalTraffic(mins)**
- **Humidity(%)**: 6 times
- **Pressure(in)**: 5 times
- **Visibility(mi)**: 2 times
- **Temperature(F)**: 2 times
- **WindChill(F)**: 2 times
- **WindSpeed(mph)**: 2 times

---

## Analyzing Weather Variables for Traffic Target: **TrafficJamDuration(mins)**

| traffictarget | predictorslist | significantvars | R² |
|--------------|-----------------|-----------------|----|
| TrafficJamDuration(mins) | [Pressure(in), WindChill(F)] | 2 | 0.0479 |
| TrafficJamDuration(mins) | [WindChill(F), Temperature(F)] | 2 | 0.0470 |
| TrafficJamDuration(mins) | [Visibility(mi), WindChill(F)] | 2 | 0.0458 |
| TrafficJamDuration(mins) | [WindSpeed(mph), WindChill(F)] | 2 | 0.0457 |
| TrafficJamDuration(mins) | [Pressure(in), Temperature(F)] | 2 | 0.0456 |

### **Most Influential Weather Variables for TrafficJamDuration(mins)**
- **WindChill(F)**: 6 times
- **Temperature(F)**: 5 times
- **Pressure(in)**: 2 times
- **Visibility(mi)**: 2 times
- **WindSpeed(mph)**: 2 times
- **Humidity(%)**: 1 time

---

## Analyzing Weather Variables for Traffic Target: **SpeedNumeric**

| traffictarget | predictorslist | significantvars | R² |
|--------------|-----------------|-----------------|----|
| SpeedNumeric | [Visibility(mi), Pressure(in)] | 2 | 0.0332 |
| SpeedNumeric | [Visibility(mi), Humidity(%)] | 2 | 0.0234 |
| SpeedNumeric | [Visibility(mi), WindChill(F)] | 2 | 0.0232 |
| SpeedNumeric | [Visibility(mi), Temperature(F)] | 2 | 0.0230 |
| SpeedNumeric | [Visibility(mi), WindSpeed(mph)] | 2 | 0.0229 |

### **Most Influential Weather Variables for SpeedNumeric**
- **Visibility(mi)**: 6 times
- **Pressure(in)**: 5 times
- **Humidity(%)**: 2 times
- **WindChill(F)**: 2 times
- **Temperature(F)**: 2 times
- **WindSpeed(mph)**: 2 times

---

## **Global Influence: Most Influential Weather Variables Across All Targets**
| Weather Variable | Frequency |
|-----------------|-----------|
| **Pressure(in)** | **22 times** |
| **Visibility(mi)** | **18 times** |
| **WindChill(F)** | **16 times** |
| **Temperature(F)** | **15 times** |
| **Humidity(%)** | **15 times** |
| **WindSpeed(mph)** | **8 times** |

