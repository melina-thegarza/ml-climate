
## **Week 2/17 - 2/23**
## 2/21/2025, Melina Garza
- Collecting data and doing an initial analysis of it. Focusing on real-estate sector initially.
- Data found: 
    - financial/stock data: [Yahoo Finance API](https://rapidapi.com/sparior/api/yahoo-finance15)

    - housing data: [Zillow Housing Data](https://www.zillow.com/research/data/)
    - flood data: [Flood events 1985-2016](https://floodobservatory.colorado.edu/Archives/index.html)
    - weather data: [NCEI US Climate Data (1992-2021)](https://www.kaggle.com/datasets/nachiketkamod/weather-dataset-us/data)
- Started programmtic analysis in `src/initial_analysis.py`

## 2/23/2025, Dhwani Sreenivas
- Trying to find data on housing prices over time in different regions of the US, as well as potentially looking at the housing assets that large companies own to be able to predict how their stock prices will be affected
- Data found:
    - housing price index data (1991 - 2024): [Federal Housing Finance Agency](https://www.fhfa.gov/data/hpi/datasets?tab=regional-hpi)
    - investigating company asset data (need demo): [Sovereign Wealth Fund Institute](https://www.swfinstitute.org/fund-rankings/real-estate-company)
- Drew diagram mapping our initial hypothesis/abstract

<img src="./etc/abstract_diagram.png" alt="drawing" width="700"/>


## **Week 2/24 - 3/2**
## 2/28/2025, Melina Garza
This week's focus is exploring different weather prediction methods and natural disaster prediction methods. Also took a snapshot of the weather data and exploring how to use Google Cloud to run models.

**Weather Data:**

Columns
<img src="./etc/weather_data_columns.png" alt="drawing" width="500"/>
<img src="./etc/weather_data_snapshot_example.png" alt="drawing" width="500"/>


**Weather Prediction Methods:**

Going to test these out for predicting rainfall (eventually use to predict floods):

1.  ARIMA (AutoRegressive Integrated Moving Average)
    - for short-term predicitons(daily/weekly)

2. LSTM 
    - for long-term rainfall trends/dependencies



**Disaster Prediction Methods:**

TBD

## 3/2/2025, Dhwani Sreenivas
- Focusing on 3 methods of weather forecasting, using rainfall data
- Current kaggle dataset's temporal resolution: daily, spatial resolution: 56950 unique weather stations seen in "precipitation_data_test.png"
- Kaggle dataset: 688 NaNs for precip in first 1000 entries
- Linearly interpolated the data (not best solution)

- Alternative data found:
    - rainfall data: [hourly rainfall 1900-2014](https://www.ncdc.noaa.gov/cdo-web/search?datasetid=PRECIP_HLY#)
    - time based resolution: available in 15 minute increments and hourly increments (hourly sample uploaded)
    - spatial resolution: available by zip codes, cities, states, countries in the US

    - aggregating this data: TBD


## **Week 3/3 - 3/10**
## 3/4/2025, Melina Garza
This week's focus is exploring different weather prediction methods using different models.

**DATA**(for Massachuseets only, eventually will explore other areas): `src/precipitation_data.csv`

**CODE:**`src/ml_climate_precipitation_prediction.ipynb`
- note: ran code in google colab, kept on getting issues locally with libraries

[COLUMNS:](https://www.ncei.noaa.gov/pub/data/cdo/documentation/PRECIP_HLY_documentation.pdf)
`STATION,STATION_NAME,ELEVATION,LATITUDE,LONGITUDE,DATE,HPCP,Measurement Flag,Quality Flag`

**HPCP:** The amount of precipitation recorded at the station for the hour ending at the
time specified for DATE above given in hundredths of inches or tenths of millimeters
depending on user’s specification of standard or metric units. The values 99999 means
the data value is missing. Hours with no precipitation are not shown.

**STEPS**
* **1  Clean the data:  `preprocess_for_ml()`**
* 2   Create plots for each station(for visualiaztion of data, commented out when running entire pipeline): `plot_per_station()`

    - 1.  precipitation over time: raw-time series data
    - 2. monthly precipitation: seasonal trends, cyclical patterns
    - 3. seasonal decomposition: isolates trend, seasonal, and residual components
    - 4. autocorrelation: relationships between past and future values, for models like ARIMA and LSTM

    Example for Blue Hill LCD MA Station:

    <img src="./etc/precip_data_ex_overview_graphs.png" alt="drawing" width="500"/>
    <img src="./etc/autocorrelation_precip_data_ex.png" alt="drawing" width="500"/>
    

* **3 Lag Features:** capture temporal dependencies(past values of a variable used as input features to predict future values): `add_lag_features()`

* **4 Modeling**
    - **Regression for Precipitation Prediction**: Decision Trees & XGBoost Models are used to predict the amount of precipitation
    - **Classification**: binary classification to predict whether the precipitaiton exceeds a certain threshold(e.g. 0.3 inches), indicating heavy rain


* **5 RESULTS:**
    1) Without elevation as a feature
        <img src="./etc/modeling_without_elevation_results.png" alt="drawing" width="500"/>
       
        
    2) With elevation as a feature
          <img src="./etc/modeling_with_elevation_results.png" alt="drawing" width="500"/>
    

* **6 Analysis/Next Steps**
    - ...      