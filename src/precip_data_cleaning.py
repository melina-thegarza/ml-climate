import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
# df = pd.read_csv('Weather Data (US).csv', nrows=1000)
# print(df['PRCP'].isna().sum())
# print(df.head())

# df['PRCP'] = df['PRCP'].interpolate()
# print(df.head())



def precipitation_cleaning():
    df = pd.read_csv('precipitation_data.csv')
    print(f"prefiltered length: {len(df)}")

    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d %H:%M')
    #remove missing or invalid data for precipitation
    df = df[df['HPCP'] != 999.99]
    df['HPCP'] = pd.to_numeric(df['HPCP'], errors='coerce')
    df = df.dropna(subset=['HPCP'])

    #create additional features for time-based analysis
    df['hour'] = df['DATE'].dt.hour
    df['day_of_week'] = df['DATE'].dt.dayofweek
    df['month'] = df['DATE'].dt.month


    pd.set_option('display.max_columns', None) 
    print(df.head())
    print(f"filtered length: {len(df)}")

    return df

sns.set(style="whitegrid")



def plot_per_station(df):
    station_mapping = dict(zip(df['STATION'], df['STATION_NAME']))

    for station in df['STATION'].unique():
        station_data = df[df['STATION'] == station]
        station_data['DATE'] = pd.to_datetime(station_data['DATE'], format='%Y%m%d %H:%M')

        fig, axes = plt.subplots(2, 2, figsize=(24, 10))

        # plot 1: precipitation over time
        axes[0, 0].plot(station_data['DATE'], station_data['HPCP'], label=f"Precipitation for {station_mapping[station]}", color="blue")
        axes[0, 0].set_title(f"Precipitation over Time for {station_mapping[station]}", fontsize=14)
        axes[0, 0].set_xlabel("Date")
        axes[0, 0].set_ylabel("Precipitation (inches)")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # plot 2: monthly precipitation (seasonal trend)
        station_data['Month'] = station_data['DATE'].dt.month
        monthly_precip = station_data.groupby('Month')['HPCP'].sum()
        axes[0, 1].bar(monthly_precip.index, monthly_precip.values, color="green")
        axes[0, 1].set_title(f"Monthly Precipitation for {station_mapping[station]}", fontsize=14)
        axes[0, 1].set_xlabel("Month")
        axes[0, 1].set_ylabel("Total Precipitation (inches)")
        axes[0, 1].set_xticks(np.arange(1, 13))

        # Plot 3: Seasonal decomposition using seasonal_decompose
        # Set 'DATE' as index for the seasonal_decompose function
        station_data.set_index('DATE', inplace=True)

        # Use seasonal_decompose for the HPCP column, assuming annual seasonality
        result = seasonal_decompose(station_data['HPCP'], model='additive', period=365)  # period=365 for annual seasonality

        # Plot the decomposition result
        axes[1, 0].plot(result.observed, label="Observed Data", color='blue')
        axes[1, 0].set_title(f"Observed Data (Precipitation) for {station_mapping[station]}")
        axes[1, 0].set_xlabel("Date")
        axes[1, 0].set_ylabel("Precipitation (inches)")

        axes[1, 1].plot(result.trend, label="Trend", color='green')
        axes[1, 1].set_title(f"Trend Component for {station_mapping[station]}")
        axes[1, 1].set_xlabel("Date")
        axes[1, 1].set_ylabel("Precipitation (inches)")

        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        # plot 4: autocorrelation plot
        # autocorrelation helps to see if there are any correlations with previous values (helpful for ARIMA models)
        fig_acf, ax_acf = plt.subplots(figsize=(10, 6))
        autocorrelation_plot(station_data['HPCP'], ax=ax_acf)
        ax_acf.set_title(f"Autocorrelation for {station_mapping[station]}", fontsize=14)

        plt.tight_layout()
        plt.show()
   



def main():
    print("---PRECIPITATION DATA FOR MA---\n")
    cleaned_df = precipitation_cleaning()
    plot_per_station(cleaned_df)
    print("------------------------\n")


if __name__ == '__main__':
    main()