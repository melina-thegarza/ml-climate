import requests
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import geopandas as gpd
import matplotlib.cm as cm
from dotenv import load_dotenv


load_dotenv()

# yahoo finance api
# see how stocks fluctuated based on natural disaster occurrences
# note: output for stock value in etc/stock.json


def yahoo_api():
    # LGI Homes stock profile
    url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/modules"
    querystring = {"ticker": "LGIH", "module": "asset-profile"}
    # headers = {
    #     "x-rapidapi-key": os.getenv('X-RAPIDAPI-KEY'),
    #     "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com"
    # }
    # response = requests.get(url, headers=headers, params=querystring)
    # print(response.json())

    # LGI Homes stock value , 3 month interval
    url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/history"
    querystring = {"symbol": "LGIH",
                   "interval": "3mo", "diffandsplits": "false"}
    # headers = {
    #     "x-rapidapi-key":  os.getenv('X-RAPIDAPI-KEY'),
    #     "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com"
    # }
    # SAVED response in src/stock.json
    # response = requests.get(url, headers=headers, params=querystring)
    # print(response.json())

    # visualize stock data
    # Extract date and close values
    dates = []
    prices = []
    with open('src/stock.json', 'r') as file:
        data = json.load(file)
    for key, value in data["body"].items():
        dates.append(datetime.strptime(value["date"], "%d-%m-%Y"))
        prices.append(value["close"])

    df = pd.DataFrame({"Date": dates, "Close Price": prices})

    # Plot the stock trend
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Close Price"],
             marker='o', linestyle='-', color='b')
    plt.title(f"Stock Price Trend for {data['meta']['symbol']}")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('etc/stock_plot.png')
    plt.show()

# housing data


def zillow():
    pass

# historical flood data


def flood_data():

    df = pd.read_excel('src/FloodArchive.xlsx', engine='openpyxl')
    print(df.head())

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['long'], df['lat']))
    norm = plt.Normalize(vmin=df['Severity'].min(), vmax=df['Severity'].max())
    cmap = cm.get_cmap('coolwarm')

    # Plot the map
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"

    world = gpd.read_file(url)

    fig, ax = plt.subplots(figsize=(12, 8))
    world.plot(ax=ax, color='lightgrey')
    gdf.plot(ax=ax, color=cmap(
        norm(gdf['Severity'])), marker='o', markersize=50)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Flood Severity')

    # Add titles and labels
    plt.title("Flood Locations (1985-2021)", fontsize=15)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Show the plot
    plt.show()


def main():
    print("---YAHOO FINANCE DATA---\n")
    yahoo_api()
    print("------------------------\n")

    print("------FLOOD DATA-------\n")
    flood_data()
    print("------------------------\n")


if __name__ == '__main__':
    main()
