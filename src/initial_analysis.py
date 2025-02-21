import requests
import os, json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

    
load_dotenv()

# yahoo finance api
# see how stocks fluctuated based on natural disaster occurrences
# note: output for stock value in etc/stock.json
def yahoo_api():
    # LGI Homes stock profile
    url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/modules"
    querystring = {"ticker":"LGIH","module":"asset-profile"}
    headers = {
        "x-rapidapi-key": os.getenv('X-RAPIDAPI-KEY'),
        "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com"
    }
    # response = requests.get(url, headers=headers, params=querystring)
    # print(response.json())

    # LGI Homes stock value , 3 month interval
    url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/history"
    querystring = {"symbol":"LGIH","interval":"3mo","diffandsplits":"false"}
    headers = {
        "x-rapidapi-key":  os.getenv('X-RAPIDAPI-KEY'),
        "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com"
    }
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
    plt.plot(df["Date"], df["Close Price"], marker='o', linestyle='-', color='b')
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
    pass

# use this to eventualy predict natural disasters
def weather_data():
    pass

def main():
    print("---YAHOO FINANCE DATA---\n")
    yahoo_api()
    print("------------------------\n")


if __name__ == '__main__':
    main()