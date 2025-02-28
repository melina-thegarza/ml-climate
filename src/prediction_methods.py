import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# use this to eventualy predict natural disasters
def weather_data():
    #show top 100 rows
    df = pd.read_csv('src/Weather Data (US).csv', nrows=1000)  # Load only the first 1000 rows
    print(df)

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']))
    norm = plt.Normalize(vmin=df['PRCP'].min(), vmax=df['PRCP'].max()) 
    cmap = cm.get_cmap('coolwarm')  
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    us = world[world['NAME'] == 'United States of America']

    fig, ax = plt.subplots(figsize=(12, 8))
    us.plot(ax=ax, color='lightgrey')
    gdf.plot(ax=ax, color=cmap(norm(gdf['PRCP'])), marker='o', markersize=50)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Rainfall (PRCP)')  

    plt.title("Weather Stations based on Longitude and Latitude", fontsize=15)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig('etc/precipitation_data_test.png')
    plt.show()


  
   

def main():
    print("---WEATHER DATA---\n")
    weather_data()
    print("------------------------\n")


if __name__ == '__main__':
    main()