import geopandas as gpd
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

mapbox_style = "mapbox://styles/plotlymapbox/cjvprkf3t1kns1cqjxuxmwixz"
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"


def generate_geojson(path, formatted_data):
    # Reading the us-counties json file
    temp = gpd.read_file("{}\\us-counties.json".format(path))
    # Append the FIPS with previous zeros
    temp['id'] = temp['id'].apply(lambda x: x.zfill(5))

    # Set the FIPS id to be consistent
    formatted_data['fips'] = formatted_data['fips'].apply(lambda x: str(x).zfill(5))
    date_list = formatted_data['date'].unique().tolist()
    print(formatted_data.date.unique().tolist()[-1])
    # Merge the json with the collected data
    merged_df = temp.merge(formatted_data, right_on='fips', left_on='id')

    # Save the different geojsons for all the days
    for date in date_list:
        layer_slice = merged_df[merged_df['date'] == date]

        # Dropping unused columns
        layer_slice = layer_slice.drop(['Residential', 'Unnamed: 0', 'Index'], axis=1)

        # Saving the file
        layer_slice.to_file("{}\\{}.geojson".format(path, date), driver='GeoJSON')

    print("Finished geojson generation...")
