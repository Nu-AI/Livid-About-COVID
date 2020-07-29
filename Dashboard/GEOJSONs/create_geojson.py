import geopandas as gpd
import pandas as pd
import geojson
import plotly.express as px
from urllib.request import urlopen
from os import path
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

mapbox_style = "mapbox://styles/plotlymapbox/cjvprkf3t1kns1cqjxuxmwixz"
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

# basepath = "Livid-About-COVID\Dashboard"
# filepath = path.abspath(path.join(''))
# print (filepath)
def generate_geojson(path):
    # Reading the us-counties json file
    temp = gpd.read_file("{}\\us-counties.json".format(path))
    # Append the FIPS with previous zeros
    temp['id'] = temp['id'].apply(lambda x: x.zfill(5))
    # Read the data conflation response
    formatted_data = pd.read_csv("formatted_all_data.csv",dtype={"fips":str})

    # Set the FIPS id to be consistent
    formatted_data['fips'] = formatted_data['fips'].apply(lambda x:str(x).zfill(5))
    date_list = formatted_data['date'].unique().tolist()
    print(formatted_data.date.unique().tolist()[-1])
    # Merge the json with the collected data
    merged_df= temp.merge(formatted_data,right_on='fips', left_on='id')


    # Save the different geojsons for all the days
    for date in date_list:
        layer_slice = merged_df[merged_df['date']==date]
        # Dropping unused columns
        layer_slice = layer_slice.drop(['Residential', 'Unnamed: 0', 'Index'], axis=1)
        # Saving the file
        layer_slice.to_file("{}\\{}.geojson".format(path,date), driver='GeoJSON')

# Sample reading of a file to check for generation
# with open ("2020-04-27.geojson","r") as readfile:
#     geojson_file = geojson.load(readfile)
# print (geojson_file['features'][0]['properties'])
#
#
#
# # Sample figure to check whether the geojson files were generated
# fig = px.choropleth_mapbox(formatted_data, geojson=geojson_file, locations='fips', color='Retail & recreation',
#                            featureidkey="properties.id",
#                            color_continuous_scale="Viridis",
#                            mapbox_style="carto-darkmatter",
#                            zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
#                            opacity=0.5,
#                            labels={'Retail & recreation':'Number of cases'}
#                           )
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()
# generate_geojson(filepath)
# print ("completed")
# # Saving the figure in the base directory
# # fig.write_image("fig_test_2.png")
# print ("completed")

