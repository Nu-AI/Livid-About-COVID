import geopandas as gpd
import pandas as pd
import geojson
import plotly.express as px
from urllib.request import urlopen

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#pd.set_option('display.max_colwidth', -1)

mapbox_style = "mapbox://styles/plotlymapbox/cjvprkf3t1kns1cqjxuxmwixz"
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

temp = gpd.read_file("us-counties.json")
temp['id'] = temp['id'].apply(lambda x: x.zfill(5))

# #us_counties_data = pd.read_csv("us-counties.csv")
formatted_data = pd.read_csv("formatted_all_data.csv",dtype={"fips":str})

# print (temp.head(10))
# print (formatted_data.head(10))

formatted_data['fips'] = formatted_data['fips'].apply(lambda x:str(x).zfill(5))
date_list = formatted_data['date'].unique().tolist()
merged_df= temp.merge(formatted_data,right_on='fips', left_on='id')

#print (merged_df.head(10))

layer_slice = merged_df[merged_df['date']==date_list[0]]
#print (layer_slice.head(10), len(list(layer_slice['fips'].values)))

for date in date_list:
    layer_slice = merged_df[merged_df['date']==date]
    print (layer_slice.keys())
    layer_slice = layer_slice.drop(['Residential', 'Unnamed: 0', 'Index'], axis=1)
    print (layer_slice.keys())
    layer_slice.to_file("{0}.geojson".format(date), driver='GeoJSON')

with open ("2020-04-27.geojson","r") as readfile:
    geojson_file = geojson.load(readfile)
print (geojson_file['features'][0]['properties'])

# with urlopen("https://raw.githubusercontent.com/jackparmer/mapbox-counties/master/2015/2.1-4.geojson") as response:
#     sample_file= geojson.load(response)
#
# print (sample_file['features'][0])

formatted_data = formatted_data[formatted_data['date']=="2020-04-27"]



#formatted_data = formatted_data[['fips','Cases']]
#
# df_full_data = pd.read_csv("age_adjusted_death_rate_no_quotes.csv")
# df_full_data["County Code"] = df_full_data["County Code"].apply(
#     lambda x: str(x).zfill(5)
# )
# df_full_data["County"] = (
#     df_full_data["Unnamed: 0"] + ", " + df_full_data.County.map(str)
# )
# df_full_data = df_full_data[df_full_data['Year']==2015]
#print (df_full_data.head(20))
#print (formatted_data.head(20))


fig = px.choropleth_mapbox(formatted_data, geojson=geojson_file, locations='fips', color='Retail & recreation',
                           featureidkey="properties.id",
                           color_continuous_scale="Viridis",
                           mapbox_style="carto-darkmatter",
                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5,
                           labels={'Retail & recreation':'Number of cases'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
print ("completed")
fig.write_image("fig_test_2.png")
#img_bytes  = fig.to_image(format='png')
print ("completed")



#import plotly.express as px
#
# df = px.data.election()
# geojson = px.data.election_geojson()
#
# fig = px.choropleth_mapbox(df, geojson=geojson, color="Bergeron",
#                            locations="district", featureidkey="properties.district",
#                            center={"lat": 45.5517, "lon": -73.7073},
#                            mapbox_style="carto-positron", zoom=9)
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()
#
#print ("completed")
