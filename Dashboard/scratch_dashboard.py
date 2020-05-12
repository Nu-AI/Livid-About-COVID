import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
import cufflinks as cf
import urllib.request
#import retrieve_data_v2 as rc
#import GEOJSONs
import numpy as np
import plotly.express as px
import geojson
from os import path

basepath = path.dirname("scratch_dashboard.py")
filepath = path.abspath(path.join(basepath,"GEOJSONs/"))
#geojson_file = geojson.load("2020-03-18.geojson")
# with open ("2020-04-25.geojson","r") as readfile:
#     geojson_file = geojson.load(readfile)
# print (geojson_file['features'][0])

formatted_data = pd.read_csv("formatted_all_data.csv",dtype={"fips":str})
# formatterd_data_orig = formatted_data.copy()
# print (temp.head(10))
# print (formatted_data.head(10))
formatted_data['fips'] = formatted_data['fips'].apply(lambda x:str(x).zfill(5))

# formatted_data = formatted_data[formatted_data['date']=="2020-04-25"]

# formatted_data = formatted_data[['fips','Cases'x]]

print (formatted_data.head(10))
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"
mapbox_style = "mapbox://styles/plotlymapbox/cjvprkf3t1kns1cqjxuxmwixz"
px.set_mapbox_access_token(mapbox_access_token)

paramdict = {}
paramdict['country'] = 'United States'
paramdict['states'] = ['Texas']
paramdict['counties'] = ['Bexar County']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server
county_list = ['Bexar County', 'Travis County', 'Dallas County', 'Tarrant County', 'Harris County']
county_cases_df = pd.read_csv(urllib.request.urlopen(
    "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"))
# print (county_cases_df['fips'].head(10), county_cases_df.dtypes)
county_cases_df['fips'] = county_cases_df['fips'].fillna(0).astype(np.int64)
county_cases_df['fips'] = county_cases_df['fips'].apply(lambda x: str(x).zfill(5))
# print (county_cases_df['fips'].head(10))
# df = rc.get_data(paramdict)
df = pd.read_csv("formatted_all_data.csv")
date_list = sorted(df['date'].unique().tolist())

dates = df['date'].unique().tolist()
# print (len(dates))

DATE_MODIFIED = [dates[i] for i in range(len(dates)) if i % 5 == 0]
print (DATE_MODIFIED)
app.layout = html.Div(
    id='root',
    children=[
        html.Div
        (
            id='header',
            children=[
                html.H4(children="COVID-19 predictions in counties in Texas"),
                html.P(
                    id="description",
                    children=" These show the number of active cases registered and observed for coronavirus "
                             "for these given day milestones.",
                ),
            ],
        ),
        html.Div
        (
            id='app-container',
            children=
            [
                html.Div
                (
                    id="left-column",
                    children=
                    [
                        html.Div
                        (
                            id="slider_container",
                            children=
                            [
                                dcc.Slider(
                                    id='date_slider',
                                    min=0,
                                    max=len(DATE_MODIFIED),
                                    value=0,
                                    marks={
                                        str(date): {
                                            "label": "-".join(str(DATE_MODIFIED[date]).split("-")[1:]),
                                            "style": {"color": "#7fafdf"},
                                        } for date in range(len(DATE_MODIFIED))

                                    },
                                    step=None,

                                ),
                            ],
                        ),
                        html.Div
                        (
                            id ='heatmap-container',
                            children=
                            [
                                html.P
                                (
                                "Heatmap of cases \
                                in Texas counties on selected date {0}".format(DATE_MODIFIED[0]),
                                id="heatmap-title",
                                ),
                                dcc.Dropdown
                                (
                                    options =
                                    [
                                        {
                                          "label": "Retail & recreation",
                                          "value": "Retail & recreation",
                                        },
                                        {
                                          "label": "Parks",
                                          "value": "Parks",
                                        },
                                        {
                                          "label": "Residential",
                                          "value": "Residential",
                                        },
                                        {
                                          "label": "Grocery & Pharmacy",
                                          "value": "Grocery & pharmacy",
                                        },
                                        {
                                          "label": "Transit Stations",
                                          "value": "Transit stations",
                                        },
                                        {
                                          "label": "Workplace",
                                          "value": "Workplace",
                                        },
                                    ],
                                        value = "Residential",
                                        id = "chart-dropdown",
                                ),
                                dcc.Graph
                                (
                                    id="county_chloropleth"
                                ),
                                dcc.Graph
                                (
                                    id='slider_graph_3',
                                    figure=dict(
                                        data=[dict(x=0, y=0)],
                                        layout=dict(
                                            paper_bgcolor="#F4F4F8",
                                            plot_bgcolor="#F4F4F8",
                                            autofill=True,
                                            margin=dict(t=75, r=50, b=50, l=50),
                                        ),
                                    ),
                                 ),
                            ],
                        ),
                    ],
                ),
                html.Div
                (
                    id="graph-container",
                    children=
                    [
                        html.P
                        (
                            "Mobility percentage with respect to baseline",
                            id = "prediction_title",
                        ),
                        dcc.RadioItems(
                            id="Radio_block1",
                            options=[
                                {'label': '0%','value':'0'},
                                {'label': '25%','value':'1'},
                                {'label': '75%','value':'2'},
                                {'label': '100%','value':'3'},
                                ],
                                value='0',
                                labelStyle = {'display': 'inline-block', 'margin': '5px'}
                        ),
                        dcc.Graph
                        (id='slider_graph',
                            figure=dict(
                                data=[dict(x=0, y=0)],
                                layout=dict(
                                    paper_bgcolor="#F4F4F8",
                                    plot_bgcolor="#F4F4F8",
                                    autofill=True,
                                    margin=dict(t=75, r=50, b=50, l=50),
                                ),
                            ),
                        ),
                        dcc.Graph
                        (id='slider_graph_2',
                            figure=dict(
                                data=[dict(x=0, y=0)],
                                layout=dict(
                                    paper_bgcolor="#F4F4F8",
                                    plot_bgcolor="#F4F4F8",
                                    autofill=True,
                                    margin=dict(t=75, r=50, b=50, l=50),
                                ),
                            ),
                        ),
                    ],
                ),
            ],
        ),
    ],
)

@app.callback(
    Output("county_chloropleth", "figure"),
    [Input("date_slider", "value"),
     Input("chart-dropdown", "value")]
)
def plot_map(selected_date, selected_mob):
    # if selected_date=="2020-04-25":
    #     with open("2020-04-25.geojson", "r") as readfile:
    #         geojson_file = geojson.load(readfile)
    # else:
    path_new = path.abspath(path.join(filepath,str(DATE_MODIFIED[selected_date])))

    #path_new = path.join(filepath,str(DATE_MODIFIED[selected_date]))
    print (path_new, "*********", DATE_MODIFIED[selected_date])
    with open("{}.geojson".format(path_new, "r")) as readfile:
        geojson_file = geojson.load(readfile)

    px.set_mapbox_access_token(mapbox_access_token)
    data_copy = formatted_data.copy()
    target = str(selected_mob)
    data_copy = data_copy[['County',target, 'fips']]

    fig = px.choropleth_mapbox(data_copy, geojson=geojson_file, locations='fips', color=target,
                       featureidkey="properties.id",
                       color_continuous_scale="Inferno",
                       mapbox_style=mapbox_style,
                       hover_data= ['County', target, 'fips'],
                       zoom=4.8, center = {"lat": 31.3, "lon": -99.2},
                       opacity=0.5,
                       labels={'Cases':'Number of cases'}
                )
    title_string = "Heatmap of cases in Texas counties on selected date {0}".format(selected_date)
    fig.update_layout(
        width=800,
        #title= title_string,
        margin=dict(
          l=50,
          r=50,
          b=50,
          t=50,
          pad=2
        ),
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ),
        paper_bgcolor="#252e3f",
        )
    return fig

def set_figure_template(fig_data,fig_layout):
    fig_data[0]["marker"]["color"] = "#2cfec1"
    fig_data[0]["marker"]["opacity"] = 1
    fig_data[0]["marker"]["line"]["width"] = 0
    #fig_data[0]["textposition"] = "outside"
    fig_layout["paper_bgcolor"] = "#1f2630"
    fig_layout["plot_bgcolor"] = "#1f2630"
    fig_layout["font"]["color"] = "#2cfec1"
    fig_layout["title"]["font"]["color"] = "#2cfec1"
    fig_layout["xaxis"]["tickfont"]["color"] = "#2cfec1"
    fig_layout["yaxis"]["tickfont"]["color"] = "#2cfec1"
    fig_layout["xaxis"]["gridcolor"] = "#5b5b5b"
    fig_layout["yaxis"]["gridcolor"] = "#5b5b5b"
    fig_layout["margin"]["t"] = 75
    fig_layout["margin"]["r"] = 50
    fig_layout["margin"]["b"] = 100
    fig_layout["margin"]["l"] = 50


@app.callback(
    [Output("slider_graph", "figure"),
    Output("slider_graph_2", 'figure'),
    Output("slider_graph_3", 'figure')],
    [Input("date_slider", "value"),
     Input("Radio_block1", "value")]
)
def plot_data(selected_date, selected_percent):

    #case_df_filtered = county_cases_df[county_cases_df['date'] == selected_date]
    print(selected_date, "****", dates[int(selected_date)])

    full_df_filtered = df[df['date'] == DATE_MODIFIED[selected_date]]
    if (selected_percent) == "0":
        print ("entered the condition")
    else:
        print ("other conditions")
    mob_df = df[df['County']=='Bexar County']
    mob_df = mob_df[['date','Retail & recreation',
                                    'Grocery & pharmacy', 'Parks', 'Transit stations', 'Workplace', 'Residential']]
    mob_df.reset_index(drop=True, inplace=True)

    #print (mob_df)
    mob_df = mob_df.set_index('date')
    #print (mob_df.keys())

    full_df_filtered = full_df_filtered.loc[full_df_filtered['County'].isin(county_list)]
    required_df = full_df_filtered[['County','Retail & recreation',
                                    'Grocery & pharmacy', 'Parks', 'Transit stations', 'Workplace', 'Residential','Cases', 'Deaths']].reset_index()
    #print(df.loc[df['County'] == 'Bexar County']['Cases'], "\n\n\n*****", required_df['Cases','County'])
    print (required_df[['Cases','County']], "\n\n\n")
    #print(required_df.head(10))
    #fig = required_df.iplot(kind='bar', title="Mobility data per day")

    fig = px.bar(required_df, x="County",y="Cases", title="Cases in the counties")
    fig_layout = fig["layout"]
    fig_data = fig["data"]
    print ("test_value")
    #print (fig_data)

    fig3 = px.bar(required_df, x="County", y="Deaths", title="Deaths in the different counties")
    fig3_layout= fig3["layout"]
    fig3_data = fig3["data"]
    # fig_data[0]["text"] = deaths_or_rate_by_fips.values.tolist()
    fig2 = mob_df.iplot(asFigure=True, title="Average Mobility over time")
    fig2_layout = fig2["layout"]
    fig2_data = fig2["data"]
    fig2.update_layout(
        legend=dict(bgcolor="#1f2630")
    )
    set_figure_template(fig_data,fig_layout)
    set_figure_template(fig2_data,fig2_layout)
    set_figure_template(fig3_data, fig3_layout)

    # fig_data[0]["marker"]["color"] = "#2cfec1"
    # fig_data[0]["marker"]["opacity"] = 1
    # fig_data[0]["marker"]["line"]["width"] = 0
    # fig_data[0]["textposition"] = "outside"
    # fig_layout["paper_bgcolor"] = "#1f2630"
    # fig_layout["plot_bgcolor"] = "#1f2630"
    # fig_layout["font"]["color"] = "#2cfec1"
    # fig_layout["title"]["font"]["color"] = "#2cfec1"
    # fig_layout["xaxis"]["tickfont"]["color"] = "#2cfec1"
    # fig_layout["yaxis"]["tickfont"]["color"] = "#2cfec1"
    # fig_layout["xaxis"]["gridcolor"] = "#5b5b5b"
    # fig_layout["yaxis"]["gridcolor"] = "#5b5b5b"
    # fig_layout["margin"]["t"] = 75
    # fig_layout["margin"]["r"] = 50
    # fig_layout["margin"]["b"] = 100
    # fig_layout["margin"]["l"] = 50

    return fig,fig3,fig2


if __name__ == "__main__":
    app.run_server(debug=True,use_reloader = False)
