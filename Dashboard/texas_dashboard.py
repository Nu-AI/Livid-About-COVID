import sys
import urllib.request
import os
from os import path

import numpy as np
import pandas as pd

import dash
import cufflinks   # noqa
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_daq as daq
import geojson
import plotly.express as px
import plotly.graph_objects as go

basepath = os.path.join(os.path.dirname(__file__))
ROOT_DIR = os.path.join(basepath, '..')
sys.path.append(ROOT_DIR)
import parameters as param
from scripts import fit_bexar_mask
from Dashboard.GEOJSONs.create_geojson import generate_geojson

basepath = os.path.join(ROOT_DIR, 'Dashboard')
filepath = path.abspath(path.join(basepath, 'GEOJSONs'))

# Get the data from the data collection module
formatted_data = pd.read_csv('formatted_all_data.csv', dtype={'fips': str})

# Generating the GEOJSON files
generate_geojson(filepath, formatted_data)

# Appending zeros to fips ids
formatted_data['fips'] = formatted_data['fips'].apply(lambda x: str(x).zfill(5))

# Setting the mapbox details
mapbox_access_token = 'pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A'
mapbox_style = 'mapbox://styles/plotlymapbox/cjvprkf3t1kns1cqjxuxmwixz'
px.set_mapbox_access_token(mapbox_access_token)

# The dash app config
# https://dash.plotly.com/deployment
app = dash.Dash(
    __name__,
    meta_tags=[
        {'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}
    ],
)
server = app.server

# Reading the cases from the counties
county_cases_df = pd.read_csv(urllib.request.urlopen(
    'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'))
county_cases_df['fips'] = county_cases_df['fips'].fillna(0).astype(np.int64)
county_cases_df['fips'] = county_cases_df['fips'].apply(
    lambda x: str(x).zfill(5))

# Getting the total number of cases in the state ( Texas for this case)
state_cases_df = pd.read_csv(urllib.request.urlopen(
    'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'))

texas_df = state_cases_df[state_cases_df.state == 'Texas']

# Get the model predictions for the counties, default set to Bexar county
df = pd.read_csv("formatted_all_data.csv")
date_list = sorted(df['date'].unique().tolist())

dates = df['date'].unique().tolist()

# Default model predictions
actives, totals = fit_bexar_mask.pipeline(
    param, data=df[df['County'] == 'Bexar County'].reset_index(),
    county='Bexar County')

# Date list in the slider menu
DATE_MODIFIED = [dates[::-1][i] for i in range(len(dates)) if i % 10 == 0][::-1]

# The screen layout
app.layout = html.Div(
    id='root',
    children=[
        html.Div
            (
            id='header',
            style={'fontSize': 16, 'display': 'inline-block'},
            children=[
                html.H4(children="COVID-19 predictions in counties in Texas"),
                html.P(
                    id="description",
                    style={'fontSize': 20},
                    children=" These show the number of active cases registered and observed for coronavirus "
                             "for these given day milestones.",
                ),
            ],
        ),
        html.Div
            (
            id='test_right',
            style={'display': 'inline-block'},
            children=[
                daq.LEDDisplay(
                    id="operator-led",
                    value=texas_df.cases.tolist()[-1],
                    color="#2cfec1",
                    label={'label': "TOTAL CASES",
                           'style': {
                               'fontSize': 24,
                               'color': '#2cfec1'
                           }
                           },
                    backgroundColor="#1e2130",
                    size=50,
                ),

            ],
        ),
        html.Div
            (
            id='test_right2',
            style={'display': 'inline-block'},
            children=[
                daq.LEDDisplay(
                    id="operator-led2",
                    value=texas_df.deaths.tolist()[-1],
                    color="#2cfec1",
                    label={'label': "DEATHS",
                           'style': {
                               'fontSize': 24,
                               'color': '#2cfec1'
                           }
                           },
                    backgroundColor="#1e2130",
                    size=50,
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
                                            "label": "-".join(
                                                str(DATE_MODIFIED[date]).split(
                                                    "-")[1:]),
                                            "style": {"color": "#7fafdf",
                                                      'fontSize': 16},
                                        } for date in range(len(DATE_MODIFIED))

                                    },
                                    step=None,

                                ),
                            ],
                        ),

                        html.Div
                            (
                            id='heatmap-container',
                            style={'fontSize': 20},
                            children=
                            [
                                html.P
                                    (
                                    "Heatmap of mobility \
                                    in Texas counties on selected date {0}".format(
                                        DATE_MODIFIED[0]),
                                    id="heatmap-title",
                                ),
                                dcc.Dropdown
                                    (
                                    options=
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
                                    value="Residential",
                                    id="chart-dropdown",
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
                    style={'fontSize': 20},
                    children=
                    [
                        html.P
                            (
                            "Mobility percentage with respect to baseline",
                            id="prediction_title",
                        ),
                        dcc.RadioItems(
                            id="Radio_block1",
                            options=[
                                {'label': '25%', 'value': '25'},
                                {'label': '50%', 'value': '50'},
                                {'label': '75%', 'value': '75'},
                                {'label': '100%', 'value': '100'},
                            ],
                            value='25',
                            labelStyle={'display': 'inline-block',
                                        'margin': '5px'}
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


# The map display callback function
@app.callback(
    Output('county_chloropleth', 'figure'),
    [Input('date_slider', 'value'),
     Input('chart-dropdown', 'value')]
)
def plot_map(selected_date, selected_mob):
    new_path = path.abspath(path.join('GEOJSONs'))
    path_new = path.abspath(
        path.join(new_path, str(DATE_MODIFIED[selected_date])))
    with open('{}.geojson'.format(path_new)) as readfile:
        geojson_file = geojson.load(readfile)

    px.set_mapbox_access_token(mapbox_access_token)
    data_copy = formatted_data.copy()
    target = str(selected_mob)
    data_copy = data_copy[['County', target, 'fips']]

    fig = px.choropleth_mapbox(data_copy, geojson=geojson_file,
                               locations='fips', color=target,
                               featureidkey='properties.id',
                               color_continuous_scale='Inferno',
                               mapbox_style=mapbox_style,
                               hover_data=['County', target, 'fips'],
                               zoom=4.8, center={'lat': 31.3, 'lon': -99.2},
                               opacity=0.5,
                               labels={'Cases': 'Number of cases'}
                               )

    fig.update_layout(
        margin=dict(
            l=50,
            r=50,
            b=50,
            pad=2
        ),
        font=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        ),
        paper_bgcolor='#252e3f',
        clickmode='event+select'
    )
    return fig


# Default graph layout settings
def set_figure_template(fig_data, fig_layout):
    fig_data[0]['marker']['color'] = '#2cfec1'
    fig_data[0]['marker']['opacity'] = 1
    fig_data[0]['marker']['line']['width'] = 0
    fig_layout['paper_bgcolor'] = '#1f2630'
    fig_layout['plot_bgcolor'] = '#1f2630'
    fig_layout['font']['color'] = '#2cfec1'
    fig_layout['title']['font']['color'] = '#2cfec1'
    fig_layout['xaxis']['tickfont']['color'] = '#2cfec1'
    fig_layout['yaxis']['tickfont']['color'] = '#2cfec1'
    fig_layout['xaxis']['gridcolor'] = '#5b5b5b'
    fig_layout['yaxis']['gridcolor'] = '#5b5b5b'
    fig_layout['margin']['t'] = 75
    fig_layout['margin']['r'] = 50
    fig_layout['margin']['b'] = 100
    fig_layout['margin']['l'] = 50


# Graph callback function
@app.callback(
    [Output('slider_graph', 'figure'),
     Output('slider_graph_2', 'figure'),
     Output('slider_graph_3', 'figure')],
    [Input('date_slider', 'value'),
     Input('Radio_block1', 'value'),
     Input('county_chloropleth', 'clickData')]
)
def plot_data(selected_date, selected_percent, clickData):
    if clickData is not None:
        temp = clickData['points'][0]['customdata']
        updated_county = temp[0]
    else:
        updated_county = 'Bexar County'

    print(selected_date, '****', dates[int(selected_date)])

    mob_df = df[df['County'] == updated_county]
    mob_df.reset_index(drop=True, inplace=True)
    county = mob_df.County.unique().tolist()[0]  # TODO(tmp)
    actives, totals = fit_bexar_mask.pipeline(param, data=mob_df,
                                              county=county)
    totalpred_df = pd.DataFrame.from_dict(totals[0.1])
    total_predicted_cases_0_05 = pd.DataFrame.from_dict(totals[0.05])
    total_predicted_cases_0_3 = pd.DataFrame.from_dict(totals[0.3])

    active_df = pd.DataFrame.from_dict(actives[0.1])
    active_predicted_cases_0_05 = pd.DataFrame.from_dict(actives[0.05])
    active_predicted_cases_0_3 = pd.DataFrame.from_dict(actives[0.3])

    mob_df = mob_df[['date', 'Retail & recreation',
                     'Grocery & pharmacy', 'Parks', 'Transit stations',
                     'Workplace', 'Residential']]
    mob_df = mob_df.set_index('date')

    fig3 = go.Figure()
    fig3 = cont_error_bar(fig3, totalpred_df['date'],
                          total_predicted_cases_0_3[int(selected_percent)],
                          totalpred_df[int(selected_percent)],
                          total_predicted_cases_0_05[int(selected_percent)],
                          selected_percent)
    fig3_layout = fig3['layout']
    fig3_data = fig3['data']
    fig3.update_layout(
        title='Total predicted cases based on the reporting rate',
        showlegend=False
    )
    fig = go.Figure()

    fig = cont_error_bar(fig, active_df['date'],
                         active_predicted_cases_0_3[int(selected_percent)],
                         active_df[int(selected_percent)],
                         active_predicted_cases_0_05[int(selected_percent)],
                         selected_percent)
    fig_layout = fig["layout"]
    fig_data = fig["data"]
    fig.update_layout(
        title='Active predicted cases based on the reporting rate',
        showlegend=False
    )
    fig2 = mob_df.iplot(asFigure=True, title="Average Mobility over time")
    fig2_layout = fig2["layout"]
    fig2_data = fig2["data"]

    fig2.update_layout(
        title='Mobility over time in {}'.format(updated_county),
        legend=dict(bgcolor='#1f2630')
    )
    set_figure_template(fig_data, fig_layout)
    set_figure_template(fig2_data, fig2_layout)
    set_figure_template(fig3_data, fig3_layout)

    return fig, fig3, fig2


# Mobility transformation plot settings in the graph
def cont_error_bar(fig, x, y1, y2, y3, selected_percent):
    if selected_percent == '25':
        color = 'rgb(50, 171, 96)'
        fillcolor = 'rgba(50, 171, 96,0.2)'
    elif selected_percent == '50':
        color = 'rgb(55, 128, 191)'
        fillcolor = 'rgba(55, 128, 191,0.2)'
    elif selected_percent == '75':
        color = 'rgb(255, 153, 51)'
        fillcolor = 'rgba(255, 153, 51, 0.2)'
    else:
        color = 'rgb(219, 64, 82)'
        fillcolor = 'rgba(219, 64, 82, 0.2)'

    fig.add_trace(go.Scatter(x=x,
                             y=y1,
                             marker=dict(color='#444'),
                             line=dict(width=0),
                             fillcolor=fillcolor,
                             ))

    fig.add_trace(go.Scatter(x=x,
                             y=y2,
                             mode='lines',
                             line=dict(color=color),
                             fillcolor=fillcolor,
                             fill='tonexty'))

    fig.add_trace(go.Scatter(x=x,
                             y=y3,
                             marker=dict(color='#444'),
                             line=dict(width=0),
                             mode='lines',
                             fillcolor=fillcolor,
                             fill='tonexty'
                             ))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
