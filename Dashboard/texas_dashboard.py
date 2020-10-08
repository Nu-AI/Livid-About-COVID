import glob
import os
import sys
import urllib.request
from os import path

import cufflinks  # noqa
import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import geojson
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

basepath = os.path.join(os.path.dirname(__file__))
ROOT_DIR = os.path.join(basepath, '..')
DASH_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Dashboard'))
PREDS_DIR = os.path.join(DASH_DIR, 'model_predictions')

file_list = glob.glob(os.path.join(PREDS_DIR, '*.json'))
# prediction_file = max(file_list, key=os.path.getctime)
prediction_file = max(file_list)
latest_prediction_filename = os.path.splitext(os.path.basename(prediction_file))[0]
latest_prediction_date = str(latest_prediction_filename).split("_")[2:5]
latest_date = "-".join(i for i in latest_prediction_date)
print(latest_date)
sys.path.append(ROOT_DIR)
sys.path.append(DASH_DIR)

from Dashboard.GEOJSONs.create_geojson import generate_geojson
import json


def read_json(json_path):
    with open(json_path, 'r') as json_data:
        data_dict = json.load(json_data)
    return pd.DataFrame(data_dict)


basepath = os.path.join(ROOT_DIR, 'Dashboard')
filepath = path.abspath(path.join(basepath, 'GEOJSONs'))
directory = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(directory, 'formatted_all_data.csv')
# Get the data from the data collection module
formatted_data = pd.read_csv(filename, dtype={'fips': str})

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

# '''
# app.config.update({
#     # as the proxy server will remove the prefix
#
#     # the front-end will prefix this string to the requests
#     # that are made to the proxy server
#     'requests_pathname_prefix': ''
# })
# '''
app.config.update({
    'url_base_pathname': '',
    'routes_pathname_prefix': '',
    'requests_pathname_prefix': '',
})
app.title = 'SIRNet - COVID-19 Case Forecasts in Texas'

server = app.server

# app.scripts.config.serve_locally = True
app.css.config.serve_locally = True
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
df = pd.read_csv(filename)
date_list = sorted(df['date'].unique().tolist())

dates = df['date'].unique().tolist()

# Default model predictions
# actives, totals = forecast.pipeline(
#     param, data=df[df['County'] == 'Bexar County'].reset_index(),
#     county='Bexar County')
prediction_df = read_json(prediction_file)
actives, totals = prediction_df['Bexar']
prediction_list = prediction_df.keys().tolist()
# Date list in the slider menu
DATE_MODIFIED = [sorted(dates)[::-1][i] for i in range(len(dates)) if i % 20 == 0][::-1]
print(DATE_MODIFIED, "The date modified")
# The screen layout
app.layout = html.Div(
    id='root',
    children=[
        html.Div
            (
            id='header',
            # style={'fontSize': 16, 'display': 'inline-block', 'width': '70%'},
            children=[
                html.H4(children="COVID-19 predictions in counties in Texas"),
                dcc.Markdown(
                    id="description",
                    # children=" These show the number of active cases registered and observed for coronavirus "
                    #          "for these given day milestones.",
                    children="The rate of COVID-19 infectious spread is dependent on the policies that constrain social interaction and travel. "
                             "In our work, we model the growth of COVID-19 cases in the state of Texas using mobility data provided publicly by Google. "
                             "We have used this model, which we call SIRNet, to study the impact of social distancing policies on "
                             "COVID-19 spread. This dashboard visualizes such mobility data and our daily-updated forecasts at the county level."
                             "\n\n"
                             "Our research was initially presented at the 2020 International Conference on Machine Learning (Machine Learning for Global Health Workshop), "
                             "with the latest findings accepted to the 2020 International Joint Conference on AI "
                             "(Disease Computational Modeling Workshop). If you are interested in using our model or "
                             "digging deeper into our methods, please see below."
                ),
                html.Details(
                    [
                        html.Summary('Read more', style={'cursor': 'pointer', 'font-weight': 'bold'},
                                     id='read-more'),
                        dcc.Markdown(
                            id='read-more-content',
                            children=r'''### **Mobility Data**

Mobility data is taken from publicly available cell phone data from Google. 

### **Forecasts**

The forecasts shown here are given for various reporting rates, i.e., the presumed percentage of cases that have been reported. This acts as a natural confidence interval, showing the range of scenarios that may hold true depending on reporting methodology and completeness. Furthermore, a mobility percentage can be specified in our dashboard with respect to the baseline. This is an assumed mobility rate for the days that SIRNet makes predictions as there is no data for future dates.

### **Using Our Model**
Please use the citation below if you intend to use our predictions or model. You can read the [preprint of the SIRNet on arXiv here](https://arxiv.org/abs/2004.10376). 

You can view and run our codebase for this work (and dashboard) [on GitHub](https://github.com/Nu-AI/Livid-About-COVID). 

You can also contact our lab and see what other work we are doing [at our Nu.AI lab website](https://www.nuailab.com/).

Our latest work, accepted as a paper at the 25th International Joint Conferences on Artificial Intelligence([IJCAI'20](https://dcm-2020.github.io/)) :
```text
@inproceedings{souresSIRNetIJCAI2020,
  title     = {SIRNet: Understanding Social Distancing Measures with Hybrid Neural Network Model for COVID-19 Infectious Spread},
  authors   = {Soures, Nicholas and Chambers, David and Carmichael, Zachariah and Daram, Anurag and Clark, Kal and Shah, Pankil and Shah, Dimpy and Potter, Lloyd and Kudithipudi, Dhireesha},
  booktitle = {Proceedings of the International Joint Conference on Artificial Intelligence, {IJCAI}},
  series    = {Disease Computational Modeling Workshop},
  year      = {2020}
}
```

Earlier progress of SIRNet, presented as a poster at the 37th International Conference on Machine Learning([ICML'20](https://mlforglobalhealth.org/posters-and-spotlights/)):
```text
@inproceedings{souresSIRNetICML2020,
  title     = {SIRNet: Understanding Social Distancing Measures with Hybrid Neural Network Model for COVID-19 Infectious Spread},
  authors   = {Soures, Nicholas and Chambers, David and Carmichael, Zachariah and Daram, Anurag and Clark, Kal and Shah, Pankil and Shah, Dimpy and Potter, Lloyd and Kudithipudi, Dhireesha},
  booktitle = {Proceedings of the International Conference on Machine Learning, {ICML}},
  series    = {Machine Learning for Global Health Workshop},
  year      = {2020},
  url       = {\url{https://mlforglobalhealth.org/posters-and-spotlights/}}
}
```'''
                        ),
                    ],
                ),
            ],
        ),

        html.Div
            (
            id='test_right',
            style={'display': 'inline-block', 'vertical-align': 'top', 'horizontal-align': 'right'},
            children=[
                daq.LEDDisplay(
                    id="operator-led",
                    value=texas_df.cases.tolist()[-1],
                    color="#2cfec1",
                    label={'label': "TOTAL CASES*",
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
            style={'display': 'inline-block', 'vertical-align': 'top'},
            children=[
                daq.LEDDisplay(
                    id="operator-led2",
                    value=texas_df.deaths.tolist()[-1],
                    color="#2cfec1",
                    label={'label': "DEATHS*",
                           'style': {
                               'fontSize': 24,
                               'color': '#2cfec1'
                           }
                           },
                    backgroundColor="#1e2130",
                    size=50,
                ),
                html.P('*For the state of Texas.'),
                html.P("**Predictions last updated on {}".format(str(latest_date)),
                             id='predictions-update',
                             ),
                html.P("*** On mobile the plots look best in landscape mode",
                       id='mobile-note'),
            ],
        ),
        # html.Div
        #     (
        #     id='test_right3',
        #     style={'display': 'inline-block'},
        #     children=[html.P("**Predictions last updated on {}".format(str(latest_date)),
        #                      id='predictions-update2',
        #                      ),
        #               ]
        # ),
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
                            id='heatmap-container',
                            style={'fontSize': 20},
                            children=
                            [
                                html.P
                                    (
                                    "Heatmap of mobility \
                                    in Texas counties on selected date {0}".format(
                                        DATE_MODIFIED[0]),
                                    # children=["init"],
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
                                            tooltip={'placement': 'bottom'},
                                            step=1,
                                        ),
                                    ],
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
                            style={'margin-right':'5%', 'padding-right':'1rem'}
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


@app.callback(
    Output('heatmap-title', 'children'),
    [Input('date_slider', 'value')]
)
def update_text(value):
    return "Heatmap of mobility in Texas counties on selected date {0}".format(DATE_MODIFIED[value])


# The map display callback function
@app.callback(
    Output('county_chloropleth', 'figure'),
    [Input('date_slider', 'value'),
     Input('chart-dropdown', 'value')]
)
def plot_map(selected_date, selected_mob):
    print("The root directory", ROOT_DIR)
    basepath = os.path.join(ROOT_DIR, 'Dashboard')
    new_path = path.abspath(path.join(basepath,'GEOJSONs'))
    path_json = path.abspath(path.join(new_path, str(DATE_MODIFIED[selected_date])))
    with open('{}.geojson'.format(path_json)) as readfile:
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
def set_figure_template(fig_layout, *args):
    for arg in args:
        fig_data = arg
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
    mob_df = mob_df[['date', 'Retail & recreation',
                     'Grocery & pharmacy', 'Parks', 'Transit stations',
                     'Workplace', 'Residential']]
    mob_df = mob_df.set_index('date')
    fig3 = go.Figure()

    fig = go.Figure()

    county = county.split(' ')[0]
    '''
    # actives, totals = forecast.pipeline(param, data=mob_df,
    #                                     county=county)
    '''
    if county in prediction_list:
        actives, totals = prediction_df[county]
        totalpred_df = pd.DataFrame.from_dict(totals['0.1'])
        total_predicted_cases_0_05 = pd.DataFrame.from_dict(totals['0.05'])
        total_predicted_cases_0_3 = pd.DataFrame.from_dict(totals['0.3'])

        active_df = pd.DataFrame.from_dict(actives['0.1'])
        active_predicted_cases_0_05 = pd.DataFrame.from_dict(actives['0.05'])
        active_predicted_cases_0_3 = pd.DataFrame.from_dict(actives['0.3'])
        active_df.date = pd.to_datetime(active_df.date, format='%d-%b %Y')
        totalpred_df.date = pd.to_datetime(totalpred_df.date, format='%d-%b %Y')
        fig3 = cont_error_bar(fig3, totalpred_df['date'],
                              total_predicted_cases_0_3[str(selected_percent)],
                              totalpred_df[str(selected_percent)],
                              total_predicted_cases_0_05[str(selected_percent)],
                              selected_percent)

        fig3_layout = fig3['layout']
        fig3_data = fig3['data']
        fig3.update_layout(
            title='Total predicted cases based on the reporting rate for {} County'.format(county),
            showlegend=False
        )

        fig = cont_error_bar(fig, active_df['date'],
                             active_predicted_cases_0_3[str(selected_percent)],
                             active_df[str(selected_percent)],
                             active_predicted_cases_0_05[str(selected_percent)],
                             selected_percent)
        fig_layout = fig["layout"]
        fig_data = fig["data"]

        fig.update_layout(
            title='Active predicted cases based on the reporting rate for {} County'.format(county),
            showlegend=False
        )
        set_figure_template(fig_layout, fig_data)
        set_figure_template(fig3_layout, fig3_data)


    else:
        fig_layout = fig["layout"]
        fig.update_layout(
            title='Active predicted cases not available for {} County'.format(county),
            showlegend=False
        )
        fig3_layout = fig3["layout"]
        fig3.update_layout(
            title='Total predicted cases not available for {} County'.format(county),
            showlegend=False
        )
        set_figure_template(fig_layout)
        set_figure_template(fig3_layout)

    fig2 = mob_df.iplot(asFigure=True, title="Average Mobility over time")
    fig2_layout = fig2["layout"]
    fig2_data = fig2["data"]

    fig2.update_layout(
        title='Mobility over time in {}'.format(updated_county),
        legend=dict(bgcolor='#1f2630', font=dict(color="#2cfec1"))
    )
    set_figure_template(fig2_layout, fig2_data)

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
    # app.run_server(debug=True, use_reloader=False)
    app.run_server(debug=False, use_reloader=False)
