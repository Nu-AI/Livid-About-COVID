import os
import csv
import sys
import math
import random
from collections import OrderedDict
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import datetime as dt
import retrieve_data 

############### Paths ##############################
####################################################

# root of workspace
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(ROOT_DIR)
# directory of data
DATA_DIR = os.path.join(ROOT_DIR, 'data')

WEIGHTS_DIR = os.path.join(ROOT_DIR, 'model_weights')
if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

# directory of results
RESULTS_DIR = os.path.join(ROOT_DIR, 'Prediction_results')
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

from SIRNet import util, trainer

## ASSUMPTIONS: Let's put these properties right up front where they belong ###
###############################################################################
reporting_rate = 0.1    # Portion of cases that are actually detected
delay_days = 10          # Days between becoming infected / positive confirmation (due to incubation period / testing latency
bed_pct = 0.40           # Portion of hospital beds that can be allocated for Covid-19 patients
start_model = 20         # The day where we begin our fit

# Retrieve Data
paramdict = {}
paramdict['country'] = 'United States'
paramdict['states'] = ['Texas']
paramdict['counties'] = ['Bexar County']
df = retrieve_data.get_data(paramdict)


############### Gathering Data #####################
####################################################
urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/nytimes/'
    'covid-19-data/master/us-counties.csv',
    'us-counties.csv'
)
urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/nytimes/'
    'covid-19-data/master/us-states.csv',
    'us-states.csv'
)


###########   Get case data         ########################
############################################################
county_name, state_name, population, beds = 'Bexar', 'Texas', 2003554, 7793

# Load US County cases data
cases = OrderedDict()
with open('us-counties.csv', 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        if row[1] == county_name and row[2] == state_name:
            cases[row[0]] = row[4]  # cases by date
k = list(cases.keys())
print ('Cases data goes from {} to {}'.format(k[0],k[-1]))

# Shift case data assuming it lags actual by 10 days
shift_cases = OrderedDict()
day0 = list(cases.keys())[0]
for i in range(len(cases.keys()) - delay_days):
    k = list(cases.keys())[i]
    k10 = list(cases.keys())[i + delay_days]
    shift_cases[k] = cases[k10]
cases = shift_cases
# Our cases are driven by mobility data from 10 days before
k = list(cases.keys())
print ('Cases data goes from {} to {}'.format(k[0],k[-1])) # These are the dates we will use to look up mobility data

# Format retrieved mobility data
mkeys = ['Retail & recreation', 'Grocery & pharmacy', 'Parks', 'Transit stations', 'Workplace', 'Residential']
mobility = OrderedDict()
for i, key in enumerate(df['date']):
  mobility[key] = [df[mkey][i] for mkey in mkeys]
k = list(mobility.keys())
print ('Mobility data goes from {} to {}'.format(k[0],k[-1]))

# Common Case Data + Activity Data Dates
data = []
common_dates = []
for k in cases.keys():
    if k not in mobility.keys():
        continue
    data.append(mobility[k] + [cases[k]])  # total cases
    common_dates.append(k)
print('The common dates are', common_dates)
day0 = common_dates[0]  # common day (will need to delay later)


###################### Formatting Data ######################
#############################################################
# Data is 6 columns of mobility, 1 column of case number
data = np.asarray(data).astype(np.float32)
print('The data is')
for row in data:
  print(row)

data[:, :6] = (1.0 + data[:, :6] / 100.0)  # convert percentages of change to fractions of activity
print('data.shape', data.shape)
print('Initial mobility', data[0, :])

# Initial conditions
i0 = float(data[start_model-1, 6]) / population / reporting_rate
e0 = 2.2*i0/5.0
data = data[start_model:] # start on day 5

# Split into input and output data
X, Y = data[:, :6], data[:, 6]
# X is now retail&rec, grocery&pharm, parks, transit_stations, workplace,residential
# Y is the total number of cases

plt.plot(Y)
plt.xlabel('Day')
plt.ylabel('Total cases')
plt.show()

mag = [np.mean(X[t, :5]) for t in range(X.shape[0])]
plt.plot(mag)
plt.xlabel('Day')
plt.ylabel('Activity')
plt.show()

# divide out population of county
Y = Y / population
# multiply by suspected under-reporting rate
Y = Y / reporting_rate
print ('population, reporting rate', population, reporting_rate)

# To Torch on device
X = torch.from_numpy(X.astype(np.float32))
Y = torch.from_numpy(Y.astype(np.float32))

# Add batch dimension
X = X.reshape(X.shape[0], 1, X.shape[1])  # time x batch x channels
Y = Y.reshape(Y.shape[0], 1, 1)  # time x batch x channels

#################### Training #######################
#####################################################
weights_name = WEIGHTS_DIR + '/{}_weights.pt'.format(county_name)
trainer = trainer.Trainer(weights_name)
model = trainer.build_model(e0,i0)
trainer.train(model, X, Y, 300)

################ Forecasting #######################
####################################################
active = {}
total = {}
cases = [25, 50, 75, 100]
for case in cases:
    xN = torch.ones((1, 6), dtype=torch.float32) * case/100
    rX = xN.expand(200, *xN.shape)  # 200 x 1 x 6
    rX = torch.cat((X, rX), dim=0)

    sir_state, total_cases = model(rX)

    # Plot the SIR state
    s = util.to_numpy(sir_state)
    #util.plot_sir_state(s, title='SEIR {}%'.format(case))

    active[case] = s[:,0] * reporting_rate * population
    total[case] = (s[:,0] + s[:,1]) * reporting_rate * population

############## Forecast Dates ####################
##################################################
yy, mm, dd = day0.split('-')
date0 = dt.datetime(int(yy),int(mm),int(dd))
days = np.arange(rX.shape[0])
dates = [date0 + dt.timedelta(days=int(d + delay_days + start_model)) for d in days]

############### Reporting #########################
###################################################
print('\n#########################################\n\n')
timestamp = dt.datetime.now().strftime('%Y_%m_%d')
for case in cases:
    M = np.max(active[case])
    idx = np.argmax(active[case])
    print ('Case: {}%'.format(case))
    print ('  Max value: {}'.format(M))
    print ('  Day: {}, {}'.format(idx, dates[idx]))

############### Plotting ##########################
###################################################
gt = np.squeeze(Y.numpy()) * reporting_rate * population

# plot styles
cs = {}
cs[25] = 'b-'
cs[50] = 'g--'
cs[75] = 'y-.'
cs[100] = 'r:'

# plot letters
cl = {}
cl[25] = 'a'
cl[50] = 'b'
cl[75] = 'c'
cl[100]= 'd'

# Default text size
plt.rcParams.update({'font.size': 22})

# Plot 1. Total Cases (Log)
pidx = gt.shape[0] + 60  # write letter prediction at 60 days in the future
plt.figure(dpi=100, figsize=(16,8))
for case in total.keys():
  plt.plot(dates, total[case], cs[case], linewidth=4.0, label='{}. {}% Mobility'.format(cl[case], case))
  plt.text(dates[pidx],total[case][pidx],cl[case])
plt.plot(dates[:Y.shape[0]], gt, 'ks', label='SAHMD Data')

plt.title('Total Case Count')
plt.ylabel('Count')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.gcf().autofmt_xdate()
plt.savefig('{}_Total_Cases.pdf'.format(timestamp))
plt.show()

# Plots 2 & 3. Active Cases (zoomed out and zoomed in)
for zoom in [True, False]:
  plt.figure(dpi=100, figsize=(16,8))
  for case in total.keys():
    plt.plot(dates, active[case], cs[case],linewidth=4.0, label='{}. {}% Mobility'.format(cl[case], case))
    pidx = gt.shape[0]+10 if zoom else np.argmax(active[case])  # write at 10 days or peak
    plt.text(dates[pidx],active[case][pidx],cl[case])

  plt.title('Active (Infectious) Case Count')
  plt.ylabel('Count')
  plt.grid(True)
  plt.legend()
  if zoom: plt.ylim((0, gt[-1]))
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
  plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
  plt.gcf().autofmt_xdate()
  plt.savefig('{}_Active_Cases{}.pdf'.format(timestamp, zoom))
  plt.show()


