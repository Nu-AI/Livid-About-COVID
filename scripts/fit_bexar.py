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
from torch import optim
import datetime as dt

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
from SIRNet import Bexar_plotter as fpp

## ASSUMPTIONS: Let's put these properties right up front where they belong ###
###############################################################################

reporting_rate = 0.60    # Portion of cases that are actually detected
delay_days = 10          # Days between becoming infected / positive confirmation (due to incubation period / testing latency
bed_pct = 0.40           # Portion of hospital beds that can be allocated for Covid-19 patients
hosp_rate = 0.20         # Portion of cases that result in hospitalization
start_model = 5          # The day where we begin our fit


############### Gathering Data #####################
####################################################

# TODO: move these CSV downloads to the correct place (also check timestamp with
#  what is available online?)
if not os.path.exists('us-counties.csv'):
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

# Hospital beds/1k people
# https://www.kff.org/other/state-indicator/beds-by-ownership/?activeTab=graph&currentTimeframe=0&startTimeframe=19&selectedRows=%7B%22states%22:%7B%22texas%22:%7B%7D%7D%7D&sortModel=%7B%22colId%22:%22Location%22,%22sort%22:%22asc%22%7D

df = pd.read_excel(
    'https://www2.census.gov/programs-surveys/popest/tables/2010-2019/'
    'counties/totals/co-est2019-annres-48.xlsx',
    skiprows=(0, 1, 2, 4))
df.dropna(inplace=True)
df.rename(columns={df.columns[0]: 'County'}, inplace=True)
df.loc[:, 'County'] = df['County'].apply(
    lambda name: name.replace('.', '')[:name.index(' County') - 1]
)
df.loc[:, 'State'] = 'Texas'
counties = df[['County', 'State', 2019]].values.tolist()

# bed_ratio = counties[0][3] / counties[0][2]
bed_ratio = 7793 / 2003554  # Bexar beds / Bexar population
for c in counties:
    if len(c) == 3:
        c.append(c[-1] * bed_ratio)


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

# Shift case data assuming it lags actual by 10 days
shift_cases = OrderedDict()
day0 = list(cases.keys())[0]
for i in range(len(cases.keys()) - delay_days):
    k = list(cases.keys())[i]
    k10 = list(cases.keys())[i + delay_days]
    shift_cases[k] = cases[k10]
cases = shift_cases
# Our cases are driven by mobility data from 10 days before
# Our first case date

# Load Activity Data
state_path = os.path.join(DATA_DIR, 'Mobility data',
                          '{}_mobility.csv'.format(state_name))
mkeys = ['Retail & recreation', 'Grocery & pharmacy', 'Parks',
         'Transit stations', 'Workplace', 'Residential']

if os.path.exists(state_path):
    row1_filt = state_name
    row2_filt = lambda s: s.replace(' County', '') == county_name
else:
    print('No county-level mobility available.  Reverting to state')
    state_path = os.path.join(DATA_DIR, 'Mobility data', 'US_mobility.csv')
    row1_filt = 'US'
    row2_filt = lambda s: s == state_name

mobilities = {}
for category in mkeys:
    mobilities[category] = OrderedDict()
    with open(state_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        header = next(csv_reader)
        for row in csv_reader:
            if (row[1] == row1_filt and row2_filt(row[2]) and
                    row[3] == category):
                dates = eval(row[6])
                vals = eval(row[7])
                for i, date in enumerate(dates):
                    mobilities[category][date] = vals[i]

# Rearrange activity data
mobility = OrderedDict()
dates = list(mobilities['Retail & recreation'].keys())
for i, date in enumerate(dates):
    # NOTE: some dates are missing in the data, so we fall back to the
    #  previous day's data here
    def k_by_date_or_fallback(k):
        j = i  # start with current date
        while j >= 0:
            date_or_fallback = dates[j]
            if mobilities[k].get(date_or_fallback) is not None:
                return mobilities[k][date_or_fallback]
            j -= 1
        raise ValueError('Data did not have a valid fallback date.')

    mobility[date] = [k_by_date_or_fallback(k) for k in mkeys]

# Common Case Data + Activity Data Dates
data = []
common_dates = []
for k in cases.keys():
    if k not in mobility.keys():
        continue
    data.append(mobility[k] + [cases[k]])  # total cases

# Estimate hospitalization rate
p = []
df = pd.read_csv(
    os.path.join(DATA_DIR, 'US_County_AgeGrp_2018.csv'),
    encoding='cp1252'
)
a = df.loc[(df['STNAME'] == state_name) &
           (df['CTYNAME'] == county_name + ' County')]
key_list = []
for keys in a.keys():
    key_list.append(keys)
print(key_list)

# TODO: comment these (or make more efficient via pandas)
p0_19 = 0
p20_44 = 0
p45_64 = 0
p65_74 = 0
p75_84 = 0
p85_p = 0
for i in range(len(key_list)):
    if 4 < i < 8:
        p0_19 += int(a[key_list[i]])
    elif 7 < i < 13:
        p20_44 += int(a[key_list[i]])
    elif 12 < i < 17:
        p45_64 += int(a[key_list[i]])
    elif 16 < i < 19:
        p65_74 += int(a[key_list[i]])
    elif 18 < i < 21:
        p75_84 += int(a[key_list[i]])
    else:
        p85_p = int(a[key_list[21]])
p = [p0_19, p20_44, p45_64, p65_74, p75_84, p85_p]
print('the p value is', p)
rates = []

hr_filename = os.path.join(DATA_DIR, 'covid_hosp_rate_by_age.csv')
with open(hr_filename, 'r', encoding='mac_roman') as f:
    csv_reader = csv.reader(f, delimiter=',')
    header = next(csv_reader)
    for row in csv_reader:
        rates.append(np.asarray(row))

temp = np.mean(np.asarray(rates).astype(float))
hosp_rate = np.sum(np.asarray(p).astype(float) * temp, axis=0)
hosp_rate /= a[key_list[3]]


###################### Formatting Data ######################
#############################################################
# Data is 6 columns of mobility, 1 column of case number
data = np.asarray(data).astype(np.float32)

data[:, :6] = (1.0 + data[:, :6] / 100.0)  # convert percentages of change to fractions of activity
print('data.shape', data.shape)
print('Initial mobility', data[0, :])

# Initial conditions
i0 = float(data[start_model-1, 6]) / population / reporting_rate
e0 = 2.2*i0
data = data[start_model:] # sart on day 5

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
trainer.train(model, X, Y, 1000)

# Plot R vs. mobility
W = np.squeeze(model.state_dict()['SEIRNet.i2b.weight'].numpy())
k = np.squeeze(model.state_dict()['SEIRNet.k'].numpy())
p = np.squeeze(model.state_dict()['SEIRNet.p'].numpy())
q = np.squeeze(model.state_dict()['SEIRNet.q'].numpy())
ms = np.linspace(0, 1.2, 20)

Re = [ q*np.sum(W*m)**p/k for m in ms]
plt.plot(ms,Re)
plt.xlabel('Average mobility')
plt.ylabel('Reproduction number')
plt.title('R0 vs. Mobility')
plt.grid()
plt.show()

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
cl[100] = 'd'

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
plt.show()

# Plots 2 & 3. Active Cases (zoomed out and zoomed in)
for zoom in [True, False]:
  plt.figure(dpi=100, figsize=(16,8))
  for case in total.keys():
    plt.plot(dates, active[case], cs[case],linewidth=4.0, label='{}. {}% Mobility'.format(cl[case], case))
    pidx = np.argmax(active[case])  # write letter prediction at 60 days in the future
    plt.text(dates[pidx],active[case][pidx],cl[case])

  plt.title('Active (Infectious) Case Count')
  plt.ylabel('Count')
  plt.grid(True)
  plt.legend()
  if zoom: plt.ylim((0, gt[-1]))
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
  plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
  plt.gcf().autofmt_xdate()
  plt.show()




