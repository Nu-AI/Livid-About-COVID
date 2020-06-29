import os
from os.path import join as pjoin
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import datetime as dt

############### Paths ##############################
####################################################

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(ROOT_DIR)

WEIGHTS_DIR = os.path.join(ROOT_DIR, 'model_weights')
if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

RESULTS_DIR = os.path.join(ROOT_DIR, 'Prediction_results')
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

from SIRNet import util, trainer
from scripts import retrieve_data

########### ASSUMPTIONS ##############################
######################################################
country = 'United States'
state = 'Texas'
county = 'Bexar County'
# @formatter:off
reporting_rate = 0.1     # Portion of cases that are actually detected
delay_days = 10          # Days between becoming infected / positive confirmation (due to incubation period / testing latency
start_model = 23         # The day where we begin our fit
mask_modifier = False    #
mask_day = 65            # Day of mask order
incubation_days = 5      # 5 day incubation period [Backer et al]
estimated_r0 = 2.2       # 2.2 R0 estimated in literature
n_epochs = 200           # number of training epochs
lr_step_size = 4000      # learning rate decay step size
mob_pct_cases = [25, 50, 75, 100]
# @formatter:on

MOBILITY_KEYS = ['Retail & recreation', 'Grocery & pharmacy', 'Parks',
                 'Transit stations', 'Workplace', 'Residential']


def load_data():
    ############## Simplified Data ####################
    ###################################################
    paramdict = {
        'country': country,
        'states': [state],
        'counties': [county]
    }
    df = retrieve_data.get_data(paramdict)

    mobility = df[MOBILITY_KEYS]
    cases = df['Cases']
    population = df['Population'][0]  # All values the same

    total_delay = delay_days + start_model
    day0 = df['date'][total_delay]

    # The previous number of cases after model delays
    prev_cases = cases[total_delay - (1 if total_delay > 0 else 0)]

    # offset case data by delay days (treat it as though it was recorded earlier)
    cases = np.array(cases[delay_days:])
    mobility = np.array(mobility[:-delay_days])

    ###################### Formatting Data ######################
    #############################################################
    mobility[-1][-1] = 17.0
    mobility = np.asarray(mobility).astype(np.float32)
    # convert percentages of change to fractions of activity
    mobility[:, :6] = 1.0 + mobility[:, :6] / 100.0

    # Turn the last column into 1-hot social-distancing enforcement
    mobility[:, 5] = 0
    if mask_modifier:
        mobility[mask_day:, 5] = 1.0

    # start with delay
    mobility = mobility[start_model:]
    cases = cases[start_model:]

    return mobility, cases, day0, population, prev_cases


## Cases ##
mobility, cases, day0, population, prev_cases = load_data()

if county.lower().endswith(' county'):
    county_name = county[:-len(' county')]
else:
    county_name = county

# timestamp = dt.datetime.now().strftime('%Y_%m_%d')
timestamp = dt.datetime.now().strftime('%Y_%m_%d_%H_%M')

actives = {}
totals = {}

# TODO: if weights_dir provided...
weights_dir_base = pjoin(WEIGHTS_DIR, timestamp)
if not os.path.exists(weights_dir_base):
    os.mkdir(weights_dir_base)

for reporting_rate in [0.05, 0.1, 0.3]:

    # Initial conditions
    i0 = float(prev_cases) / population / reporting_rate
    e0 = estimated_r0 * i0 / incubation_days

    # Split into input and output data
    X, Y = mobility, cases

    # divide out population of county, reporting rate
    Y = (Y / population) / reporting_rate

    # To Torch on device
    X = torch.from_numpy(X.astype(np.float32))
    Y = torch.from_numpy(Y.astype(np.float32))

    # Add batch dimension
    X = X.reshape(X.shape[0], 1, X.shape[1])  # time x batch x channels
    Y = Y.reshape(Y.shape[0], 1, 1)  # time x batch x channels

    #################### Training #######################
    #####################################################
    # TODO: if weights_dir provided...
    # weights_name = pjoin(weights_dir_base, '{}_report{}_weights.pt'.format(
    #     county_name, reporting_rate))
    weights_name = pjoin(weights_dir_base, '{}_weights.pt'.format(county_name))
    trnr = trainer.Trainer(weights_name)
    model = trnr.build_model(e0, i0)
    trnr.train(model, X, Y, iters=n_epochs, step_size=lr_step_size)

    ################ Forecasting #######################
    ####################################################
    active = {}
    total = {}

    for case in mob_pct_cases:
        xN = torch.ones((1, 6), dtype=torch.float32) * case / 100
        xN[0, 5] = 0
        rX = xN.expand(200, *xN.shape)  # 200 x 1 x 6
        rX = torch.cat((X, rX), dim=0)
        if mask_modifier:
            rX[mask_day:, 0, 5] = 1.0
        sir_state, total_cases = model(rX)
        s = util.to_numpy(sir_state)
        active[case] = s[:, 0] * reporting_rate * population
        total[case] = (s[:, 0] + s[:, 1]) * reporting_rate * population
    actives[reporting_rate] = active
    totals[reporting_rate] = total

    ############## Forecast Dates ####################
    ##################################################
    yy, mm, dd = day0.split('-')
    date0 = dt.datetime(int(yy), int(mm), int(dd))
    days = np.arange(rX.shape[0])
    dates = [date0 + dt.timedelta(days=int(d)) for d in days]

    ############### Reporting #########################
    ###################################################
    print('\n#########################################\n\n')
    for case in mob_pct_cases:
        M = np.max(active[case])
        idx = np.argmax(active[case])
        print('Case: {}%'.format(case))
        print('  Max value: {}'.format(M))
        print('  Day: {}, {}'.format(idx, dates[idx]))

############### Plotting ##########################
###################################################
gt = np.squeeze(cases)

# plot styles & plot letters
cs = {25: 'b-', 50: 'g--', 75: 'y-.', 100: 'r:'}
cl = {25: 'a', 50: 'b', 75: 'c', 100: 'd'}

plt.rcParams.update({'font.size': 22})

# Plot 1. Total Cases (Log)
pidx = gt.shape[0] + 60  # write letter prediction at 60 days in the future
plt.figure(dpi=100, figsize=(16, 8))
for case in total.keys():
    plt.plot(dates, totals[.1][case], cs[case], linewidth=4.0,
             label='{}. {}% Mobility'.format(cl[case], case))
    # plt.fill_between(dates, totals[.05][case], totals[.30][case],
    #                  color=cs[case][0], alpha=.1)
    plt.fill_between(dates, totals[.05][case], totals[.1][case],
                     color=cs[case][0], alpha=.1)
    plt.fill_between(dates, totals[.1][case], totals[.30][case],
                     color=cs[case][0], alpha=.1)
    plt.text(dates[pidx], totals[.1][case][pidx], cl[case])
plt.plot(dates[:Y.shape[0]], gt, 'ks', label='SAMHD Data')

plt.title('Total Case Count')
plt.ylabel('Count')
plt.yscale('log')
util.plt_setup()
plt.savefig(RESULTS_DIR + '/{}_Total_Cases.pdf'.format(timestamp))
plt.show()

# Plots 2 & 3. Active Cases (zoomed out and zoomed in)
for zoom in [True, False]:
    plt.figure(dpi=100, figsize=(16, 8))
    for case in total.keys():
        plt.plot(dates, actives[.1][case], cs[case], linewidth=4.0,
                 label='{}. {}% Mobility'.format(cl[case], case))
        # plt.fill_between(dates, actives[.05][case], actives[.30][case],
        #                  color=cs[case][0], alpha=.1)
        plt.fill_between(dates, actives[.05][case], actives[.1][case],
                         color=cs[case][0], alpha=.1)
        plt.fill_between(dates, actives[.1][case], actives[.30][case],
                         color=cs[case][0], alpha=.1)
        pidx = (gt.shape[0] + 10 if zoom else
                np.argmax(actives[.1][case]))  # write at 10 days or peak
        if case == 50:
            pidx += 5
        if zoom:
            plt.text(dates[pidx], min(actives[.1][case][pidx], 1400), cl[case])
        else:
            plt.text(dates[pidx], actives[.1][case][pidx], cl[case])

    plt.title('Active (Infectious) Case Count')
    plt.ylabel('Count')
    if zoom:
        plt.ylim((0, gt[-1]))
    util.plt_setup()
    plt.savefig(RESULTS_DIR + '/{}_Active_Cases{}.pdf'.format(timestamp, zoom))
    plt.show()
