import os
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
# @formatter:off
reporting_rate = 0.1     # Portion of cases that are actually detected
delay_days = 10          # Days between becoming infected / positive confirmation (due to incubation period / testing latency
start_model = 23         # The day where we begin our fit
mask_modifier = False    #
mask_day = 65            # Day of mask order
# @formatter:on

## Cases ##
actives = {}
totals = {}
for reporting_rate in [0.05, 0.1, 0.3]:

    ############## Simplified Data ####################
    ###################################################
    paramdict = {}
    paramdict['country'] = 'United States'
    paramdict['states'] = ['Texas']
    paramdict['counties'] = ['Bexar County']
    df = retrieve_data.get_data(paramdict)

    mobility = df[['Retail & recreation', 'Grocery & pharmacy', 'Parks',
                   'Transit stations', 'Workplace', 'Residential']]
    cases = df['Cases']
    day0 = df['date'][0]
    population = df['Population'][0]

    # offset case data by delay days (treat it as though it was recorded earlier)
    cases = np.array(cases[delay_days:])
    mobility = np.array(mobility[:-delay_days])
    county_name = 'Bexar'

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

    # Initial conditions
    i0 = float(cases[start_model - 1]) / population / reporting_rate
    e0 = 2.2 * i0 / 5.0
    mobility = mobility[start_model:]  # start with delay
    cases = cases[start_model:]  # delay days

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
    weights_name = WEIGHTS_DIR + '/{}_weights.pt'.format(county_name)
    trnr = trainer.Trainer(weights_name)
    model = trnr.build_model(e0, i0)
    trnr.train(model, X, Y, 200)

    ################ Forecasting #######################
    ####################################################
    active = {}
    total = {}
    cases = [25, 50, 75, 100]
    for case in cases:
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
    dates = [date0 + dt.timedelta(days=int(d + delay_days + start_model)) for d
             in days]

    ############### Reporting #########################
    ###################################################
    print('\n#########################################\n\n')
    timestamp = dt.datetime.now().strftime('%Y_%m_%d')
    for case in cases:
        M = np.max(active[case])
        idx = np.argmax(active[case])
        print('Case: {}%'.format(case))
        print('  Max value: {}'.format(M))
        print('  Day: {}, {}'.format(idx, dates[idx]))

############### Plotting ##########################
###################################################
gt = np.squeeze(Y.numpy()) * reporting_rate * population

# plot styles & plot letters
cs = {25: 'b-', 50: 'g--', 75: 'y-.', 100: 'r:'}
cl = {25: 'a', 50: 'b', 75: 'c', 100: 'd'}

# Default text size
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
