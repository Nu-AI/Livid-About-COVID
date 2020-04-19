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

import torch
from torch import optim

# root of workspace
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(ROOT_DIR)
# directory of data
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# TODO review naming scheme of directory (also maybe download weights via
#  LFS-like thing instead of commit??)
# directory of model weights
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'model_weights')
if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

# TODO review decision to have this directory (at least naming scheme)
# directory of results
RESULTS_DIR = os.path.join(ROOT_DIR, 'Prediction_results')
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

import SIRNet
from SIRNet import util
from SIRNet import forecast_plotter as fp

## ASSUMPTIONS: Let's put these properties right up front where they belong ###
###############################################################################
# @formatter:off
# reporting_rate = 0.20  # Portion of cases that are actually detected
reporting_rate = 0.60    # Portion of cases that are actually detected
# delay_days = 4         # Days between becoming infected / positive confirmation (due to incubation period / testing latency
delay_days = 10          # Days between becoming infected / positive confirmation (due to incubation period / testing latency
bed_pct = 0.40           # Portion of hospital beds that can be allocated for Covid-19 patients
hosp_rate = 0.20         # Portion of cases that result in hospitalization
# @formatter:on

# TRAIN_MULTIPLE = True
TRAIN_MULTIPLE = False

## TODO LIST
# - What is residential mobility? David thinks it should be ignored, not very
#   helpful
# - Use population-weighted density instead (census tracts)
# - Account for better underreporting (paper from Stanford estimated 50-80x
#   undercount (relative to confirmed) of cases in Santa Clara CA, US
#   [Bendavid, Mulaney, et al. "COVID-19 Antibody Seroprevalence..."]
# - pull in automatically, keep up to date:
#   https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv

# NOTE: if CUDA is slower for you, just make device 'cpu'...
# TODO move to argparse/main
if not torch.cuda.is_available():
    device = torch.device('cpu')  # use CPU
else:
    device = torch.device('cuda')  # use GPU/CUDA

# Download latest data
import urllib.request

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
if TRAIN_MULTIPLE:
    # TODO: bring in hosp. bed for these rows...
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

    # TODO TODO very temporary START
    # bed_ratio = counties[0][3] / counties[0][2]
    bed_ratio = 7793 / 2003554  # Bexar beds / Bexar population
    for c in counties:
        if len(c) == 3:
            c.append(c[-1] * bed_ratio)
    # TODO TODO very temporary END
else:
    # TODO: TEMPORARY START
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

    # TODO TODO very temporary START
    # bed_ratio = counties[0][3] / counties[0][2]
    bed_ratio = 7793 / 2003554  # Bexar beds / Bexar population
    for c in counties:
        if len(c) == 3:
            c.append(c[-1] * bed_ratio)
    # TODO TODO very temporary END

    # TODO: TEMPORARY END

    # # Determine the 5 biggest county case rates in these 5 states:
    # # NY, NJ, CA, MI, PA, TX
    # counties = [
    #     # county, state, population, hospital beds
    #     # ['King', 'Washington', 2.25e6, 8000],
    #     # ['Harris','Texas', 4.7e6, 12000],
    #     # ['New York City', 'New York', 8.0e6, 32000],
    #     # ['Bexar', 'Texas', 1.99e6, 7793],
    #     ['Bexar', 'Texas', 2.00e6, 7793],
    # ]

# Instead of showing all plots, save them to disk instead
SAVE_PLOTS = True


def main(Xs, Ys, names=None):
    global hosp_rate

    if TRAIN_MULTIPLE:
        # TODO: THINGS MOVED OUT OF COUNTY FOR LOOP HERE - ZACH DO NOT COMMIT YET.
        #  DEFINITELY DO NOT PUSH.
        n_samples = len(Xs)
        data_idxs = list(range(n_samples))
        random.shuffle(data_idxs)
        train_frac = 0.5
        split_idx = max(int(n_samples * train_frac), 1)

        Xs_train = [Xs[idx] for idx in data_idxs[:split_idx]]
        Ys_train = [Ys[idx] for idx in data_idxs[:split_idx]]
        names_train = [names[idx] for idx in data_idxs[:split_idx]]

        Xs_test = [Xs[idx] for idx in data_idxs[split_idx:]]
        Ys_test = [Ys[idx] for idx in data_idxs[split_idx:]]
        names_test = [names[idx] for idx in data_idxs[split_idx:]]

        print('{} training samples, {} testing samples'.format(
            len(Xs_train), len(Xs_test)))
    else:
        X, Y = Xs, Ys

    def build_model(e0, i0, b_lstm=False, update_k=False):
        model = torch.nn.Sequential()
        model.add_module('SEIRNet', SIRNet.SEIRNet(e0=e0, i0=i0, b_lstm=b_lstm,
                                                   update_k=update_k))
        return model.to(device=device)

    def train(model, loss, optimizer, x, y, log_loss=True):
        optimizer.zero_grad()

        hx, fx = model.forward(x)

        if log_loss:
            output = loss.forward(torch.log(fx), torch.log(y))
        else:
            output = loss.forward(fx, y)
        output.backward()
        optimizer.step()
        # for name, param in model.named_parameters():
        #    if name == "SEIRNet.i2b.weight":
        #      param.data.clamp_(1e-4)
        return output.data.item()

    ##################### Build and Train Model #################
    #############################################################
    TRY_LSTM = False
    # TRY_LSTM = True
    UPDATE_K = False

    model = build_model(e0, i0, b_lstm=TRY_LSTM, update_k=UPDATE_K)
    loss = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.1)
    # batch_size = Y.shape[0]
    torch.autograd.set_detect_anomaly(True)

    if TRAIN_MULTIPLE:
        weights_name = '{}_weights_multiple.pt'.format(state_name)
    else:
        weights_name = '{}_weights.pt'.format(county_name)
    weights_name = os.path.join(WEIGHTS_DIR, weights_name)

    if not os.path.exists(weights_name):
        iters = 1000
    else:
        model.load_state_dict(torch.load(weights_name))
        iters = 1000
        # iters = 0

    # print([np.isnan(util.to_numpy(x)).any() for x in Xs])

    for i in range(iters):
        if TRAIN_MULTIPLE:
            iterator = zip(Xs_train, Ys_train)
        else:
            iterator = zip([X], [Y])
        for X, Y in iterator:
            batch_size = Y.shape[0]  # TODO
            cost = 0.
            num_batches = math.ceil(len(X) / batch_size)
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                cost += train(model, loss, optimizer, X[start:end],
                              Y[start:end])
            if i % 100 == 0 or True:
                print('Epoch = %d, cost = %s' % (i + 1, cost / num_batches))
                print('The model fit is: ')
                for name, param in model.named_parameters():
                    print(name, param.data)
        # TODO: scheduler may restart learning rate if trying to load from file
        #  Mitigation: store epoch number in filename
        scheduler.step()

    if TRAIN_MULTIPLE:
        def mean_squared_error_samplewise(y_pred, y_true, agg_func=np.mean):
            assert len(y_pred) == len(y_true)
            mses = []
            for yp, yt in zip(y_pred, y_true):
                mses.append(
                    np.mean((util.to_numpy(yp) - util.to_numpy(yt)) ** 2)
                )
            return agg_func(mses)

        def mean_squared_error_elementwise(y_pred, y_true):
            assert len(y_pred) == len(y_true)
            mses = []
            for yp, yt in zip(y_pred, y_true):
                mses.append(
                    (util.to_numpy(yp) - util.to_numpy(yt)) ** 2
                )
            return np.mean(np.concatenate(mses))

        metrics = [
            mean_squared_error_samplewise,
            mean_squared_error_elementwise
        ]

        def evaluate(inputs, outputs):
            preds = [model(xi) for xi in inputs]

            scores = []
            for metric in metrics:
                score = metric(preds, outputs)
                print(metric.__name__, score)
                scores.append(score)
            return scores

        evaluate(Xs_train, Ys_train)
        evaluate(Xs_test, Ys_test)

    torch.save(model.state_dict(), weights_name)

    if TRAIN_MULTIPLE:
        imax = min(3, len(Xs_train))

        # 3 Random samples of train
        X_plot = list(Xs_train[:imax])
        X_plot += list(Xs_test[:imax])

        # 3 Random samples of test
        Y_plot = list(Ys_train[:imax])
        Y_plot += list(Ys_test[:imax])

        plot_names = list(names_train[:imax])
        plot_names += list(names_test[:imax])

        iterator = zip(X_plot, Y_plot)
    else:
        iterator = zip([X], [Y])

    for i, (X, Y) in enumerate(iterator):
        sir_state, total_cases = model(X)
        YY = util.to_numpy(total_cases)

        # Plot the SIR state
        sir_state = util.to_numpy(sir_state)
        util.plot_sir_state(sir_state, show=not SAVE_PLOTS)
        if SAVE_PLOTS:
            plt.savefig(county_name + '_sir_state.pdf')
            plt.close()

        # Plot the total cases
        if TRAIN_MULTIPLE:
            name = plot_names[i]
            plt.title('Cases for {}'.format(name))
        else:
            plt.title('Cases')
        plt.xlabel('Day')
        plt.ylabel('Cases')
        Y = util.to_numpy(Y)  # Torch -> NumPy and squeeze
        pcs = plt.plot(range(Y.shape[0]), Y, 'r',
                       range(Y.shape[0]), YY, 'g')
        plt.legend(pcs, ['Ground Truth', 'Predicted'])
        if SAVE_PLOTS:
            plt.savefig(county_name + '_day_cases_gt_predicted.pdf')
            plt.close()
        else:
            plt.show()

    if TRAIN_MULTIPLE:
        # TODO
        print('TRAIN_MULTIPLE - plots past here not implemented')
        return

    ######## Forecast mobility from 0 to 100 % #################################
    ############################################################################
    for i in range(0, 101, 10):
        p = i / 100.0
        # xN = (torch.ones((1, 6), dtype=torch.float32, device=device) * p +
        #       X[-1, :, :] * (1 - p))
        xN = torch.ones((1, 6), dtype=torch.float32, device=device) * p
        print('xN', xN)
        rX = xN.expand(200, *xN.shape)  # 200 x 1 x 6
        rX = torch.cat((X, rX), dim=0)

        # Give mobility number as percentage, exclude residences
        pct = int(np.mean(util.to_numpy(xN)[:5]) * 100)

        sir_state, total_cases = model(rX)

        # Plot the SIR state
        s = util.to_numpy(sir_state)
        days = range(s.shape[0])
        days = np.asarray(days)
        days_ones = np.ones((len(days),))
        bed40 = beds / population * days_ones * bed_pct * population
        bed70 = beds / population * days_ones * .70 * population
        active = s[:, 0] * reporting_rate * population
        total = (s[:, 0] + s[:, 1]) * reporting_rate * population
        hospitalized = s[:, 0] * float(hosp_rate) * reporting_rate * population
        # recovered * WHO mortality rate
        #   (recovered is actually recovered + deceased)
        total_deaths = s[:, 1] * 0.034 * reporting_rate * population

        fig, axs = plt.subplots(2)
        axs[0].plot(days, active, 'b', days, total, 'r')
        axs[0].legend(['Active', 'Total'], loc='upper left')
        axs[0].set(ylabel='Cases')
        axs[0].set_title('Quarantine/Normal: {:02d}/{:02d} | '
                         'Avg. Mobility: {:02d}%'.format(100 - i, i, pct))
        axs[0].set_xticklabels([])

        axs[1].plot(days, hospitalized, 'b', days, total_deaths, 'k', days,
                    bed40, 'c', bed70, 'r')
        axs[1].legend(['Hospitalized', 'Total Deaths', '40% of Hospital Beds',
                       '70% of Hospital Beds'], loc='upper left')
        axs[1].set(xlabel='Months', ylabel='Cases')
        tick_font = {'fontname': 'Arial', 'size': '11', 'weight': 'bold'}
        axs[1].set_xticks(range(0, 211, 30))
        axs[1].set_xticklabels(
            ['March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct'])

        # plt.savefig('output/mobility{:03d}.png'.format(i))
        if SAVE_PLOTS:
            plt.savefig(county_name + '_mobility{:03d}.pdf'.format(i))
            plt.close()
        else:
            plt.show()

    ######## Forecast 200 more days at 20% quarantine mobility #############
    ############################################################################
    # xN = X[-1, :, :]  # current mobility...
    # xN = torch.tensor([[.1, .1, .1, .1, .1, 3]],
    #                   dtype=torch.float32, device=device)
    xN = torch.ones((1, 6), dtype=torch.float32, device=device) * .20
    qX = xN.expand(200, *xN.shape)  # 200 x 1 x 6
    qX = torch.cat((X, qX), dim=0)

    sir_state, total_cases = model(qX)

    # Plot the SIR state
    sir_state1 = util.to_numpy(sir_state)
    util.plot_sir_state(sir_state1, title='SIR_state (lockdown mobility)',
                        show=not SAVE_PLOTS)
    if SAVE_PLOTS:
        plt.savefig(county_name + '_sir_state1_lockdown_mob.pdf')
        plt.close()

    ######## Forecast 200 more days returning to normal mobility ###############
    ############################################################################
    xN = torch.ones((1, 6), dtype=torch.float32, device=device)
    rX = xN.expand(200, *xN.shape)  # 200 x 1 x 6
    rX = torch.cat((X, rX), dim=0)

    sir_state, total_cases = model(rX)

    # Plot the SIR state
    sir_state2 = util.to_numpy(sir_state)
    util.plot_sir_state(sir_state2, title='SIR_state (full mobility)',
                        show=not SAVE_PLOTS)
    if SAVE_PLOTS:
        plt.savefig(county_name + '_sir_state2_full_mob.pdf')
        plt.close()

    ######## Forecast 200 more days at 50% normal mobility ##########
    ############################################################################
    xN = torch.ones((1, 6), dtype=torch.float32, device=device) * .50
    rX = xN.expand(200, *xN.shape)  # 200 x 1 x 6
    rX = torch.cat((X, rX), dim=0)

    sir_state, total_cases = model(rX)

    # Plot the SIR state
    sir_state3 = util.to_numpy(sir_state)
    util.plot_sir_state(sir_state3, title='SIR_state (split mobility)',
                        show=not SAVE_PLOTS)
    if SAVE_PLOTS:
        plt.savefig(county_name + '_sir_state3_split_mob.pdf')
        plt.close()

    ######## Forecast 200 more days at 20%-return to normal mobility ###########
    ############################################################################
    xN = (torch.ones((1, 6), dtype=torch.float32, device=device) * .20 +
          X[-1, :, :] * .80)
    rX = xN.expand(200, *xN.shape)  # 200 x 1 x 6
    rX = torch.cat((X, rX), dim=0)

    sir_state, total_cases = model(rX)

    # Plot the SIR state
    sir_state4 = util.to_numpy(sir_state)
    util.plot_sir_state(sir_state4, title='SIR_state (split mobility)',
                        show=not SAVE_PLOTS)
    if SAVE_PLOTS:
        plt.savefig(county_name + '_sir_state4_split_mob.pdf')
        plt.close()

    # Plot the hospitalization forecast
    hosp_rate = float(hosp_rate)
    days = range(sir_state1.shape[0])
    bedp = beds / population * np.ones((len(days),)) * bed_pct
    print(sir_state1.shape, hosp_rate, reporting_rate)
    hosp_qrt = sir_state1[:, 0] * hosp_rate * reporting_rate
    hosp_nom = sir_state2[:, 0] * hosp_rate * reporting_rate
    hosp_hlf = sir_state3[:, 0] * hosp_rate * reporting_rate
    hosp_qtr = sir_state4[:, 0] * hosp_rate * reporting_rate

    pcs = plt.plot(days, hosp_qrt, 'g',
                   days, hosp_nom, 'r',
                   days, hosp_hlf, 'm',
                   days, hosp_qtr, 'c',
                   days, bedp, 'b')
    plt.xlabel('Day')
    plt.ylabel('Hospitalized')
    plt.legend(pcs, ['Current Mobility', 'Return to Normal Mobility',
                     '50% Return to Normal', '20% Return to Normal',
                     'Available Beds'])
    if SAVE_PLOTS:
        plt.savefig(county_name + '_day_hosp_mobility_beds.pdf')
        plt.close()
    else:
        plt.show()

    # Save the data
    # cases = ['Current Mobility', 'Return to Normal Mobility',
    #          '50% Return to Normal', '20% Return to Normal']
    cases = ['20% Mobility', 'Normal Mobility', '50% Mobility', '75% Mobility']
    for i, s in enumerate([sir_state1, sir_state2, sir_state3, sir_state4]):
        data = {
            'Days': np.array(days), 'Active Cases (latent)': s[:, 0],
            'Active Cases (observed)': s[:, 0] * reporting_rate,
            'Total Cases (latent)': s[:, 0] + s[:, 1],
            'Total Cases (observed)': (s[:, 0] + s[:, 1]) * reporting_rate,
            'Hospitalized': s[:, 0] * hosp_rate * reporting_rate,
            # recovered * WHO mortality rate
            #   (recovered is actually recovered + deceased)
            'Total Deaths': s[:, 1] * 0.034 * reporting_rate
        }
        # Alternative way to save
        string = str(cases[i]) + str(county_name)
        # str = 'Average Case ' + str(cases[i]) + str(county_name) + '.npy'
        # np.save(str, data)
        np.save(os.path.join(RESULTS_DIR, 'Average Case {}.npy'.format(string)),
                data)

    # legend_list = ['Current Mobility', '20% Mobility', '50% Mobility',
    #                'Normal Mobility']
    legend_list = ['25% Mobility', 'Normal Mobility', '50% Mobility',
                   '75% Mobility']
    # data_list, day_list = fp.get_arrays(
    #     fp.get_scenario_dict(fp.scenario_list, county_name),
    #     fp.scenario_list, fp.population
    # )
    data_list, day_list = fp.get_arrays(
        fp.get_scenario_dict(fp.scenario_list, county_name),
        fp.scenario_list, population
    )
    # fp.plot_data(data_list, day_list, legend_list, 0)
    # fp.plot_data(data_list, day_list, legend_list, 1)
    # fp.plot_data(data_list, day_list, legend_list, 0,
    #              Y * fp.population * reporting_rate, show=not SAVE_PLOTS)
    fp.plot_data(data_list, day_list, legend_list, 0,
                 Y * population * reporting_rate, show=not SAVE_PLOTS)
    if SAVE_PLOTS:
        plt.savefig(county_name + '_forecast_plotted.pdf')
        plt.close()


## Iterate through counties ##
##############################
Xs, Ys, names = [], [], []
for county_data in counties:
    county_name, state_name, population, beds = county_data

    ############### Loading data #########################
    ######################################################

    # Load US County cases data
    cases = OrderedDict()
    with open('us-counties.csv', 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            if row[1] == county_name and row[2] == state_name:
                cases[row[0]] = row[4]  # cases by date

    # Shift case data assuming it lags actual by 10 days
    shift_cases = OrderedDict()
    for i in range(len(cases.keys()) - delay_days):
        k = list(cases.keys())[i]
        k10 = list(cases.keys())[i + delay_days]
        shift_cases[k] = cases[k10]
    cases = shift_cases

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
    # TODO
    if any(not mobilities[k] for k in mobilities):
        print('WARN: skipping {}, {} because it does not have all mobility '
              'categories'.format(county_name, state_name))
        continue
    else:
        print('INFO: Beginning {}, {}'.format(county_name, state_name))

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
    # with open('US_County_AgeGrp_2018.csv', 'rb') as f:
    #     txt = f.read().decode('iso-8859-1').encode('utf-8')
    #     open("temp.csv", "w").write(str(txt))
    #     csv_reader = csv.reader(open('temp.csv','r'), delimiter=',')
    #     header = next(csv_reader)
    #     for row in csv_reader:
    #         print(row[2])
    #         if (row[2] == '{} County'.format(county_name) and
    #               row[1] == state_name):
    #             p0_19 = row[4] + row[5] + row[6] + row[7]
    #             p20_44 = row[8] + row[9] + row[10] + row[11] + row[12]
    #             p45_64 = row[13] + row[14] + row[15] + row[16]
    #             p65_74 = row[17] + row[18]
    #             p75_84 = row[19] + row[20]
    #             p85_p  = row[21]
    #             p = [p0_19, p20_44, p45_64, p65_74, p75_84, p85_p]
    #             print (p, "******************")
    #     rates = []
    #     with open('covid_hosp_rate_by_age.csv', 'r',
    #               encoding='mac_roman') as f:
    #         csv_reader = csv.reader(f, delimiter=',')
    #         header = next(csv_reader)
    #         for row in csv_reader:
    #             rates.append(np.array(row))
    #     if not p:
    #         print ("p is empty")
    #     else:
    #         hosp_rate = (np.sum(np.array(p).astype(float) *
    #                             np.mean(np.array(rates).astype(float),axis=0)))

    # hosp_rate = 0.20

    ###################### Formatting Data ######################
    #############################################################
    # Data is 6 columns of mobility, 1 column of case number
    data = np.asarray(data).astype(np.float32)
    # TODO: integrate this 10+ patients part correctly...
    # data = data[5:, :]  # Skip 5 days until we have 10+ patients
    # data[:,5] = 0.0 # residential factored out
    data[:, :6] = (
            1.0 + data[:, :6] / 100.0
    )  # convert percentages of change to fractions of activity
    print('data.shape', data.shape)
    print('Initial mobility', data[0, :])

    # TODO: I made the initialization for i0 come from the previous day, so the
    #  predicted and actual match better
    METHOD_e0i0 = 2
    if METHOD_e0i0 == 1:
        # e0 = i0 = float(data[4, 6]) / population
        e0 = i0 = float(data[4, 6]) / population / reporting_rate

    # Split into input and output data
    X, Y = data[:, :6], data[:, 6]
    # X is now retail&rec, grocery&pharm, parks, transit_stations, workplace,
    #   residential
    # Y is the total number of cases
    if not TRAIN_MULTIPLE:
        plt.plot(Y)
        plt.xlabel('Day')
        plt.ylabel('Total cases')
        if SAVE_PLOTS:
            plt.savefig(county_name + '_day_total_cases.pdf')
            plt.close()
        else:
            plt.show()

        mag = [np.linalg.norm(X[t, :5]) for t in range(X.shape[0])]
        plt.plot(mag)
        plt.xlabel('Day')
        plt.ylabel('Activity')
        if SAVE_PLOTS:
            plt.savefig(county_name + '_day_activity.pdf')
            plt.close()
        else:
            plt.show()

    # divide out population of county
    Y = Y / population
    # multiply by suspected under-reporting rate
    Y = Y / reporting_rate

    if METHOD_e0i0 == 2:
        # i0 and e0 (assume incubation of `delay_days`)
        e0 = Y[0]
        # e0 minus the gradient between of Y (point at 0 and point at
        # delay_days) times delay_days = 2 * e0 - Y[delay_days], clip at 0
        i0 = max(2 * e0 - Y[delay_days], 0)
        print('e0={} | i0={}'.format(e0, i0))
    else:
        raise NotImplementedError('no METHOD_e0i0 for "{}"'.format(METHOD_e0i0))
    # To Torch on device
    X = torch.from_numpy(X).to(device=device)
    Y = torch.from_numpy(Y).to(device=device)

    # Add batch dimension
    X = X.reshape(X.shape[0], 1, X.shape[1])  # time x batch x channels
    Y = Y.reshape(Y.shape[0], 1, 1)  # time x batch x channels

    # \ COMBINE TO ONE DATA SET
    if TRAIN_MULTIPLE:
        Xs.append(X)
        Ys.append(Y)
        names.append(county_name + state_name)

    if not TRAIN_MULTIPLE:
        # TODO: temporary try/except...
        try:
            main(X, Y)
        except Exception as e:
            print('EXCEPTION WHILE RUNNING {}'.format(county_name))
            print(e)
            print('Continuing...\n')

if TRAIN_MULTIPLE:
    main(Xs, Ys, names)
