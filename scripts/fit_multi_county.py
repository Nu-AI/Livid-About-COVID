import os
import csv
import sys
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
import SIRNet
from SIRNet import util
from SIRNet import forecast_plotter as fp

## ASSUMPTIONS: Let's put these properties right up front where they belong ###
###############################################################################
# @formatter:off
reporting_rate = 0.20  # Portion of cases that are actually detected
delay_days = 4         # Days between becoming infected / positive confirmation (due to incubation period / testing latency
bed_pct = 0.40         # Portion of hospital beds that can be allocated for Covid-19 patients
hosp_rate = 0.20       # Portion of cases that result in hospitalization
# @formatter:on

## TODO LIST
# - What is residential mobility? David thinks it should be ignored, not very
#   helpful
# - X

# NOTE: if CUDA is slower for you, just make device 'cpu'...
# TODO move to argpargse/main
if not torch.cuda.is_available():
    device = torch.device('cpu')  # use CPU
else:
    device = torch.device('cuda')  # use GPU/CUDA

# Download latest data
import urllib.request

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

# # Anurag's addition (TODO: correct spot for this?)
# # Determine the 5 biggest county case rates in these 5 states:
# def get_county_state_dict(path):
#     top5_df = pd.read_excel(path)
#
#     state_list = top5_df['State'].unique().tolist()
#
#     county_dict = {}
#     for state in state_list:
#         county_dict[state] = top5_df[top5_df['State'] == state][
#             'Counties'].tolist()
#     return pd.DataFrame.from_dict(county_dict)
#
#
# def get_population_dict(path):
#     df = pd.read_excel(path, skiprows=2, skipfooter=5)
#     new_df = df[['Geographic Area', 'Unnamed: 12']].reset_index().iloc[
#              1:].reset_index()
#     Area_list = new_df['Geographic Area']
#     area_list = [i.split(',')[0].split(' ')[0].replace('.', '') for i in
#                  Area_list]
#     new_df['Geographic Area'] = area_list
#     return new_df
#
#
# state_name = 'Texas'
# county_df = get_county_state_dict(path='Top5counties.xlsx')
#
# pop_df = get_population_dict(
#     "https://www2.census.gov/programs-surveys/popest/tables/2010-2019"
#     "/counties/totals/co-est2019-annres-48.xlsx "
# )
#
#
# # NY, NJ, CA, MI, PA, TX
# def get_county_pop_list(county_df, pop_df):
#     county_list = county_df[state_name].tolist()
#     new_pop_df = pop_df[pop_df['Geographic Area'].isin(county_list)]
#     pop_list = new_pop_df['Unnamed: 12'].tolist()
#     return county_list, pop_list
#
#
# county_list, pop_list = get_county_pop_list(county_df, pop_df)
# counties = []
# hospital_beds = [19000, 14000, 5000, 5000, 7893]
# for i in range(len(county_list)):
#     counties.append([county_list[i], state_name, pop_list[i],
#                      hospital_beds[i]])

# Determine the 5 biggest county case rates in these 5 states:
# NY, NJ, CA, MI, PA, TX
counties = [
    # county, state, population, hospital beds
    # ['New York City', 'New York', 8.0e6, 32000],
    ['Bexar', 'Texas', 1.99e6, 7793],
]
## Iterate through counties ##
##############################
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
    if os.path.exists(state_path):
        mkeys = ['Retail & recreation', 'Grocery & pharmacy', 'Parks',
                 'Transit stations', 'Workplace', 'Residential']
        mobilities = {}
        for category in mkeys:
            mobilities[category] = OrderedDict()
            with open(os.path.join(DATA_DIR, 'Mobility data',
                                   '{}_mobility.csv'.format(state_name)),
                      'r') as f:
                csv_reader = csv.reader(f, delimiter=',')
                header = next(csv_reader)
                for row in csv_reader:
                    if (row[1] == state_name and
                            row[2].replace(' County', '') == county_name and
                            row[3] == category):
                        dates = eval(row[6])
                        vals = eval(row[7])
                        for i, date in enumerate(dates):
                            mobilities[category][date] = vals[i]
    else:
        print('No county-level mobility available.  Reverting to state')
        mkeys = ['Retail & recreation', 'Grocery & pharmacy', 'Parks',
                 'Transit stations', 'Workplace', 'Residential']
        mobilities = {}
        for category in mkeys:
            mobilities[category] = OrderedDict()
            with open(os.path.join(DATA_DIR, 'Mobility data',
                                   'US_mobility.csv'), 'r') as f:
                csv_reader = csv.reader(f, delimiter=',')
                header = next(csv_reader)
                for row in csv_reader:
                    if (row[1] == 'US' and row[2] == state_name and
                            row[3] == category):
                        dates = eval(row[6])
                        vals = eval(row[7])
                        for i, date in enumerate(dates):
                            mobilities[category][date] = vals[i]

    # Rearrange activity data
    mobility = OrderedDict()
    for date in mobilities['Retail & recreation']:
        mobility[date] = [mobilities[k][date] for k in mkeys]

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
        encoding="cp1252"
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
    data = data[5:, :]  # Skip 5 days until we have 10+ patients

    data[:, :6] = (
            1.0 + data[:, :6] / 100.0
    )  # convert percentages of change to fractions of activity
    print('data.shape', data.shape)

    # Split into input and output data
    X, Y = data[:, :6], data[:, 6]
    # X is now retail&rec, grocery&pharm, parks, transit_stations, workplace,
    #   residential
    # Y is the total number of cases
    plt.plot(Y)
    plt.show()

    # divide out population of county
    Y = Y / population
    # multiply by suspected under-reporting rate
    Y = Y / reporting_rate
    # i0 and e0 (assume incubation of `delay_days`)
    e0 = Y[0]
    # e0 minus the gradient between of Y (point at 0 and point at delay_days)
    # times delay_days = 2 * e0 - Y[delay_days], clip at 0
    i0 = max(2 * e0 - Y[delay_days], 0)
    print('e0={} | i0={}'.format(e0, i0))
    # To Torch on device
    X = torch.from_numpy(X).to(device=device)
    Y = torch.from_numpy(Y).to(device=device)

    # Add batch dimension
    X = X.reshape(X.shape[0], 1, X.shape[1])  # time x batch x channels
    Y = Y.reshape(Y.shape[0], 1, 1)  # time x batch x channels


    def build_model(e0, i0, b_lstm=False, update_k=False):
        model = torch.nn.Sequential()
        model.add_module('SEIRNet', SIRNet.SEIRNet(e0=e0, i0=i0, b_lstm=b_lstm,
                                                   update_k=update_k))
        return model.to(device=device)


    def train(model, loss, optimizer, x, y, log_transform=True):
        optimizer.zero_grad()

        hx, fx = model.forward(x)

        if log_transform:
            output = loss.forward(torch.log(fx), torch.log(y))
        else:
            output = loss.forward(fx, y)
        output.backward()
        optimizer.step()
        # for name, param in model.named_parameters():
        #    if name != "name.h2o.weight":
        #        param.data.clamp_(1e-4)
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
    batch_size = Y.shape[0]
    torch.autograd.set_detect_anomaly(True)
    weights_name = '{}_weights.pt'.format(county_name)

    if not os.path.exists(weights_name):
        iters = 1000
    else:
        model.load_state_dict(torch.load(weights_name))
        iters = 1000

    for i in range(iters):
        cost = 0.
        num_batches = len(X) // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer, X[start:end], Y[start:end])
        if i % 100 == 0:
            print("Epoch = %d, cost = %s" % (i + 1, cost / num_batches))
            print('The model fit is: ')
            for name, param in model.named_parameters():
                print(name, param.data)
        scheduler.step()

    torch.save(model.state_dict(), weights_name)

    sir_state, total_cases = model(X)
    YY = util.to_numpy(total_cases)

    # Plot the SIR state
    sir_state = util.to_numpy(sir_state)
    util.plot_sir_state(sir_state)

    # Plot the total cases
    plt.title('Cases')
    plt.xlabel('Day')
    plt.ylabel('Cases')
    pcs = plt.plot(range(Y.shape[0]), util.to_numpy(Y), 'r',
                   range(Y.shape[0]), YY, 'g')
    plt.legend(pcs, ['Ground Truth', 'Predicted'])
    plt.show()

    ######## Forecast mobility from 0 to 100 % #################################
    ############################################################################
    for i in range(0, 101, 10):
        p = i / 100.0
        # xN = (torch.ones((1, 6), dtype=torch.float32) * p +
        #       X[-1, :, :] * (1 - p))
        xN = torch.ones((1, 6), dtype=torch.float32) * p
        print('xN', xN)
        rX = xN[:, None, ...].expand(200, 1, 6).to(device=device)  # 200 x 1 x 6
        rX = torch.cat((X, rX), axis=0)

        # Give mobility number as percentage, exclude residences
        pct = int(np.mean(util.to_numpy(xN)[:5]) * 100)

        sir_state, total_cases = model(rX)
        YY = util.to_numpy(total_cases)

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
        print('HTF? ', s[-1, 1], 0.034, reporting_rate, population)

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
        plt.show()
        plt.close()

    ######## Forecast 200 more days at current quarantine mobility #############
    ############################################################################
    xN = X[-1, :, :]  # lockdown
    xN[0, :] = torch.Tensor(
        np.array([.1, .1, .1, .1, .1, 3]).astype(np.float32))
    qX = xN[:, None, ...].expand(200, 1, 6).to(device=device)  # 200 x 1 x 6
    qX = torch.cat((X, qX), axis=0)

    sir_state, total_cases = model(qX)
    YY = util.to_numpy(total_cases)

    # Plot the SIR state
    sir_state1 = util.to_numpy(sir_state)
    util.plot_sir_state(sir_state1, title='SIR_state (lockdown mobility)')

    ######## Forecast 120 more days returning to normal mobility ###############
    ############################################################################
    xN = torch.ones((1, 6), dtype=torch.float32)
    rX = xN[:, None, ...].expand(200, 1, 6).to(device=device)  # 200 x 1 x 6
    rX = torch.cat((X, rX), axis=0)

    sir_state, total_cases = model(rX)
    YY = util.to_numpy(total_cases)

    # Plot the SIR state
    sir_state2 = util.to_numpy(sir_state)
    util.plot_sir_state(sir_state2, title='SIR_state (full mobility)')

    ######## Forecast 120 more days at half-return to normal mobility ##########
    ############################################################################
    xN = (torch.ones((1, 6), dtype=torch.float32).to(device=device) +
          X[-1, :, :]) / 2
    rX = xN[:, None, ...].expand(200, 1, 6).to(device=device)  # 200 x 1 x 6
    rX = torch.cat((X, rX), axis=0)


    sir_state, total_cases = model(rX)
    YY = util.to_numpy(total_cases)

    # Plot the SIR state
    sir_state3 = util.to_numpy(sir_state)
    util.plot_sir_state(sir_state3, title='SIR_state (split mobility)')

    ######## Forecast 120 more days at 25%-return to normal mobility ###########
    ############################################################################
    xN = (torch.ones((1, 6), dtype=torch.float32).to(device=device) * .20 +
          X[-1, :, :] * .80)
    rX = xN[:, None, ...].expand(200, 1, 6)  # 200 x 1 x 6
    rX = torch.cat((X, rX), axis=0)

    sir_state, total_cases = model(rX)
    YY = util.to_numpy(total_cases)

    # Plot the SIR state
    sir_state4 = util.to_numpy(sir_state)
    util.plot_sir_state(sir_state4, title='SIR_state (split mobility)')

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
    plt.show()

    # Save the data
    cases = ['Current Mobility', 'Return to Normal Mobility',
             '50% Return to Normal', '20% Return to Normal']
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
        # str = 'Average Case ' + str(cases[i]) + str(county_name) + '.npy'
        # np.save(str, data)
        np.save('Average Case {}.npy'.format(cases[i]), data)

legend_list = ['Current Mobility', '20% Mobility', '50% Mobility',
               'Normal Mobility']
data_list, day_list = fp.get_arrays(fp.get_scenario_dict(fp.scenario_list),
                                    fp.scenario_list, fp.population)
fp.plot_data(data_list, day_list, legend_list, 0)
fp.plot_data(data_list, day_list, legend_list, 1)
