import sys
import datetime as dt
import os
from os.path import join as pjoin

import numpy as np
import matplotlib.pyplot as plt

############### Paths ##############################
####################################################
ROOT_DIR = pjoin(os.path.dirname(__file__), '..')
sys.path.append(ROOT_DIR)

WEIGHTS_DIR = pjoin(ROOT_DIR, 'model_weights')
if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

RESULTS_DIR = pjoin(ROOT_DIR, 'prediction_results')
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

from SIRNet import util, trainer  # noqa
from SIRNet.data_collection import retrieve_data  # noqa

MOBILITY_KEYS = ['Retail & recreation', 'Grocery & pharmacy', 'Parks',
                 'Transit stations', 'Workplace', 'Residential']

timestamp = dt.datetime.now().strftime('%Y_%m_%d_%H_%M')


def load_data(params):
    """
    Loads the data used in training and forecasting according to the
    configuration specified in `params`.

    :param params: object holding global configuration values

    :return: mobility data
    :return: cases data
    :return: day0 - date string of first day of data
    :return: population - the population for the region
    :return: prev_cases - the number of cases on the day preceding day0
    """
    ############## Simplified Data ####################
    paramdict = {
        'country': params.country,
        'states': [params.state],
        'counties': [params.county]
    }
    df = retrieve_data.conflate_data(paramdict)

    mobility = df[MOBILITY_KEYS]
    cases = df['Cases']
    population = df['Population'][0]  # All values the same

    total_delay = params.delay_days + params.start_model
    day0 = df['date'][total_delay]

    # The previous number of cases after model delays
    prev_cases = cases[total_delay - (1 if total_delay > 0 else 0)]

    # offset case data by delay days (treat it as though it was recorded
    # earlier)
    cases = np.array(cases[params.delay_days:])
    mobility = np.array(mobility[:-params.delay_days])

    ###################### Formatting Data ######################
    # mobility[-1][-1] = 17.0  # TODO what is this
    mobility = np.asarray(mobility).astype(np.float32)
    # convert percentages of change to fractions of activity
    mobility[:, :6] = 1.0 + mobility[:, :6] / 100.0

    # Turn the last column into 1-hot social-distancing enforcement
    mobility[:, 5] = 0
    if params.mask_modifier:
        mobility[params.mask_day:, 5] = 1.0

    # start with delay
    mobility = mobility[params.start_model:]
    cases = cases[params.start_model:]

    return mobility, cases, day0, population, prev_cases


def model_and_fit(weights_name, X, Y, scale_factor, prev_cases, params,
                  summary_writer=None):
    """
    Builds a model with weights at a provided file and fits the model with
    provided data.

    :param weights_name: path to the weights file
    :param X: input examples (time steps x batch size x features)
    :param Y: output labels (time steps x batch size x classes)
    :param scale_factor: factor with which to scale cases
    :param prev_cases: the number of cases preceding the first day in the
        forecast (used in I_0, the initial infection rate)
    :param params: object holding global configuration values
    :param summary_writer: optional Torch SummaryWriter object

    :return: SIRNet PyTorch model
    """
    # Initial conditions
    i0 = float(prev_cases) / scale_factor
    e0 = params.estimated_r0 * i0 / params.incubation_days

    trnr = trainer.Trainer(weights_name, summary_writer=summary_writer)
    model = trnr.build_model(e0, i0)
    trnr.train(model, X, Y,
               iters=params.n_epochs, step_size=params.lr_step_size)

    return model


def forecast(X, model, dates, scale_factor, params):
    """
    Forecasts future values according to configuration set in `params`.

    :param X: the input data that the model forecasts upon
    :param model: the PyTorch SIRNet model
    :param dates: the dates of the forecast
    :param scale_factor: factor with which to scale cases
    :param params: object holding global configuration values

    :return: active and total case forecasts for each mobility scenario
    """
    ################ Forecasting #######################
    print('Begin forecasting...')
    active = {}
    total = {}

    for case in params.mobility_cases:
        xN = torch.ones((1, 6), dtype=torch.float32) * case / 100
        xN[0, 5] = 0
        rX = xN.expand(params.forecast_days, *xN.shape)  # days x 1 x 6
        rX = torch.cat((X, rX), dim=0)
        if params.mask_modifier:
            rX[params.mask_day:, 0, 5] = 1.0
        sir_state, total_cases = model(rX)
        s = util.to_numpy(sir_state)
        active[case] = s[:, 0] * scale_factor
        total[case] = (s[:, 0] + s[:, 1]) * scale_factor

        ############### Reporting #########################
        M = np.max(active[case])
        idx = np.argmax(active[case])
        print('Case: {}%'.format(case))
        print('  Max value: {}'.format(M))
        print('  Day: {}, {}'.format(idx, dates[idx]))

    return active, total


def plot(cases, actives, totals, dates, params):
    """
    Plot forecasts.

    :param cases: ground truth case data
    :param actives: active cases (dict mapping reporting rates to forecast
        dicts)
    :param totals: total cases (dict mapping reporting rates to forecast dicts)
    :param dates: dates for each forecast value
    :param params: object holding global configuration values
    """
    gt = np.squeeze(cases)

    # plot styles & plot letters - note this supports 4 mobility cases max
    cs = dict(zip(params.mobility_cases, ['b-', 'g--', 'y-.', 'r:']))
    cl = dict(zip(params.mobility_cases, 'abcd'))

    plt.rcParams.update({'font.size': 22})

    rep_rates = sorted(params.reporting_rates)
    rep_rate_mid = rep_rates[len(rep_rates) // 2]

    # Plot 1. Total Cases (Log)
    pidx = gt.shape[0] + 60  # write letter prediction at 60 days in the future
    plt.figure(dpi=100, figsize=(16, 8))
    for case in params.mobility_cases:
        plt.plot(dates, totals[rep_rate_mid][case], cs[case], linewidth=4.0,
                 label='{}. {}% Mobility'.format(cl[case], case))
        for rep_rate_i in range(len(rep_rates) - 1):
            plt.fill_between(dates,
                             totals[rep_rates[rep_rate_i]][case],
                             totals[rep_rates[rep_rate_i + 1]][case],
                             color=cs[case][0], alpha=.1)
        plt.text(dates[pidx],
                 totals[rep_rate_mid][case][pidx],
                 cl[case])
    plt.plot(dates[:gt.shape[0]], gt, 'ks', label='SAMHD Data')

    plt.title('Total Case Count')
    plt.ylabel('Count')
    plt.yscale('log')
    util.plt_setup()
    plt.savefig(pjoin(RESULTS_DIR, '{}_Total_Cases.pdf'.format(timestamp)))
    plt.show()

    # Plots 2 & 3. Active Cases (zoomed out and zoomed in)
    for zoom in [True, False]:
        plt.figure(dpi=100, figsize=(16, 8))
        for case in params.mobility_cases:
            plt.plot(dates, actives[.1][case], cs[case], linewidth=4.0,
                     label='{}. {}% Mobility'.format(cl[case], case))
            plt.fill_between(dates,
                             actives[.05][case],
                             actives[.1][case],
                             color=cs[case][0], alpha=.1)
            plt.fill_between(dates,
                             actives[.1][case],
                             actives[.3][case],
                             color=cs[case][0], alpha=.1)
            pidx = (gt.shape[0] + 10 if zoom else
                    np.argmax(actives[.1][case]))  # write at 10 days or peak
            if case == 50:
                pidx += 5
            if zoom:
                plt.text(dates[pidx],
                         min(actives[.1][case][pidx], 1400),
                         cl[case])
            else:
                plt.text(dates[pidx],
                         actives[.1][case][pidx],
                         cl[case])

        plt.title('Active (Infectious) Case Count')
        plt.ylabel('Count')
        if zoom:
            plt.ylim((0, gt[-1]))
        util.plt_setup()
        plt.savefig(pjoin(RESULTS_DIR,
                          '{}_Active_Cases{}.pdf'.format(timestamp, zoom)))
        plt.show()

    print('Plots saved to "{}"'.format(RESULTS_DIR))


def pipeline(params):
    """
    Pipeline for loading COVID-19-related data, building and fitting an instance
    of the PyTorch SIRNet model, forecasting for future dates, and plotting the
    results. Weights and figures are saved in the process.

    :param params: object holding global configuration values
    """
    print('Loading data for {}, {}, {}...'.format(
        params.county, params.state, params.country))

    mobility, cases, day0, population, prev_cases = load_data(params)

    if params.county.lower().endswith(' county'):
        county_name = params.county[:-len(' county')]
    else:
        county_name = params.county

    weights_dir_base = (params.weights_dir if params.weights_dir else
                        pjoin(WEIGHTS_DIR, timestamp))
    if not os.path.exists(weights_dir_base):
        os.mkdir(weights_dir_base)

    if params.tensorboard:
        from torch.utils import tensorboard

    actives = {}
    totals = {}

    # Dates used in forecasting
    yy, mm, dd = day0.split('-')
    date0 = dt.datetime(int(yy), int(mm), int(dd))
    days = np.arange(mobility.shape[0] + params.forecast_days)
    dates = [date0 + dt.timedelta(days=int(d)) for d in days]

    for reporting_rate in params.reporting_rates:
        print('\n' + '=' * 80)
        print('Begin pipeline with reporting rate {}'.format(reporting_rate))
        print('=' * 80 + '\n')

        # Split into input and output data
        X, Y = mobility, cases
        # divide out population of county, reporting rate
        scale_factor = population * reporting_rate
        Y = Y / scale_factor

        # To Torch on device
        X = torch.from_numpy(X.astype(np.float32))
        Y = torch.from_numpy(Y.astype(np.float32))

        # Add batch dimension
        X = X.reshape(X.shape[0], 1, X.shape[1])  # time x batch x channels
        Y = Y.reshape(Y.shape[0], 1, 1)  # time x batch x channels

        #################### Training #######################
        # NOTE: this one is broken
        # weights_name = pjoin(weights_dir_base, '{}_report{}_weights.pt'.format(
        #     county_name, reporting_rate))  # TODO this does not fit properly...

        # NOTE: this one is working but reuses weights for all reporting rates..
        weights_name = pjoin(weights_dir_base, '{}_weights.pt'.format(
            county_name))

        if params.tensorboard:
            writer = tensorboard.SummaryWriter(
                log_dir=pjoin('run_logs', timestamp, '{}_report{}'.format(
                    county_name, reporting_rate))
            )
        else:
            writer = None

        model = model_and_fit(weights_name, X, Y, scale_factor, prev_cases,
                              params, summary_writer=writer)

        if params.tensorboard:
            writer.close()
        print('Done training.')

        ################ Forecasting #######################
        active, total = forecast(X, model, dates, scale_factor, params)
        actives[reporting_rate] = active
        totals[reporting_rate] = total

    ############### Plotting ##########################
    print('Begin plotting...')
    plot(cases, actives, totals, dates, params)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(  # noqa
        description='Unified interface to SIRNet training, evaluation, and '
                    'forecasting.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--weights-dir', default=None,
                        help='Optional directory to load old weight or store '
                             'newly trained ones.')
    parser.add_argument('--country', default='United States',
                        help='The country to look for state and county in data '
                             'loading')
    parser.add_argument('--state', default='Texas',
                        help='The state to look for county in data loading')
    parser.add_argument('--county', default='Bexar County',
                        help='The county used in data loading')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Store logs to that can be visualized in '
                             'tensorboard (this needs to be installed '
                             'beforehand). Run tensorboard with e.g. '
                             '--samples_per_plugin images=1000')
    parser.add_argument('--forecast-days', default=200, type=int,
                        help='Number of days to forecast')
    parser.add_argument('--reporting-rates', default=[0.05, 0.1, 0.3],
                        type=float, nargs='+',
                        help='Portion of cases that are actually detected. '
                             'Multiple space-separated values can be passed in '
                             'here.')
    parser.add_argument('--mobility-cases', default=[25, 50, 75, 100],
                        type=float, nargs='+',
                        help='Percentage of mobility assumed in forecasts. '
                             'Multiple space-separated values can be passed in '
                             'here. Note that 4 is the maximum number of cases'
                             'supported.')
    parser.add_argument('--n-epochs', default=200, type=int,
                        help='Number of training epochs')
    parser.add_argument('--lr-step-size', default=4000, type=int,
                        help='Learning rate decay step size')
    parser.add_argument('--delay-days', default=10, type=int,
                        help='Days between becoming infected / positive '
                             'confirmation (due to incubation period / testing '
                             'latency')
    parser.add_argument('--start-model', default=23, type=int,
                        help='The day where we begin our fit (after delay '
                             'days)')
    parser.add_argument('--incubation-days', default=5, type=int,
                        help='Incubation period, default from [Backer et al]')
    parser.add_argument('--estimated-r0', default=2.2, type=float,
                        help='R0 estimated in literature')
    parser.add_argument('--mask-modifier', action='store_true',
                        help='Run mobility scenarios considering mask-wearing')
    parser.add_argument('--mask-day', default=65, type=int,
                        help='Day of mask order')
    # TODO: future integration
    # if disable_cuda or not torch.cuda.is_available():
    #     device = torch.device('cpu')  # use CPU
    # else:
    #     device = torch.device('cuda')  # use GPU/CUDA
    # parser.add_argument('--disable-cuda', action='store_true',
    #                     help='Disables use of CUDA, forcing to execute on '
    #                          'CPU even if a CUDA-capable GPU is available.')

    args = parser.parse_args()

    # Delayed imports for CLI speed
    import torch

    pipeline(args)
else:
    import torch
