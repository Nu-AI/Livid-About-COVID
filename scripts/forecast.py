#!/usr/bin/env python
import sys
import datetime as dt
from copy import deepcopy
import os
from os.path import join as pjoin

# === start paths ===
ROOT_DIR = pjoin(os.path.dirname(__file__), '..')
sys.path.append(ROOT_DIR)

WEIGHTS_DIR = pjoin(ROOT_DIR, 'model_weights')
if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

RESULTS_DIR = pjoin(ROOT_DIR, 'prediction_results')
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
# === end paths ===
from SIRNet.data_collection import data_utils

MOBILITY_KEYS = ['Retail & recreation', 'Grocery & pharmacy', 'Parks',
                 'Transit stations', 'Workplace', 'Residential']

timestamp = dt.datetime.now().strftime('%Y_%m_%d_%H_%M')


def load_data(params):
    """
    Loads the data used in training and forecasting according to the
    configuration specified in `params`.

    :param params: object holding global configuration values

    :return: df: dataframe of the collected data for the required case
    """
    paramdict = {
        'country': params.country,
        'states': None if params.state is None else [params.state],
        'counties': None if params.county is None else [params.county]
    }

    df = retrieve_data.conflate_data(paramdict)
    # TODO: df is empty if data isn't available for input...ideally should have
    #  message saying, hey looks like you typed X but maybe you meant Y? Like
    #  Russian Federation exists but Russia does not
    return df


def process_data(params, df):
    """
    Loads the data used in training and forecasting according to the
    configuration specified in `params`.

    :param params: object holding global configuration values
    :param df: dataframe of the required county

    :return: mobility data
    :return: cases data
    :return: day0 - date string of first day of data
    :return: population - the population for the region
    :return: prev_cases - the number of cases on the day preceding day0
    """
    # Rid NaNs
    print (df.Population[0],df.shape, "the total poulation in the df")

    print (df.Population.unique(), df.shape, "the new total poulation in the df")

    # df.reset_index(inplace=True)

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

    orig_len = len(mobility)
    mobility = np.array(mobility[:-params.delay_days])
    mobility = data_utils.filter_mobility_data(mobility)
    mobility = np.asarray(mobility).astype(np.float32)
    not_nan_idx = ~np.isnan(mobility).any(axis=1)
    mobility = mobility[not_nan_idx]
    cases = cases[not_nan_idx]

    if orig_len > len(mobility):
        print('WARNING: data contained NaNs (%d/%d, %.2f%%) removed. You may '
              'experience issues when fitting the model.' %
              (orig_len - len(mobility), orig_len,
               (orig_len - len(mobility)) / orig_len * 100))
    # convert percentages of change to fractions of activity
    mobility[:, :6] = 1.0 + mobility[:, :6] / 100.0

    # Turn the last column into 1-hot social-distancing enforcement
    mobility[:, 5] = 0  # rid residential mobility...TODO...
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
    model = trnr.build_model(e0, i0, b_model=params.b_model)
    if params.train or not os.path.exists(weights_name):
        print('Training on', params.county or params.state or params.country)
        trnr.train(model, X, Y,
                   iters=params.n_epochs,
                   learning_rate=params.learning_rate,
                   step_size=params.lr_step_size)
        print('Done training.')

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

        # Reporting
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


class _AttrDict(object):
    def __init__(self, d=None):
        if d:
            self.update(d)

    def update(self, d):
        self.__dict__.update(d)
        return self


def pipeline(params=None, **kwargs):
    """
    Pipeline for loading COVID-19-related data, building and fitting an instance
    of the PyTorch SIRNet model, forecasting for future dates, and plotting the
    results. Weights and figures are saved in the process.

    :param params: object holding global configuration values
    :param kwargs: keyword arguments that override values in params

    :return: forecast actives and totals
    """
    default_params = deepcopy(DEFAULTS)
    if params:
        # Combine two sources of parameters, skipping `params` attributes
        # starting with '_'
        params = default_params.update(
            dict(filter(lambda kv: not kv[0].startswith('_'),  # noqa
                        vars(params).items()))
        )
    else:
        params = default_params
    params.update(kwargs)

    # validate some values
    if params.cv_split:
        assert 0 < params.cv_split < 1, (
            'cv-split must be in range (0,1), but received '
            '{}'.format(params.cv_split))

    data_action = 'Loading'
    dump_config = True
    if params.data is None:
        df = load_data(params)
    elif isinstance(params.data, str):
        df = pd.read_csv(params.data)
    else:
        data_action = 'Using given'
        df = params.data
        dump_config = False

    print('{} data for {}, {}, {}...'.format(
        data_action, params.county, params.state, params.country))

    mobility, cases, day0, population, prev_cases = process_data(params, df)

    county_name = params.county
    if county_name:
        if county_name.lower().endswith(' county'):
            county_name = county_name[:-len(' county')]
    else:
        # TODO: naming
        county_name = params.county or params.state or params.country

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

        # Maybe CV prep
        X_train, Y_train = X, Y
        split_idx = None
        if params.cv_split:
            # TODO: will not work as expected for multi-region here
            split_idx = max(int((1 - params.cv_split) * X_train.shape[0]), 1)
            assert split_idx < X.shape[0], 'Too few samples for split'
            X_train, Y_train = X_train[:split_idx], Y_train[:split_idx]
            print('Training on {} samples, testing on {} '
                  'samples.'.format(split_idx, len(X) - split_idx))

        # Training
        weights_name = pjoin(weights_dir_base, '{}_report{}_weights.pt'.format(
            county_name, reporting_rate))

        writer = tensorboard.SummaryWriter(
            log_dir=pjoin('run_logs', timestamp, '{}_report{}'.format(
                county_name, reporting_rate))
        ) if params.tensorboard else None

        model = model_and_fit(weights_name, X_train, Y_train, scale_factor,
                              prev_cases, params, summary_writer=writer)

        if params.tensorboard:
            writer.close()

        # Forecasting
        active, total = forecast(X, model, dates, scale_factor, params)
        if split_idx:
            cases_true = cases[split_idx:len(X)]
            print('Total Population: {}'.format(population))
            print()
            print('These first ones will all be the same, using mobility data '
                  '(no mobility cases...)')  # TODO TODO
            for case in params.mobility_cases:
                cases_pred = total[case][split_idx:len(X)]
                mse = metrics.mean_squared_error_samplewise(
                    y_pred=cases_pred, y_true=cases_true)
                rmse = metrics.root_mean_squared_error_samplewise(
                    y_pred=cases_pred, y_true=cases_true)
                mape = metrics.mean_absolute_percentage_error_samplewise(
                    y_pred=cases_pred, y_true=cases_true)
                print('Held-out test set, mobility case {}%'.format(case))
                print('  MSE  {}'.format(mse))
                print('  RMSE {}'.format(rmse))
                print('  MAPE {}'.format(mape))

            _, total_x = forecast(X_train, model, dates, scale_factor,
                                  params)
            for case in params.mobility_cases:
                cases_pred = total_x[case][split_idx:len(X)]
                mse = metrics.mean_squared_error_samplewise(
                    y_pred=cases_pred, y_true=cases_true)
                rmse = metrics.root_mean_squared_error_samplewise(
                    y_pred=cases_pred, y_true=cases_true)
                mape = metrics.mean_absolute_percentage_error_samplewise(
                    y_pred=cases_pred, y_true=cases_true)
                print('Held-out test set, mobility case {}%'.format(case))
                print('  MSE  {}'.format(mse))
                print('  RMSE {}'.format(rmse))
                print('  MAPE {}'.format(mape))
        actives[reporting_rate] = active
        totals[reporting_rate] = total

    # Additional keys for the dashboard
    for key in totals.keys():
        totals[key]['date'] = dates
        actives[key]['date'] = dates

    # Dump parameters to saved model directory
    if dump_config:
        with open(pjoin(weights_dir_base, 'config_' + timestamp + '.json'),
                  'w') as f:
            json.dump(vars(params), f, indent=2)
    else:
        print('Not dumping config to JSON.')

    # Plot
    if not params.no_plot:
        print('Begin plotting...')
        plot(cases, actives, totals, dates, params)
    return actives, totals


DEFAULTS = _AttrDict()
DEFAULTS.update(dict(
    country='United States',
    state='Texas',
    county='Bexar County',
    forecast_days=200,
    reporting_rates=[0.05, 0.1, 0.3],
    mobility_cases=[25, 50, 75, 100],
    mask_modifier=False,
    mask_day=65,
    weights_dir=None,
    train=False,
    b_model='linear',
    n_epochs=200,
    learning_rate=1e-2,
    lr_step_size=4000,
    delay_days=10,
    start_model=23,
    incubation_days=5,
    estimated_r0=2.2,
    cv_split=None,
    tensorboard=False,
    no_plot=False,
    data=None,
))
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(  # noqa
        description='Unified interface to SIRNet training, evaluation, and '
                    'forecasting.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    g_region = parser.add_argument_group('Region Selection')
    g_region.add_argument(
        '--country', default=DEFAULTS.country, nargs='+',
        help='The country to look for state and county in data loading. '
             'Multiple space-separated values can be passed in here.')
    g_region.add_argument(
        '--state', default=DEFAULTS.state,
        help='The state to look for county in data loading')
    g_region.add_argument(
        '--county', default=DEFAULTS.county,
        help='The county used in data loading')

    g_scenario = parser.add_argument_group('Scenario Options')
    g_scenario.add_argument(
        '--forecast-days', default=DEFAULTS.forecast_days, type=int,
        help='Number of days to forecast')
    g_scenario.add_argument(
        '--reporting-rates', default=DEFAULTS.reporting_rates,
        type=float, nargs='+',
        help='Portion of cases that are actually detected. Multiple '
             'space-separated values can be passed in here.')
    g_scenario.add_argument(
        '--mobility-cases', default=DEFAULTS.mobility_cases,
        type=float, nargs='+',
        help='Percentage of mobility assumed in forecasts. Multiple '
             'space-separated values can be passed in here. Note that 4 is the '
             'maximum number of cases supported.')
    g_scenario.add_argument(
        '--mask-modifier', action='store_true',
        help='Run mobility scenarios considering mask-wearing')
    g_scenario.add_argument(
        '--mask-day', default=DEFAULTS.mask_day, type=int,
        help='Day of mask order')

    g_model = parser.add_argument_group('Model Options')
    g_model.add_argument(
        '--weights-dir', default=DEFAULTS.weights_dir,
        help='Optional directory to load old weight or store newly trained '
             'ones.')
    g_model.add_argument(
        '--train', action='store_true',
        help='Whether to train the model. If `weights-dir` does not exist, '
             'then this option is ignored. Use to continue training from '
             'previously saved weights.')
    g_model.add_argument(
        '--b-model', default=DEFAULTS.b_model,
        help='The type of model for modeling beta as a function of mobility '
             'data.')
    g_model.add_argument(
        '--n-epochs', default=DEFAULTS.n_epochs, type=int,
        help='Number of training epochs')
    g_model.add_argument(
        '--learning-rate', '--lr', default=DEFAULTS.learning_rate, type=float,
        help='Learning rate')
    g_model.add_argument(
        '--lr-step-size', default=DEFAULTS.lr_step_size, type=float,
        help='Learning rate decay step size')
    g_model.add_argument(
        '--delay-days', default=DEFAULTS.delay_days, type=int,
        help='Days between becoming infected / positive confirmation (due to '
             'incubation period/testing latency')
    g_model.add_argument(
        '--start-model', default=DEFAULTS.start_model, type=int,
        help='The day where we begin our fit (after delay days)')
    g_model.add_argument(
        '--incubation-days', default=DEFAULTS.incubation_days, type=int,
        help='Incubation period, default from [Backer et al]')
    g_model.add_argument(
        '--estimated-r0', default=DEFAULTS.estimated_r0, type=float,
        help='R0 estimated in literature')

    g_misc = parser.add_argument_group('Misc. Options')
    g_misc.add_argument(
        '--cv-split', type=float, default=DEFAULTS.cv_split,
        help='Hold out this percentage of data to test the fit on, e.g., 0.25 '
             'uses the last 25%% of days to evaluate the fit'
    )
    g_misc.add_argument(
        '--tensorboard', action='store_true',
        help='Store logs to that can be visualized in tensorboard (this needs '
             'to be installed beforehand). Run tensorboard with e.g. '
             '--samples_per_plugin images=1000. You may also want to disable '
             '30-second updates as this resets images slider positions.')
    g_misc.add_argument(
        '--no-plot', action='store_true',
        help='Do not display and save the prediction plots.')
    g_misc.add_argument(
        '--data', default=DEFAULTS.data,
        help='Alternative data path rather than automatically pulling data.')
    # TODO: for future integration
    # parser.add_argument(
    #   '--disable-cuda', action='store_true',
    #    help='Disables use of CUDA, forcing to execute on CPU even if a '
    #         'CUDA-capable GPU is available.')
    # if disable_cuda or not torch.cuda.is_available():
    #     device = torch.device('cpu')  # use CPU
    # else:
    #     device = torch.device('cuda')  # use GPU/CUDA

    # Parse provided arguments
    args = parser.parse_args()

    if args.state.lower() == 'none':
        args.state = None
    if args.county.lower() == 'none':
        args.county = None

# Delayed imports for CLI speed (used regardless of __main__)
import json  # noqa

import numpy as np  # noqa
import pandas as pd  # noqa
import matplotlib.pyplot as plt  # noqa

import torch  # noqa
from SIRNet import util, trainer  # noqa
from SIRNet.data_collection import retrieve_data  # noqa
from SIRNet import metrics  # noqa

if __name__ == '__main__':
    pipeline(args)  # noqa
