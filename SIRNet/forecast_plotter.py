import os
from datetime import datetime, timedelta
from collections import OrderedDict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import seaborn as sns

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')

RESULTS_DIR = os.path.join(ROOT_DIR, 'Prediction_results')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# # scenario_list = ['Current Mobility', '20% Return to Normal',
#                  '50% Return to Normal', 'Return to Normal Mobility']
scenario_list = ['20% Mobility', 'Normal Mobility', '50% Mobility',
                 '75% Mobility']


def get_predictions(path):
    return np.load(path, allow_pickle=True)


def get_scenario_dict(scenario_list, county_name):
    string = os.path.join(RESULTS_DIR, "Average Case ")
    pathlist = [string + i + '.npy' for i in scenario_list]
    # pathlist = ["Average Case " + i + county_name + '.npy' for i in
    #             scenario_list]
    dict = {}
    for i in range(len(pathlist)):
        dict[scenario_list[i]] = get_predictions(pathlist[i])
    return dict


# Keys
# Active Cases (observed)
# Hospitalized
# Total Cases (observed)
# Total Cases (latent)
# Active Cases (latent)

population = 2003554


def get_arrays(dict, scenario_list, population):
    data_list = []
    day_list = []

    for scenario in scenario_list:
        avg_arr = dict[scenario]

        # Creating the key based arrays. Have this in casee want to access the specific arrays
        total_deaths_arr = avg_arr.item().get('Total Deaths') * population
        active_cases_obs_arr = avg_arr.item().get(
            'Active Cases (observed)') * population
        hospitalized_arr = avg_arr.item().get('Hospitalized') * population
        Total_cases_obs = avg_arr.item().get(
            'Total Cases (observed)') * population
        Total_cases_lat = avg_arr.item().get(
            'Total Cases (latent)') * population
        Active_cases_lat = avg_arr.item().get(
            'Active Cases (latent)') * population

        data_arr = np.array([])

        # This will have the data for all the keys for a given scenario
        # Organize it based on the way you want

        data_arr = np.vstack((active_cases_obs_arr, Total_cases_obs))
        data_arr = np.vstack((data_arr, total_deaths_arr))
        data_arr = np.vstack((data_arr, hospitalized_arr))

        # Left out since they were not needed for a certain case but can be uncommented
        # data_arr = np.vstack((data_arr, Total_cases_lat))
        # data_arr = np.vstack((data_arr, total_deaths_arr))

        day_arr = np.arange(0, len(total_deaths_arr), 1)

        data_list.append(data_arr)
        day_list.append(day_arr)

    return data_list, day_list


def plot_data(data_list, day_list, legend_list, gridplot, gt_arr=None):
    # Plotting parameters
    plt.figure(dpi=100)
    plt.rcParams['axes.linewidth'] = 1

    # fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 14, 'legend.labelspacing': 1.3})
    count_x = 0
    count_y = 0
    max = 0
    x = 0
    sns.set()
    sns.set_style('whitegrid')
    plt.grid(False)
    sns.despine()
    palette = sns.color_palette("mako_r", 6)
    count = 0

    # Set of tunable parameters for the plots
    linewidth = 2
    color_list = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink',
                  'gray', 'olive', 'cyan']
    map_list = ['a', 'b', 'c', 'd']
    xlabel = "Months"
    ylabel = "Number of cases (in thousands)"
    x_labelpad = 10
    y_labelpad = 30

    xtick_rotation = 45

    dates = ["2020-02-21",
             "2020-08-21"]  # the range of dates you want the generated list to be
    start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    labels = lambda x, y: list(OrderedDict(
        ((x + timedelta(_)).strftime(r"%b"), None) for _ in
        range((y - x).days)).keys())
    max_days = (end - start).days
    xticklabel = labels(start, end)

    legend_size = 13

    # Change the title, axis, label and tick font settings here
    title_font = {'fontname': 'Arial', 'size': '16', 'color': 'black',
                  'weight': 'bold',
                  'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
    axis_font = {'fontname': 'Arial', 'size': '14', 'weight': 'bold'}
    sp_label_font = {'fontname': 'Arial', 'size': '15', 'weight': 'bold'}
    tick_font = {'fontname': 'Arial', 'size': '12', 'weight': 'bold'}
    label_font = {'fontname': 'Arial', 'size': '13', 'weight': 'bold'}

    log_scale = 0

    if (gridplot != 1):
        for i in range(len(data_list)):
            day_arr = day_list[i]
            data_arr = data_list[i]
            label_string1 = map_list[i] + "1. {} (Active)".format(
                legend_list[i])
            label_string2 = map_list[i] + "1. {} (Total)".format(legend_list[i])
            plt.plot(day_arr[0:max_days], data_arr[0][0:max_days],
                     label=label_string1, linestyle="--",
                     color=color_list[i], linewidth=linewidth)
            plt.plot(day_arr[0:max_days], data_arr[1][0:max_days],
                     label=label_string2,
                     color=color_list[i], linewidth=linewidth)

        if gt_arr is not None:
            plt.plot(day_arr, gt_arr)

        plt.xlabel(xlabel, **sp_label_font, labelpad=x_labelpad)
        plt.ylabel(ylabel, **sp_label_font, labelpad=y_labelpad)

        plt.xticks(range(0, max_days, int(max_days / len(xticklabel))),
                   xticklabel, rotation=xtick_rotation,
                   **tick_font)
        if (log_scale):
            plt.yscale('log')
        # plt.yticks(np.arange(0, 1000000, 200000), [0, 200, 400, 600, 800],
        #           **tick_font)
        plt.legend(prop={'size': legend_size, 'weight': 'bold'})
        plt.margins(0)
        manager = plt.get_current_fig_manager()
        plot_backend = mpl.get_backend()
        if plot_backend == 'TkAgg':
            manager.resize(*manager.window.maxsize())
        elif plot_backend == 'wxAgg':
            manager.frame.Maximize(True)
        elif plot_backend == 'Qt4Agg':
            manager.window.showMaximized()
        # plt.tight_layout(pad=0.5)
        # plt.savefig("Bexar_total.pdf")
        plt.show()

    else:
        fig, ax = plt.subplots(1, 2)

        while (count_y < 2):
            ax1 = ax[count_y]
            ax1.grid(False)
            print(count_y)
            for i in range(len(data_list)):
                day_arr = day_list[i]
                data_arr = data_list[i]
                if (count_y):
                    temp = 1
                    label_string = map_list[i] + ". " + legend_list[
                        i] + " (Total)"
                    linestyle = "-"
                else:
                    temp = 0
                    label_string = map_list[i] + ". " + legend_list[
                        i] + " (Active)"
                    linestyle = "--"
                ax1.plot(day_arr[0:max_days], data_arr[temp][0:max_days],
                         label=label_string, linestyle=linestyle,
                         color=color_list[i], linewidth=linewidth)

            ax1.set_xlabel(xlabel, **sp_label_font)
            ax1.set_ylabel(ylabel, **sp_label_font)

            ax1.set_xticks(range(0, max_days, int(max_days / len(xticklabel))))
            ax1.set_xticklabels(xticklabel, rotation=xtick_rotation,
                                **tick_font)
            if (log_scale):
                ax1.set_yscale('log')
            ax1.legend(frameon=False)
            # ax1.set_title(title, ** title_font)
            ax1.set_yticklabels(ax1.get_yticks(), **tick_font)
            mf = mtick.ScalarFormatter(useMathText=True)
            mf.set_powerlimits((-5, 5))
            ax1.yaxis.set_major_formatter(mf)
            count_y += 1
        # fig.suptitle("Cases based on mobility " , y=0.95, fontweight='bold', **title_font)
        # fig.text(0.5, 0.03, 'Days (Feb 23 - April 5, 2020)', ha='center', **axis_font)
        # fig.text(0.05, 0.5, 'Percent change', va='center', rotation='vertical', **axis_font)
        # fig.tight_layout()
        # plt.savefig("_split_plots.pdf", orientation='portrait')
        plt.show()

# legend_list = ['Current Mobility', '20% Mobility', '50% Mobility', 'Normal Mobility']
# data_list, day_list = get_arrays(get_scenario_dict(scenario_list), scenario_list,population)
# plot_data(data_list, day_list,legend_list,0)
