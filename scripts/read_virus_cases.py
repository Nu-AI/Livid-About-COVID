# Washington, Texas and New york.

# Avg confirmed- active confirmed - recovered - death

import numpy as np
import pandas as pd
import seaborn as sns

import os
import urllib.request
import sys

# Fix path to allow for import from scripts
if os.path.basename(os.getcwd()) == 'scripts':
    sys.path.append('..')
elif 'scripts' in os.listdir():
    sys.path.append('.')

from scripts import read_new_data as rd

sns.set()

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def dpath(filename):
    """Yield path to file in data directory"""
    return os.path.join(DATA_DIR, filename)


class PreprocessNewVirusData:

    def __init__(self, args):
        self.args = args

    #######################################################################################
    # The method below reads the csv file and updates the list of keys specified in the   #
    # file given.                                                                         #
    #######################################################################################

    def create_keylist_path(self, path):
        temp_dict = pd.read_csv(path)
        orig_dict = temp_dict
        temp_dict.fillna(0, inplace=True)
        new_list = [keys for keys in temp_dict]
        return new_list, temp_dict, orig_dict

    def create_keylist_df(self, df):
        df.fillna(0, inplace=True)
        new_list = [keys for keys in df]
        return new_list

    def get_datetime(self, keylist, dict):
        return dict[keylist[0]]

    #######################################################################################
    # The method creates a new dictionary with the updated keylist that is required by    #
    # the user.                                                                           #
    #######################################################################################

    def select_keys(self, keylist, dict):
        updated_dict = {}
        updated_dict[keylist[0]] = self.get_datetime(keylist, dict)
        for arg in self.args:
            if arg in keylist:
                updated_dict[arg] = dict[arg]

        return updated_dict

    ########################################################################################
    # This method formats the given county data into 4 different numpy arrays with the     #
    # given values. The first is the aggregated confirmed, then active confirmed, recovered#
    # and then death. The Nan values in the csv were replaced with 0                       #
    ########################################################################################

    def format_data(self, dict):
        count = 0
        avg = np.array([])
        act = np.array([])
        rec = np.array([])
        dea = np.array([])
        func = lambda x: x.split("-") if x != 0 else [0]
        for arg in self.args:
            avg_confirmed = []
            active_confirmed = []
            recovered = []
            death = []
            count += 1
            arr = dict[arg]
            arr = list(arr)

            for i in range(len(arr)):
                temp_list = func(arr[i])
                if len(temp_list) == 4:
                    avg_confirmed.append(temp_list[0])
                    active_confirmed.append(temp_list[1])
                    recovered.append(temp_list[2])
                    death.append(temp_list[3])
                elif len(temp_list) == 3:
                    avg_confirmed.append(temp_list[0])
                    active_confirmed.append(temp_list[1])
                    recovered.append(temp_list[2])
                    death.append(0)
                elif len(temp_list) == 2:
                    avg_confirmed.append(temp_list[0])
                    active_confirmed.append(temp_list[1])
                    recovered.append(0)
                    death.append(0)
                else:
                    avg_confirmed.append(temp_list[0])
                    active_confirmed.append(0)
                    recovered.append(0)
                    death.append(0)
            if count == 1:
                avg = np.array(avg_confirmed)
                print("The shape of average is ", avg.shape)
                act = np.array(active_confirmed)
                rec = np.array(recovered)
                dea = np.array(death)
            else:
                avg = np.vstack((avg, np.array(avg_confirmed)))
                act = np.vstack((act, np.array(active_confirmed)))
                rec = np.vstack((rec, np.array(recovered)))
                dea = np.vstack((dea, np.array(death)))
        return avg, act, rec, dea

    @staticmethod
    def create_nday_list(n, data_list):
        n_day_list = [int(data_list[i + n]) - int(data_list[i]) for i in
                      range(len(data_list) - n)]
        for j in range(n):
            var = int(data_list[n - j - 1])
            n_day_list.insert(0, var)
        return n_day_list

    @staticmethod
    def clear_list(list_):
        for i in range(len(list_)):
            if list_[i] == 0:
                list_[i] = ''
            else:
                break
        return list_

    def create_csv(self, full_key_list, pop_list, aggregated_confirmed,
                   active_confirmed, recovered, death):
        country_dict = {}
        total_dict = {}
        active_dict = {}
        recovered_dict = {}
        death_dict = {}

        one_day_change = {}
        three_day_change = {}
        seven_day_change = {}

        print(len(full_key_list))
        for i in range(len(full_key_list)):
            case_dict = {
                'aggregated_confirmed': aggregated_confirmed[i]
            }

            agg_numpy = aggregated_confirmed[i].astype(int)
            rec_numpy = recovered[i].astype(int)
            dea_numpy = death[i].astype(int)
            temp = np.add(rec_numpy, dea_numpy)
            actual_active_confirmed = np.subtract(agg_numpy, temp)

            case_dict['active_confirmed'] = actual_active_confirmed
            case_dict['recovered'] = recovered[i]
            case_dict['death'] = death[i]

            temp = aggregated_confirmed[i].tolist()

            one_day_change[full_key_list[i]] = self.clear_list(
                list(self.create_nday_list(1, temp)))

            three_day_change[full_key_list[i]] = self.clear_list(
                list(self.create_nday_list(3, temp)))

            seven_day_change[full_key_list[i]] = self.clear_list(
                list(self.create_nday_list(7, temp)))

            country_dict[full_key_list[i]] = {}
            country_dict[full_key_list[i]] = case_dict

            agg_list = aggregated_confirmed[i]
            total_dict[full_key_list[i]] = self.clear_list(list(agg_list))
            print(aggregated_confirmed[i].shape, "**************",
                  aggregated_confirmed[i][0])
            active_dict[full_key_list[i]] = self.clear_list(
                list(actual_active_confirmed))
            recovered_dict[full_key_list[i]] = self.clear_list(
                list(recovered[i]))
            death_dict[full_key_list[i]] = self.clear_list(list(death[i]))

        return country_dict, total_dict, active_dict, recovered_dict, death_dict, one_day_change, three_day_change, seven_day_change

    def save_csv(self, orig_dict, *args):
        csv_name_list = ['Full', 'Total_cases', 'Active_cases', 'Recovered',
                         'Death', 'One-day-change', 'Three-day-change',
                         'Seven-day-change']
        count = 0
        for arg in args:

            new_df = pd.DataFrame.from_dict(arg, orient="index")
            new_df = new_df.transpose()
            if (csv_name_list[count] == 'Total_cases' or
                    csv_name_list[count] == "Death" or
                    csv_name_list[count] == 'Recovered'):
                print("Entered the loop")
                new_df.replace(0, '', inplace=True)
            new_df.to_csv(dpath(csv_name_list[count] + "_data.csv"))
            count += 1


virus_cities = rd.PreprocessVirusData()
filename = dpath('Data319.xlsx')
new_list, dict = virus_cities.create_keylist(filename)
filtered_list = virus_cities.filter_list(new_list)

print(filtered_list)

new_list = [filtered_list[i] for i in range(len(filtered_list)) if i % 2 == 0]
print(new_list)

population_list = [filtered_list[i] for i in range(len(filtered_list)) if
                   i % 2 == 1]

new_list = [new_list[i].split("-", 1)[0] for i in range(len(new_list))]
print(new_list, "\n", len(new_list))
virus_methods = PreprocessNewVirusData(
    new_list)  # Edit for whichever area you need the data for.
key_list, dict, orig_dict = virus_methods.create_keylist_path(
    urllib.request.urlopen("http://hgis.uw.edu/virus/assets/virus.csv"))
updated_dict = virus_methods.select_keys(keylist=key_list, dict=dict)
(aggregated_confirmed, active_confirmed,
 recovered, death) = virus_methods.format_data(
    updated_dict
)
# Obtain the final np arrays.
(country_dict, total_dict, active_dict, recovered_dict, death_dict,
 one_day_dict, three_day_dict, seven_day_dict) = virus_methods.create_csv(
    new_list, population_list, aggregated_confirmed, active_confirmed,
    recovered, death
)

print(total_dict.keys())
virus_methods.save_csv(orig_dict, country_dict, total_dict, active_dict,
                       recovered_dict, death_dict, one_day_dict, three_day_dict,
                       seven_day_dict)

# Special Cases
df = pd.read_csv(dpath("Total_cases_data.csv"))
df2 = pd.read_csv(dpath("Death_data.csv"))
df3 = pd.read_csv(dpath("Recovered_data.csv"))
df.replace(0, '', inplace=True)
df2.replace(0, '', inplace=True)
df3.replace(0, '', inplace=True)
df.to_csv(dpath("Total_cases_data.csv"))
df2.to_csv(dpath("Death_data.csv"))
df3.to_csv(dpath("Recovered_data.csv"))
