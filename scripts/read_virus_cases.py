# Washington, Texas and New york.

# Avg confirmed- active confirmed - recovered - death

import numpy as np
import pandas as pd
import seaborn as sns

import os
import urllib.request
from . import read_new_data as rd

sns.set()

df = pd.read_csv('virus.csv')
full_key_list = []
count = 0
for keys in df:
    # print (keys)
    count += 1
    full_key_list.append(keys)

# print (count)
full_key_list.pop(0)


# print (full_key_list)


class preprocess_new_virus_data:

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
            # print(arg)
            if (arg in keylist):
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
            # newlist = [func(arr[i]) for i in range(len(arr))]
            # a = np.array(newlist[45])
            # print (a)

            for i in range(len(arr)):

                temp_list = func(arr[i])
                # if (count ==1):
                # print (temp_list)
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
            # print (arr[45])
            if (count == 1):
                avg = np.array(avg_confirmed)
                print("The shape of average is ", avg.shape)
                act = np.array(active_confirmed)
                rec = np.array(recovered)
                dea = np.array(death)
            else:
                # avg = np.concatenate((avg, np.array(active_confirmed)), axis =0)
                # act = np.concatenate((act, np.array(active_confirmed)), axis=0)
                # rec = np.concatenate((rec, np.array(active_confirmed)), axis=0)
                # dea = np.concatenate((dea, np.array(active_confirmed)), axis=0)
                avg = np.vstack((avg, np.array(avg_confirmed)))
                act = np.vstack((act, np.array(active_confirmed)))
                rec = np.vstack((rec, np.array(recovered)))
                dea = np.vstack((dea, np.array(death)))
            # if (count==2):
            # print (avg, "This is the avg", avg.shape)
        # print (count, len(arr), len(avg))
        # avg = np.reshape(avg,(len(arr), count))
        # print (avg[:,0], "the reshaped array", avg[:,0].shape)
        # act = np.reshape(act, (len(arr), count))
        # rec = np.reshape(rec, (len(arr), count))
        # dea = np.reshape(dea, (len(arr), count))
        # print (avg.shape)
        return avg, act, rec, dea

    def create_nday_list(self, n, data_list):
        n_day_list = [int(data_list[i + n]) - int(data_list[i]) for i in
                      range(len(data_list) - n)]
        for j in range(n):
            var = int(data_list[n - j - 1])
            n_day_list.insert(0, var)
        return n_day_list

    def clear_list(self, list):
        for i in range(len(list)):
            if (list[i] == 0):
                list[i] = ''
            else:
                break
        return list

    def create_csv(self, full_key_list, pop_list, aggregated_confirmed,
                   active_confirmed, recovered, death):
        # full_key_list, temp_dict = self.create_keylist(path)
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
            # print (i, "******")
            case_dict = {}
            # print (aggregated_confirmed.shape)
            agg_numpy = aggregated_confirmed[i].astype(int)
            rec_numpy = recovered[i].astype(int)
            dea_numpy = death[i].astype(int)

            case_dict['aggregated_confirmed'] = aggregated_confirmed[
                i]  # agg_numpyy[i]/int(pop_list[i])
            agg_numpy = aggregated_confirmed[i].astype(int)
            rec_numpy = recovered[i].astype(int)
            dea_numpy = death[i].astype(int)
            temp = np.add(rec_numpy, dea_numpy)
            actual_active_confirmed = np.subtract(agg_numpy, temp)

            case_dict['active_confirmed'] = actual_active_confirmed
            case_dict['recovered'] = recovered[i]
            case_dict['death'] = death[i]

            temp = aggregated_confirmed[i].tolist()
            # print (temp)
            # one_day_list = [active_confirmed[i][j+1] - active_confirmed[i][j] for j in range(len(active_confirmed[i])-1)]

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

            # for keys in new_df:
            #     print (keys)
            #     val_dict = new_df[keys]
            #     for key in val_dict:
            #         val_list = val_dict[key]
            #         for i in range(len(val_list)):
            #             print (val_list)
            #             if (val_list[i]==0):
            #                 val_list[i] =''
            #
            #             break
            #         val_dict[key] = val_list
            #     new_df[keys] = val_dict
            if (csv_name_list[count] == 'Total_cases' or csv_name_list[
                count] == "Death" or csv_name_list[count] == 'Recovered'):
                print("Entered the loop")
                new_df.replace(0, '', inplace=True)
            # new_df[orig_dict.isnull()] = ''
            # print (new_df['anhui'], "*******\n", arg)
            new_df.to_csv(csv_name_list[count] + "_data.csv")
            count += 1


virus_cities = rd.PreprocessVirusData()
filename = os.path.join(os.path.dirname(__file__), '..', 'data',
                        'Data319.xlsx')
new_list, dict = virus_cities.create_keylist(filename)
filtered_list = virus_cities.filter_list(new_list)

print(filtered_list)

new_list = [filtered_list[i] for i in range(len(filtered_list)) if i % 2 == 0]
print(new_list)

population_list = [filtered_list[i] for i in range(len(filtered_list)) if
                   i % 2 == 1]

new_list = [new_list[i].split("-", 1)[0] for i in range(len(new_list))]
print(new_list, "\n", len(new_list))
virus_methods = preprocess_new_virus_data(
    new_list)  # Edit for whichever area you need the data for.
key_list, dict, orig_dict = virus_methods.create_keylist_path(
    urllib.request.urlopen("http://hgis.uw.edu/virus/assets/virus.csv"))
updated_dict = virus_methods.select_keys(keylist=key_list, dict=dict)
(aggregated_confirmed, active_confirmed,
 recovered, death) = virus_methods.format_data(
    updated_dict
)  # Obtain the final np arrays.
(country_dict, total_dict, active_dict, recovered_dict, death_dict,
 one_day_dict, three_day_dict, seven_day_dict) = virus_methods.create_csv(
    new_list, population_list, aggregated_confirmed, active_confirmed,
    recovered, death
)

print(total_dict.keys())
virus_methods.save_csv(orig_dict, country_dict, total_dict, active_dict,
                       recovered_dict, death_dict, one_day_dict, three_day_dict,
                       seven_day_dict)
df = pd.read_csv("Total_cases_data.csv")
df2 = pd.read_csv("Death_data.csv")
df3 = pd.read_csv("Recovered_data.csv")
df.replace(0, '', inplace=True)
df2.replace(0, '', inplace=True)
df3.replace(0, '', inplace=True)
df.to_csv("Total_cases_data.csv")
df2.to_csv("Death_data.csv")
df3.to_csv("Recovered_data.csv")
print(aggregated_confirmed.shape, active_confirmed.shape)

# print (aggregated_confirmed[0].shape, aggregated_confirmed[0:,])
# country_dict={}
# total_dict = {}
# active_dict = {}
# recovered_dict = {}
# death_dict = {}
#
# for i in range(len(full_key_list)):
#
#     case_dict = {}
#     case_dict['aggregated_confirmed'] = aggregated_confirmed[i]
#     case_dict['active_confirmed'] = active_confirmed[i]
#     case_dict['recovered'] = recovered[i]
#     case_dict['death'] = death[i]
#
#     country_dict[full_key_list[i]] = {}
#     country_dict[full_key_list[i]] = case_dict
#
#     total_dict[full_key_list[i]] = list(aggregated_confirmed[i])
#     print (aggregated_confirmed[i].shape)
#     active_dict[full_key_list[i]] = list(active_confirmed[i])
#     recovered_dict[full_key_list[i]] = list(recovered[i])
#     death_dict[full_key_list[i]] = list(death[i])
#
# country_df = pd.DataFrame.from_dict(country_dict)
# country_df.to_csv("country_edit_data.csv")
#
# total_df = pd.DataFrame.from_dict(total_dict,orient="index")
# total_df.to_csv("total_edit_data.csv",index=True)
#
# active_df = pd.DataFrame.from_dict(active_dict,orient="index")
# active_df.to_csv("active_edit_data.csv", index=True)
#
# recovered_df = pd.DataFrame.from_dict(recovered_dict,orient="index")
# recovered_df.to_csv("recovered_edit_data.csv", index =True)
#
# death_df = pd.DataFrame.from_dict(death_dict,orient="index")
# death_df.to_csv("death_edit_data.csv", index=True)

# print (avg,'\n',active,'\n',reco,'\n',death)
# print (avg.shape)
# print(a)
# print (key_list)
#
# for key in updated_dict:
#     print(key)
#
# print (updated_dict['new york'])
