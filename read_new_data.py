
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.animation as animate
sns.set()
import csv

class preprocess_virus_data:

    def __init__(self,*args):
        self.args = args

    #######################################################################################
    # The method below reads the csv file and updates the list of keys specified in the   #
    # file given.                                                                         #
    #######################################################################################

    def create_keylist(self,path):
        #temp_dict = pd.read_csv(path,error_bad_lines=False)
        temp_dict = pd.read_excel(path)
        temp_dict.fillna(0, inplace=True)
        key_list = []
        new_list = [keys for keys in temp_dict]
        return new_list, temp_dict


    #######################################################################################
    # The method creates a new dictionary with the updated keylist that is required by    #
    # the user.                                                                           #
    #######################################################################################

    def select_keys(self, keylist, dict):
        updated_dict = {}
        #updated_dict[keylist[0]] = self.get_datetime(keylist, dict)

        for arg in self.args:
            print(arg)
            if (arg in keylist):
                updated_dict[arg] = dict[arg]

        return updated_dict

    #######################################################################################
    # The method removes the unnecessary columns which show population density and the    #
    # the hospitals per population.                                                       #
    #######################################################################################

    def filter_list(self, alist):
        active_list = []
        active_list = list(filter(lambda x:("Unnamed" not in str(x)), alist))
        return active_list


    def update_dict(self,filtered_list):
        updated_dict = {}
        china_list = []

        for i in range(0, len(filtered_list), 2):

            temp_list = list(dict[filtered_list[i]])
            if ('Active' in temp_list):
                temp_list.remove('Active')
            elif ('Total' in temp_list):
                temp_list.remove('Total')
            else:
                temp_list.remove('ActiveS')
            temp_array = np.array(temp_list, dtype=np.float64)
            # print(int(filtered_list[i+1]))
            # print(temp_array,i)
            updated_dict[filtered_list[i]] = temp_array * (int(filtered_list[i + 1]))
            china_list.append(filtered_list[i])

        return updated_dict, china_list

    def get_final_dict(self,updated_dict):
        final_dict = {}
        for key in updated_dict:
            new_arr = np.ceil(np.array(list(updated_dict[key])))
            final_dict[key] = new_arr[0:48]
        return final_dict

    def plot_grid(self,final_dict,china_list):
        bins = len(final_dict[china_list[0]])
        x_bins = np.arange(0,bins,1)

        for i in range(1, len(china_list)):
            plt.subplot(4, 4, i)
            plt.bar(x_bins, np.array(list(final_dict[china_list[i]])), alpha=0.8)
            plt.title(china_list[i])
            plt.xlabel('Days')
            plt.ylabel('Total cases')

        plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=0.5)
        plt.show()

a = preprocess_virus_data()
new_list,dict = a.create_keylist('Data319.xlsx')
filtered_list = a.filter_list(new_list)

print(filtered_list)


updated_dict, china_list = a.update_dict(filtered_list)
#print (updated_dict.keys())
#final_dict = a.get_final_dict(updated_dict)

#a.plot_grid(final_dict,china_list)

# Unused lambda function
#func = (lambda x: [np.array(list(dict[x[i]]))*int(x[i+1]) for i in range(0, (len(x)-1), 2)])
#func(filtered_list)

# Saving the csv file
#pd.DataFrame(final_dict).to_csv('active_world.csv', index = False)
