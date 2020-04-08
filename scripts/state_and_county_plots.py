import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)


# Step 1 - Update dates
# Step 2 - Update cases
# Step 3 - Update days
# Step 4 - Reshape to days,cases

class StateCountyPlots:

    def __init__(self, state, *args):
        self.state = state
        self.args = args

    #############################################################
    # Read the county and the state files.                      #
    # Source - https://github.com/nytimes/covid-19-data         #
    #############################################################
    @staticmethod
    def read_csvs(path1, path2):
        df_state = pd.read_csv(path1)
        df_county = pd.read_csv(path2)
        return df_state, df_county

    #############################################################
    # Scraping off the required county and statedata  from the  #
    # dataset. df_state_r is the state dataframe, and a         #
    # dictionary of counties in it.                             #
    #############################################################
    def get_state_county_data(self, df_state, df_county):
        df_state_r = df_state.loc[df_state['state'] == self.state]
        df_state_r = df_state_r[['date', 'state', 'cases']].reset_index()
        df_state_counties = df_county.loc[df_county['state'] == self.state]
        df_state_counties = df_state_counties[
            ['date', 'county', 'cases']].reset_index()
        return df_state_r, df_state_counties

    #############################################################
    # Decimating the unnecessary columns of data from the state #
    # and county dataset.                                       #
    #############################################################

    def get_counties_df(self, df_state_r, df_state_counties):
        df_state_counties.sort_values(by=['cases'])
        df_state_r.sort_values(by=['cases'])
        df_dict = {}
        df_state_r = df_state_r[['date', 'state', 'cases']]
        df_dict[self.state] = df_state_r
        for arg in self.args:
            df_dict[arg] = df_state_counties.loc[
                df_state_counties['county'] == arg]
            temp_df = df_dict[arg]
            df_dict[arg] = temp_df[['date', 'county', 'cases']].reset_index()
        return df_dict

    #############################################################
    # Update the date column in the dictionary for the map      #
    #############################################################

    @staticmethod
    def create_dict_list(df_state_r):
        tick_list = df_state_r['date'].tolist()
        date_list = [(str('Feb-') + str(tick_list[i].split('-')[2])) if int(
            tick_list[i].split('-')[1]) == 2 else ( \
                    str('March-') + str(tick_list[i].split('-')[2])) \
                     for i in range(len(tick_list))]
        return tick_list, date_list

    #############################################################
    # Update the date column for the date in the dictionary     #
    #############################################################

    @staticmethod
    def update_dates(df, tick_list, date_list):
        func = lambda x: max(
            [i if x[0] == tick_list[i] else 0 for i in range(len(tick_list))])
        df_list = df['date']
        index = func(df_list)
        df['date'] = date_list[index:]
        return df

    #############################################################
    # Adding new day column to the dataframe                    #
    #############################################################

    @staticmethod
    def update_days(df):
        days = len(df['cases'].tolist())
        day_list = np.arange(0, days, 1)
        df['days'] = list(day_list)
        df = df[['days', 'cases']].reset_index()
        return df

    #############################################################
    # Adding special case updates to the data for Texas and     #
    # Bexar County                                              #
    #############################################################

    @staticmethod
    def update_special_case(df):
        df_list = df['cases'].tolist()
        print(df_list, "**********************", len(df_list))
        df_list = [int(df_list[i]) - 9 for i in range(len(df_list))]
        print(df_list, "********************")
        df['cases'] = df_list
        return df

    #############################################################
    # Updating based on the number of cases to start from origin#
    #############################################################

    @staticmethod
    def case_val_updates(n, df):
        df = df[df['cases'] > n]
        return df

    #############################################################
    # Apply the above updates to the dataframe                  #
    #############################################################

    def apply_date_updates(self, df_dict, tick_list, date_list):
        for key in df_dict:
            df_dict[key] = self.update_dates(df_dict[key], tick_list, date_list)
            if key == 'Bexar' or key == 'Texas':
                df_dict[key] = self.update_special_case(df_dict[key])
            df_dict[key] = self.update_days(
                self.case_val_updates(10, df_dict[key]))
        return df_dict

    ###############################################################
    # List to show what if cases increment by N times every n days#
    ###############################################################

    def create_multiplier_arr(self, df_dict, n, N):
        temp = df_dict[self.state]
        x = np.ones(len(temp['cases']))
        double_days = [N * (2 ** (i / n)) for i in range(len(x))]
        return double_days

    #############################################################
    # Generate the plot for the given counties and state        #
    #############################################################

    def plotting_function(self, df_dict, palette, x):
        for key in df_dict:
            if key != self.state:
                string = str(key) + " county"
            else:
                string = str(key)
            sns.lineplot('days', 'cases', data=df_dict[key], linestyle="-",
                         marker='o', palette=palette, label=string)

        for i in range(3):
            plt.plot(x[i], color='lightgray', linestyle='--')


def main():
    scplot = StateCountyPlots('Texas', 'Bexar', 'Dallas', 'Travis', 'Harris',
                              'Denton')
    states_fn = os.path.join(os.path.dirname(__file__), '..', 'data',
                             "us-states.csv")
    counties_fn = os.path.join(os.path.dirname(__file__), '..', 'data',
                               "us-counties.csv")
    df_state, df_county = scplot.read_csvs(states_fn, counties_fn)
    df_state_r, df_state_counties = scplot.get_state_county_data(df_state,
                                                                 df_county)
    df_dict = scplot.get_counties_df(df_state_r, df_state_counties)

    tick_list, date_list = scplot.create_dict_list(df_state_r)
    df_dict = scplot.apply_date_updates(df_dict, tick_list, date_list)
    day_list = [1, 2, 5]
    x = scplot.create_multiplier_arr(df_dict, day_list[0], 10)
    for i in range(1, 3, 1):
        x = np.vstack(
            (x, scplot.create_multiplier_arr(df_dict, day_list[i], 10)))

    sns.set()
    sns.set_style('whitegrid')

    palette = sns.color_palette("mako_r", 6)
    scplot.plotting_function(df_dict, palette, x)
    sns.despine()

    title_font = {'fontname': 'Arial', 'size': '14', 'color': 'black',
                  'weight': 'normal',
                  'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
    axis_font = {'fontname': 'Arial', 'size': '12'}

    plt.rcParams.update({'font.size': 26})

    plt.xlabel("Number of days since 10th case", **axis_font)
    plt.ylim(1e1, 1e4)
    plt.ylabel("Total Cases (log scale)", **axis_font)
    plt.title("COVID-19 cases in Texas State and Counties", fontweight='bold',
              **title_font)
    plt.yscale(value='log')
    plt.xticks([0, 5, 10, 15, 20])
    plt.margins(0)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
