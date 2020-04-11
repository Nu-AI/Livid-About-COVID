import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)

class Mobility:

    ########################################################################
    # Take the state and county input, replace the state with US and county#
    # with state in the case of trying to get state mobility data.         #
    ########################################################################

    def __init__(self, state, county):
        self.county = county
        self.state = state

    ########################################################################
    # Scrape the county data from the state dataframe                      #
    ########################################################################
    def get_county_df(self):
        state_string = self.state+"_mobility.csv"
        state_string=state_string.replace(" ", "_",1)
        df = pd.read_csv(state_string)
        if (self.state == 'US'):
            county_string = self.county
        else:
            county_string = self.county + " County"
        return df, df[df['county'] == county_string]


    ########################################################################
    # Save the county dataframe to a csv in case you need it               #
    ########################################################################
    def save_county_df(self,df):
        string = self.county + "_mobility.csv"
        df.to_csv(string)


    ########################################################################
    # Trim the date and values from the list and form a new dataframe for  #
    # with the temporal change data for the given county                   #
    ########################################################################
    def update_county_df(self,df):
        new_dict = {}

        keylist = df['category'].tolist()
        date_list = df['dates'].tolist()
        values = df['values'].tolist()

        trim_func = lambda x: [i.replace('[', '').replace(']', '').replace(' ', '') for i in x]
        ref_list_size = len(trim_func(values[0].split(',')))

        new_dict['dates'] = trim_func(date_list[0].split(','))
        new_dict['days'] = np.arange(0,ref_list_size,1)

        for i in range(0, len(keylist)):
            list_val = trim_func(values[i].split(','))
            diff = ref_list_size - len(list_val)
            if (diff > 0):
                for j in range(diff):
                    list_val.insert(0, 0)
            new_dict[keylist[i]] = np.array(list_val)

        return pd.DataFrame.from_dict(new_dict)

    ########################################################################
    # Generates the plot for the percent change in mobility with respect to#
    # the number of days since it started tracking.                        #
    ########################################################################

    def plotting_func(self,df2):
        fig = go.Figure()
        count = 0
        max_val = 0
        min_val = 0
        day_arr = np.arange(0,len(df2['days'].tolist()),1)
        for keys in df2:

            if (keys=='days' or keys=='dates'):
                print ("Iteration")
            else:
                fig.add_trace(go.Scatter(x=df2['days'],y=df2[keys], mode='lines+markers',name=keys))
                temp_arr = np.array(list(df2[keys]))
                temp_arr = temp_arr.astype(float)
                max_val  = max(max_val,max(temp_arr.astype(int)))

                min_val = min(min_val,min(temp_arr.astype(int)))
                plt.plot(day_arr, temp_arr, linestyle="-", marker='o', label=keys)
                if (self.state == 'US'):
                    title_string = str(self.county) + " state"
                else:
                    title_string = str(self.county) + " county"
        print (max_val, min_val)
        fig.update_layout(title = {'text':"Change in the mobility in "+ title_string, 'x':0.6,'xanchor':'right', 'yanchor':'top', 'font':{'size':16}}, xaxis=dict(title='<i>Number of days (starting 23rd Feb)</i>',showticklabels=True,),
                           yaxis=dict(title='<i>Percent change in mobility</i>',showticklabels=True,))
        title_font = {'fontname':'Arial', 'size':'14', 'color':'black', 'weight':'normal',
                      'verticalalignment':'bottom'} # Bottom vertical alignment for more space
        axis_font = {'fontname':'Arial', 'size':'12'}
        #
        plt.rcParams.update({'font.size': 12})
        #
        plt.xlabel("Number of days (starting 23rd-Feb)", **axis_font)

        func_ylim_up = lambda x: x if 20*x>max_val else func_ylim_up(x+1)
        func_ylim_down = lambda x: x if 20*x < min_val else func_ylim_down(x-1)
        plt.ylim(20*func_ylim_down(0),20*func_ylim_up(0))

        plt.ylabel("Change in percent", **axis_font)
        plt.title("Mobility change in " + title_string, fontweight = 'bold', **title_font)
        # plt.yscale(value='log')
        plt.xticks([0,5,10,15,20,25,30,35,40,45,50])
        plt.margins(0)
        plt.legend(frameon=False)
        plt.tight_layout()
        # Get the compact plot
        plt.show()
        # Get the wider plot
        #fig.add_trace(go.Scatter(x=df2['days'],y=df2['Workplace'], mode='lines+markers',name='Workplace'))
        fig.show()


def main():
    m = Mobility("New_York", "Kings")
    df, county_df = m.get_county_df()
    print (county_df.keys())
    new_df = m.update_county_df(county_df)
    print (new_df)
    m.plotting_func(new_df)

if __name__=="__main__":
    main()
