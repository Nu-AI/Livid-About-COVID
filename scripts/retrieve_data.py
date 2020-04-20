import pandas as pd
import urllib.request
import click

## Steps to do ---
# Input will be county name and state name
# If all counties are required we take in all as the input
# Gather the mobility data
# Gather the pop data from the pop files ( create a lookup table for that first)
# Gather the county names and active cases data

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)

# Setting the parameters for the data required.
# If require data for only country or states, then set counties to None
defaultParams={
    'country': 'United States',         # Can be only one country
    'states' : ['Texas', 'Washington'],               # Can enter either one or multiple state
    'counties' : ['Bexar County', 'Dallas County', 'King County']    # Can enter multiple or one county. If all counties are required, fill in 'all'
}

class data_retriever():

    def __init__(self, country, states=None, counties = None ):
        self.states = states
        self.country = country
        self.counties = counties

    # Retrieves the mobility data for the respective counties or states or country
    def get_mobility_data(self):

        # Retrieve mobility data from Google's mobility reports.
        df = pd.read_csv(urllib.request.urlopen("https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"), low_memory=False)
        # Lambda function to filter the required data from the global mobility data.
        filtering_func = lambda x,y: x.where(x['sub_region_2'].isin(y) == True).dropna().reset_index() if y== self.counties else \
                        x.where(x['sub_region_2'].isin(y) == True).dropna().reset_index()
        # Check if data is required for only the country
        if self.country is not None and self.states is None:
            df_country = df[df['country_region']==self.country].dropna().reset_index()
            # If want all the county data also
            if (self.counties is not None or 'all' not in self.counties):
                df_required = filtering_func(df_country, self.counties)
            else:
                df_required = df_country.reset_index()
            return df_required

        else:
            # Get the state mobility data
            df_state = df.where(df['sub_region_1'].isin(self.states)==True).dropna(how='all').fillna('').reset_index()
            # The state total mobility
            if (self.counties is None):
                df_required = df_state[df_state['sub_region_2']==''].reset_index()
            # All the county mobility in the state
            elif ('all' in self.counties):

                df_required = df_state[df_state['sub_region_2'] != ''].reset_index()

            # Mobility data for given counties
            else:
                df_required = filtering_func(df_state, self.counties)
                #df_required = df_state.where(df_state['sub_region_2'].isin(self.counties)==True).dropna(how='all').reset_index()

            return df_required

    #Get the lookup table for getting population data

    @staticmethod
    def get_lookup_table():
        states = "Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada New_Hampshire New_Jersey New_Mexico New_York North_Carolina North_Dakota Ohio Oklahoma Oregon Pennsylvania Rhode_Island South_Carolina South_Dakota Tennessee Texas Utah Vermont Virginia Washington West_Virginia Wisconsin Wyoming"
        states_list = states.split(" ")
        keys = "01 02 04 05 06 08 09 10 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56"
        key_list = keys.split(" ")
        LUT ={}
        i = 0
        for states in states_list:
            LUT[states] = key_list[i]
            i+=1
        return LUT

    # Filter the required population data
    def get_population_data(self, df_required):
        LUT_dict = self.get_lookup_table()
        state_list = df_required['sub_region_1'].unique().tolist()
        print (state_list)
        base_path = ["https://www2.census.gov/programs-surveys/popest/tables/2010-2019/counties/totals/co-est2019-annres-{}.xlsx".format(LUT_dict[state]) for state in state_list]

        i = 0
        final_pop_df = pd.DataFrame()
        for paths in base_path:
            pop_df = pd.read_excel(urllib.request.urlopen(paths), skiprows = 2, skipfooter=5)
            pop_df = pop_df[['Geographic Area', 'Unnamed: 12']].iloc[1:].reset_index()
            Area_list = pop_df['Geographic Area']
            area_list = [i.split(',')[0].replace('.', '') for i in Area_list]
            pop_df['Geographic Area'] = area_list

            if (self.counties is not None):
                pop_df = pop_df.where(pop_df['Geographic Area'].isin(df_required[df_required['sub_region_1']==state_list[i]]\
                    ['sub_region_2'].unique())==True).dropna(how='all').reset_index()
                state_arr = [state_list[i]]*len(pop_df['Geographic Area'].tolist())
                pop_df['State'] = state_arr
            else:
                pop_df = pop_df.where(pop_df['Geographic Area']==state_list[i]).dropna(how='all').reset_index()


            final_pop_df = final_pop_df.append(pop_df)
            i+=1

        return final_pop_df



def get_data(paramdict):
    data = data_retriever(country=paramdict['country'], states = paramdict['states'], counties = paramdict['counties'])
    df_required = data.get_mobility_data()

    pop_df = data.get_population_data(df_required)

    pop_df = pop_df.reset_index(drop=True)
    print (pop_df, df_required)
    # Uncomment to save as csvs
    # pop_df.to_csv("formatted_population.csv")
    # df_required.to_csv("formatted_mobility.csv")

@click.command()
@click.option('--country', default = defaultParams['country'])
@click.option('--states', default = defaultParams['states'])
@click.option('--counties', default = defaultParams['counties'])
def main(country, states, counties):
    get_data(dict(click.get_current_context().params))

if __name__=="__main__":
    main()




