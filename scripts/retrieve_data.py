import pandas as pd
import urllib.request
import click
import numpy as np

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
    'counties' : ['Bexar County', 'Travis County', 'King County']  # Can enter multiple or one county. If all counties are required, fill in 'all'
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

        # retrieve the population data based on the state provided
        base_path = ["https://www2.census.gov/programs-surveys/popest/tables/2010-2019/counties/totals/co-est2019-annres-{}.xlsx".format(LUT_dict[state]) for state in state_list]

        i = 0
        final_pop_df = pd.DataFrame()
        # Iterate over the given paths for the states required
        for paths in base_path:
            pop_df = pd.read_excel(urllib.request.urlopen(paths), skiprows = 2, skipfooter=5)
            pop_df = pop_df[['Geographic Area', 'Unnamed: 12']].iloc[1:].reset_index()
            Area_list = pop_df['Geographic Area']
            area_list = [i.split(',')[0].replace('.', '') for i in Area_list]
            pop_df['Geographic Area'] = area_list
            get_state_arr = lambda state_list, pop_df :[state_list[i]] * len(pop_df['Geographic Area'].tolist())
            # Filter out the data required for the counties
            if (self.counties is not None):
                pop_df = pop_df.where(pop_df['Geographic Area'].isin(df_required[df_required['sub_region_1']==state_list[i]]\
                    ['sub_region_2'].unique())==True).dropna(how='all').reset_index()

                pop_df['State'] = get_state_arr(state_list, pop_df)

            # Just get the population for the required state
            else:
                pop_df = pop_df.where(pop_df['Geographic Area']==state_list[i]).dropna(how='all').reset_index()
                pop_df['State'] = get_state_arr(state_list, pop_df)
            final_pop_df = final_pop_df.append(pop_df)
            i+=1

        return final_pop_df

    # Retrieve the temporal active cases and deaths information for counties
    def get_cases_data(self, df_required):

        state_cases_update = lambda state_cases_df : state_cases_df[state_cases_df['state'].isin(self.states) & (state_cases_df['date'].isin(df_required['date'].unique().tolist()))]

        # If require county level cases data
        if self.counties is not None:
            state_cases_df = pd.read_csv(urllib.request.urlopen(
                "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"))
            state_cases_df = state_cases_update(state_cases_df)

            # For all the counties in the state
            if 'all' in self.counties:
                county_cases_df = state_cases_df.sort_values(by = ['county','date']).reset_index()

            # For selected counties in the state
            else:
                new_counties = [county.split(" ")[0] for county in self.counties]
                county_cases_df = state_cases_df[state_cases_df['county'].isin(new_counties)].sort_values(by=['county','date'])

            county_cases_df=county_cases_df[['date', 'county', 'state', 'cases', 'deaths']]
            return self.reorganize_case_data(df_required,county_cases_df)

        # If require only state data
        else:
            state_cases_df = pd.read_csv(urllib.request.urlopen(
                "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"))
            state_cases_df = state_cases_update(state_cases_df)
            state_cases_df = state_cases_df[['date', 'state', 'cases', 'deaths']]
            return state_cases_df.sort_values(by=['state','date']).reset_index()


    # Reorganizing the data to match the mobility and population table
    def reorganize_case_data(self, df_required, df_county_cases):

        new_county_df = pd.DataFrame()
        new_temp_df = {}
        date_length =  len(df_required['date'].unique().tolist())
        if (self.counties is not None):
            new_counties = [county.split(" ")[0] for county in self.counties]
        elif 'all' in self.counties:
            new_counties = list(df_county_cases['county'].unique())
        else:
            new_counties = self.states

        for county in new_counties:
            if self.counties is not None:
                temp_df = df_county_cases[df_county_cases['county']==county]
            else:
                temp_df = df_county_cases[df_county_cases['state']==county]
            extend = lambda x,y: list(x[y].unique())*date_length
            case_list = list(temp_df['cases'].values)
            death_list = list(temp_df['deaths'].values)
            for _ in range(date_length-len(temp_df['cases'].tolist())):
                case_list.insert(0,0)
                death_list.insert(0,0)
            if (len(temp_df['cases'].tolist())< date_length):
                new_temp_df['state'] = extend(temp_df,'state')
                if (self.counties is not None):
                    new_temp_df['county'] = extend(temp_df, 'county')

                new_temp_df['date'] = list(df_required['date'].unique())

                new_temp_df['cases'] = case_list
                new_temp_df['deaths'] = death_list
            else:
                new_county_df = new_county_df.append(temp_df)
            new_county_df = new_county_df.append(pd.DataFrame.from_dict(new_temp_df))
        return new_county_df


    def get_intervention_data(self):

        df_data = pd.read_csv(urllib.request.urlopen(
            "https://github.com/Keystone-Strategy/covid19-intervention-data/raw/master/complete_npis_inherited_policies.csv"),sep = ',', error_bad_lines=False)
        print (df_data.keys())
        if self.counties is not None:
            if 'all' not in self.counties:
                new_counties = [county.split(" ")[0] for county in self.counties]
                df_intervention = df_data[df_data['county'].isin(new_counties) & df_data['state'].isin(self.states)]
            else:
                df_intervention = df_data[df_data['state'].isin(self.states) & df_data['county'].isnull()==False]
        else:
            df_intervention = df_data[df_data['state'].isin(self.states) & df_data['county'].isnull()==True]
        return df_intervention

def get_data(paramdict):

    data = data_retriever(country=paramdict['country'], states = paramdict['states'], counties = paramdict['counties'])
    df_required = data.get_mobility_data()

    df_intervention = data.get_intervention_data()

    df_intervention = \
    df_intervention[df_intervention['start_date'].isnull() == False | df_intervention['start_date'].isin([' '])][
        ['county', 'state', 'npi', 'start_date']].dropna()
    print (df_intervention)

    a = np.empty((len(df_required['date'])))
    df_required['Intervention'] = a.fill(np.NaN)

    date_list =  pd.to_datetime(df_intervention['start_date'].tolist(), infer_datetime_format=True).tolist()
    date_list = [str(i).split(" ")[0] for i in date_list]
    #date_list = df_intervention['start_date'].tolist()
    county_list = df_intervention['county'].tolist()
    county_list = [i+" County" for i in county_list]
    df_intervention['county']= county_list
    df_intervention['start_date'] = date_list
    df_intervention.rename(columns={
        'start_date': 'date',
        'county'    : 'sub_region_2'
    },inplace =True)
    columns = ['date', 'sub_region_2']
    # temp_df = pd.concat((df_required[columns], df_intervention[columns]), ignore_index=True)
    # temp_df = temp_df[temp_df.duplicated()]
    #
    # temp_df = pd.merge(df_required, temp_df, how='inner')
    # temp_df['Intervention'] = df_intervention['npi'].tolist()
    # print (len(date_list), len(temp_df['date'].tolist()))
    # print (temp_df[['date', 'sub_region_2', 'Intervention']])
    #
    new_date_list = df_intervention['date'].tolist()
    county_list = df_intervention['sub_region_2'].tolist()
    comparator_list = list(zip(county_list,new_date_list))
    npi_list = df_intervention['npi'].tolist()
    unique_comparisons = sorted(list(set(comparator_list)))
    new_npi_list = []

    for i in range(len(unique_comparisons)):
        string = ''
        for j in range(len(comparator_list)):
            if unique_comparisons[i]==comparator_list[j]:
                if string !='':
                    string = string + " & " +npi_list[j]
                else:
                    string = npi_list[j]
        new_npi_list.append(string)

    print (new_npi_list, "\n\n",unique_comparisons, len(new_npi_list))

    updated_county_list = [unique_comparisons[i][0] for i in range(len(unique_comparisons))]
    updated_date_list = [unique_comparisons[i][1] for i in range(len(unique_comparisons))]
    dict_intervention = {}
    dict_intervention['sub_region_2'] = updated_county_list
    dict_intervention['date'] = updated_date_list
    dict_intervention['Intervention'] = new_npi_list
    new_df_intervention = pd.DataFrame.from_dict(dict_intervention)

    # df_required.loc[(df_required['date'].isin(new_df_intervention['date']))
    #                 & (df_required['sub_region_2'].isin(new_df_intervention['sub_region_2']))
    #                 ,'Intervention'] = new_df_intervention['Intervention'].tolist()
    #df_required = pd.merge(df_required,temp_df, how='outer', on=['date','sub_region_2'])
    #df_required.loc[(df_required['date'].isin(date_list) and df_required['county'].isin(county_list)),'Intervention'] = df_intervention['npi'].tolist()
    #df_required.loc[(df_required['date'].isin(temp_df['date']) & df_required['sub_region_2'].isin(temp_df['sub_region_2'])),['Intervention']] = df_intervention['npi'].tolist()
    # df_required['Intervention'] = df_required.where(df_required['date'].isin(new_df_intervention['date']) &
    #                   df_required['sub_region_2'].isin(new_df_intervention['sub_region_2']))['Intervention']
    # x  = df_required.loc[df_required['date'].isin(new_df_intervention['date']) &
    #                   df_required['sub_region_2'].isin(new_df_intervention['sub_region_2'])]['Intervention']
    # old_in_arr = np.array(df_required['Intervention'].tolist())


    # df_required['Intervention'] = df_required.where(df_required[['date','sub_region_2']].isin(
    #                                 new_df_intervention[['date','sub_region_2']]))['Intervention']
    #df_required = pd.concat([df_required,new_df_intervention]).drop_duplicates(['date','sub_region_2']).sort_values('date')
    #df3 = df_required.combine_first(new_df_intervention).reindex(df_required.index)
    #print (df_required)
    df_1 = df_required.set_index(['date','sub_region_2'])
    df_2 = new_df_intervention.set_index(['date','sub_region_2'])
    #print (df_1, "\n\n\n", df_2)
    df_3 = df_1.combine_first(df_2).reset_index()
    df_3 = df_3.sort_values(by=['sub_region_1','sub_region_2','date'])
    print (df_3, df_3.keys())

    count = 0
    #print (df_required)
    #df_required = pd.merge(df_required, new_df_intervention, on='Intervention', how='left')
    #print(index_list)
    # for i in range(len(old_in_arr)):
    #     if old_in_arr[i]==None:
    #         print (count, i, len(new_npi_list))
    #         #old_in_arr[i] = new_npi_list[count]
    #         count+=1
    #
    # print (np.array(old_in_arr), count)
    #
    # print ( df_required.set_index(['date', 'sub_region_2']).reindex_like(temp_df.set_index(['date', 'sub_region_2'])).
    #       combine_first(temp_df.set_index(['date', 'sub_region_2'])).reset_index())


    # for i in range(len(df_required['date'])):
    #     if (df_required[])
    #     a[i] = df_intervention['npi'][count]
    #     count+=1


    # # TODO incorporate population metrics for other countries
    if paramdict['country'] == 'United States' or paramdict['country'] is None:
        pop_df = data.get_population_data(df_required)

        pop_df = pop_df.reset_index(drop=True)
        pop_df = pop_df[['State', 'Geographic Area', 'Unnamed: 12']]
        county_list = pop_df.values.tolist()
        pop_df.rename(columns={
            'State' : 'State',
            'Geographic Area' : 'County',
            'Unnamed: 12': 'Population'
        }, inplace=True)
        pop_list = pop_df['Population'].tolist()

        pop_list = [i for i in pop_list for _ in range(int(len(df_required['sub_region_2'])/len(pop_list)))]
        df_required['Population'] = pop_list

        county_cases_df = data.get_cases_data(df_required)
        df_required['Cases'] = county_cases_df['cases'].values
        df_required['Deaths'] = county_cases_df['deaths'].values
        # Uncomment to save as csvs
        # pop_df.to_csv("formatted_population.csv")


    df_required.rename(columns={
        'index'                                             : 'Index',
        'country_region'                                    : 'Country',
        'sub_region_1'                                      : 'State',
        'sub_region_2'                                      : 'County',
        'date'                                              : 'date',
        'retail_and_recreation_percent_change_from_baseline': 'Retail & recreation',
        'grocery_and_pharmacy_percent_change_from_baseline' : 'Grocery & pharmacy',
        'parks_percent_change_from_baseline'                : 'Parks',
        'transit_stations_percent_change_from_baseline'     : 'Transit stations',
        'workplaces_percent_change_from_baseline'           : 'Workplace',
        'residential_percent_change_from_baseline'          : 'Residential'}, inplace=True)

    df_required = df_required[['Index', 'Country', 'State', 'County', 'date', 'Population', 'Cases','Deaths', 'Retail & recreation',
                               'Grocery & pharmacy', 'Parks', 'Transit stations', 'Workplace', 'Residential']]

    # df_required.to_csv("formatted_all_data.csv")
    print (df_required)

@click.command()
@click.option('--country', default = defaultParams['country'])
@click.option('--states', default = defaultParams['states'])
@click.option('--counties', default = defaultParams['counties'])
def main(country, states, counties):
    get_data(dict(click.get_current_context().params))

if __name__=="__main__":
    main()




