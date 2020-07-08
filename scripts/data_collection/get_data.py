import numpy as np
import pandas as pd
import urllib.request

import parameters as pm
import data_utils


# Retrieves the mobility data for the respective counties or states or country
def get_mobility_data():

    # Retrieve mobility data from Google's mobility reports.
    df = data_utils.read_csv(pm.MOBILITY_DATA_SOURCE)

    # Lambda function to filter the required data from the global mobility data.
    filtering_func = lambda x,y: x.where(x['sub_region_2'].isin(y) == True).dropna().reset_index()

    # Check if data is required for only the country
    if pm.params['country'] is not None and pm.params['states']is None:
        df_country = df[df['country_region']==pm.params['country']].dropna(how='all').reset_index()
        # If want all the county data also
        if (pm.params['counties'] is not None):
            if ('all' not in pm.params['counties']):
                df = filtering_func(df_country, pm.params['counties'])
                df = data_utils.apply_extension(df,'sub_region_2')
        else:
            df = df_country.reset_index()
        
    else:
        # Get the state mobility data
        df_state = df.where(df['sub_region_1'].isin(pm.params['states'])==True).dropna(how='all').fillna('').reset_index()
        # The state total mobility
        if (pm.params['counties'] is None):
            df = df_state[df_state['sub_region_2']==''].reset_index()
            df = data_utils.apply_extension(df, 'sub_region_1')
         # All the county mobility in the state
        elif ('all' in pm.params['counties']):
            df = df_state[df_state['sub_region_2'] != ''].reset_index()
            df = data_utils.apply_extension(df, 'sub_region_2')

        # Mobility data for given counties
        else:
            df = filtering_func(df_state, pm.params['counties'])
            df = data_utils.apply_extension(df,'sub_region_2')

    return df


# Filter the required population data
def get_population_data(df):

    LUT_DICT = data_utils.get_lookup_table()
    state_list = df['sub_region_1'].dropna().unique().tolist()

    state_list = ["_".join(state.split(" ")) for state in state_list]

    # retrieve the population data based on the state provided

    base_path = ["https://www2.census.gov/programs-surveys/popest/tables/2010-2019/counties/totals/co-est2019-annres-{}.xlsx".format(LUT_DICT[state]) for state in state_list]

    state_list = [" ".join(state.split("_")) for state in state_list]

    i = 0
    final_pop_df = pd.DataFrame()
    # Iterate over the given paths for the states required
    count = 0
    county = 'Geographic Area'
    for paths in base_path:
        # Can alternate between the below 2 lines if one is not working and
        # throws a hhtprequest exception
        # pop_df = pd.read_excel(urllib.request.urlopen(paths), skiprows=2,
        #                        skipfooter=5)
        pop_df = pd.read_excel(paths, skiprows=2, skipfooter=5)
        pop_df = pop_df[[county, 'Unnamed: 12']].iloc[1:].reset_index()
        _area_list = pop_df[county]
        area_list = [i.split(',')[0].replace('.', '') for i in _area_list]
        print (area_list,state_list)
        pop_df[county] = area_list
        get_state_arr = lambda state_list, pop_df :[state_list[count]] * len(pop_df[county].tolist())
        # Filter out the data required for the counties
        if (pm.params['counties'] is not None):
            pop_df = pop_df.where(pop_df[county].isin(df[df['sub_region_1']==state_list[count]]\
                ['sub_region_2'].unique())==True).dropna(how='all').reset_index()

            pop_df['State'] = get_state_arr(state_list, pop_df)

            count +=1
        # Just get the population for the required state
        else:
            pop_df = pop_df.where(pop_df[county]==state_list[i]).dropna(how='all').reset_index()
            pop_df['State'] = get_state_arr(state_list, pop_df)
            count +=1
        final_pop_df = final_pop_df.append(pop_df,sort=True)
        i+=1

    return final_pop_df


# Retrieve the temporal active cases and deaths information for counties
def get_cases_data(df):

    state_cases_update = lambda state_cases_df : state_cases_df[state_cases_df['state'].isin(pm.params['states']) & (state_cases_df['date'].isin(df['date'].unique().tolist()))]

    # If require county level cases data
    if pm.params['counties'] is not None:
        state_cases_df = data_utils.read_csv(pm.COUNTY_CASE_DATA_SOURCE)
        state_cases_df = state_cases_update(state_cases_df)

        # Special Case for New York City
        state_cases_df = state_cases_df.replace("New York City", "New York")
        state_cases_df.loc[state_cases_df["county"]=="New York", 'fips'] = 36061

        unique_counties = state_cases_df['county'].unique().tolist()
        county_list = []

        for county in unique_counties:
            county_df = state_cases_df[state_cases_df['county']==county]
            county_name = county + " County"
            required_cdf = df[df['sub_region_2']==county_name]
            county_df = county_df[county_df['date'].isin(required_cdf['date'].unique().tolist())]
            county_list.append(county_df)
        state_cases_df = pd.concat(county_list,sort=True)

        # For all the counties in the state
        if 'all' in pm.params['counties']:
            county_cases_df = state_cases_df.sort_values(by = ['county','date']).reset_index()

        # For selected counties in the state
        else:
            new_counties = [" ".join(county.split(" ")[:-1]) for county in pm.params['counties']]
            #print (new_counties)
            county_cases_df = state_cases_df[state_cases_df['county'].isin(new_counties)].sort_values(by=['county','date'])

        county_cases_df=county_cases_df[['fips', 'date', 'county', 'state', 'cases', 'deaths']]
        print (county_cases_df['fips'].unique().tolist())

        return data_utils.reorganize_case_data(df,county_cases_df)

    # If require only state data
    else:
        state_cases_df = data_utils.read_csv(pm.STATE_CASE_DATA_SOURCE)
        state_cases_df = state_cases_update(state_cases_df)
        state_cases_df = state_cases_df[['date', 'state', 'cases', 'deaths']]
        return state_cases_df.sort_values(by=['state','date']).reset_index()


# Get the intervention setting dates for the different counties and states in USA
def get_intervention_data():

    df_data = pd.read_csv(urllib.request.urlopen(
        pm.INTERVENTION_DATA_SOURCE), sep = ',', error_bad_lines=False)

    # Filter the dataframe based on whether it is required for the counties or states
    if pm.params['counties'] is not None:
        # Separate case in the case required for all the counties
        if 'all' not in pm.params['counties']:
            new_counties = [" ".join(county.split(" ")[:-1]) for county in pm.params['counties']]
            df_intervention = df_data[df_data['county'].isin(new_counties) & df_data['state'].isin(pm.params['states'])]
        else:
            df_intervention = df_data[df_data['state'].isin(pm.params['states']) & df_data['county'].isnull()==False]

    # Keep only the state intervention dates
    else:
        df_intervention = df_data[df_data['state'].isin(pm.params['states']) & df_data['county'].isnull()==True]

    #print(df_intervention.head(100))
    return df_intervention



def get_country_data():
    df = pd.read_csv(pm.COUNTRY_DATA_SOURCE, parse_dates=['DATE'])
    df_country = df[df['country_name']==pm.params['country']].reset_index()
    temp = df_country.keys()
    required_keylist = list(filter(lambda x:x.__contains__("mobility"), temp))
    npi_keylist = [x for x in temp if x.__contains__("npi")]
    add_cols = ['cases_total', 'deaths_total', 'DATE', 'country_name','census_fips_code', 'stats_population']
    required_keylist = required_keylist + add_cols + npi_keylist
    new_df = df_country[required_keylist]
    date_vals = df_country['DATE']
    date_vals.apply(lambda x:x.strftime('%Y-%m-%d'))
    new_df.DATE = date_vals
    redundant_cols = np.empty(len(new_df['DATE'].values.tolist()))
    new_df['State'] = redundant_cols.fill(np.NaN)
    new_df['County'] = redundant_cols.fill(np.NaN)
    return new_df
