import warnings

import numpy as np
import pandas as pd

from . import get_data
from . import parameters

warnings.filterwarnings('ignore')

MOBILITY_KEYS = ['Retail & recreation', 'Grocery & pharmacy', 'Parks',
                 'Transit stations', 'Workplace', 'Residential']

## Steps:
# Input will be county name and state name
# If all counties are required we take in all as the input
# Gather the mobility data
# Gather the pop data from the pop files ( create a lookup table for that first)
# Gather the county names and active cases data
# Gather the intervention data if the country selected is USA


def conflate_data(paramdict, verbose=0):
    parameters.update_params(paramdict)
    # Start with the mobility data
    if verbose:
        print("getting mobility data ..")
    df_required = get_data.get_mobility_data()
    country_flag = 1
    # The below case exists because of lack of data integration for countries
    # other than USA
    if ('United States' in paramdict['country'] and
            (paramdict['states'] is not None or
             paramdict['counties'] is not None)):
        country_flag = 0
    # Required keys in DFs
    required_keys = [
        'fips', 'Country', 'State', 'County', 'date', 'Population', 'Cases',
        'Deaths', 'Retail & recreation', 'Grocery & pharmacy', 'Parks',
        'Transit stations', 'Workplace', 'Residential'
    ]
    if country_flag == 0:
        required_keys.insert(0, 'Index')

        # Get the population data
        pop_df = get_data.get_population_data(df_required)
        pop_df.reset_index(drop=True, inplace=True)

        # Decimate non required columns
        pop_df = pop_df[['State', 'Geographic Area', 'Unnamed: 12']]

        # Rename the columns to proper distribution
        pop_df.rename(columns={
            'Geographic Area': 'County',
            'Unnamed: 12': 'Population'
        }, inplace=True)
        pop_list = pop_df['Population'].tolist()

        # Create an updated population list to account for the variability in
        # sizes of mobility data
        county_list = list(df_required['sub_region_2'].values)
        unique_list = list(df_required['sub_region_2'].unique())
        counter = [county_list.count(i) for i in unique_list]
        if counter is not None:
            pop_list = [pop_list[j] for j in range(len(counter)) for _ in
                        range(counter[j])]

        df_required['Population'] = pop_list

        # Retrieve the active cases and the total deaths
        if verbose:
            print("Getting case data for the network")
        county_cases_df = get_data.get_cases_data(df_required)

        # Add the cases and deaths to the final table
        if verbose and paramdict['counties'] is not None:
            c_list = county_cases_df['county'].unique().tolist()
            print('Unique Counties in Data:')
            print(c_list)
        df_required['Cases'] = county_cases_df['cases'].values
        df_required['Deaths'] = county_cases_df['deaths'].values
        fips_list = county_cases_df['fips'].values
        df_required['fips'] = list(map(int, fips_list))

        # Uncomment to save as csv
        # pop_df.to_csv("formatted_population.csv")

        ########################################################################
        # Add the intervention data to the required dataframe
        if paramdict['counties'] is None or 'all' not in paramdict['counties']:
            df_intervention = get_data.get_intervention_data()

            # Decimate the useless columns from the dataframe
            df_intervention = \
                df_intervention[
                    (~df_intervention['start_date'].isnull()) |
                    df_intervention['start_date'].isin([' '])][
                    ['county', 'state', 'npi', 'start_date']]

            # Select whether it is required for counties or states
            if paramdict['counties'] is None:
                id_string = 'sub_region_1'
            else:
                id_string = 'sub_region_2'
                # Update the county names to map with the main table
                county_list_i = [str(i) + ' County'
                                 for i in df_intervention['county'].values]
                df_intervention['county'] = county_list_i

            a = np.empty((len(df_required['date'])))
            df_required['Intervention'] = a.fill(np.NaN)

            # Updating the date values to map with the main table
            date_list = pd.to_datetime(df_intervention['start_date'].tolist(),
                                       infer_datetime_format=True).tolist()
            date_list = [str(i).split(' ')[0] for i in date_list]

            # Rename the columns of the intervention dataframe to sync with the
            # required table
            df_intervention['start_date'] = date_list
            df_intervention.rename(columns={
                'start_date': 'date',
                'state': 'sub_region_1',
                'county': 'sub_region_2'
            }, inplace=True)

            # Create a new dictionary to be merged with the required table
            new_date_list = df_intervention['date'].tolist()
            county_list_i2 = df_intervention[id_string].tolist()

            # Combine the state/county with the dates
            comparator_list = list(zip(county_list_i2, new_date_list))
            npi_list = df_intervention['npi'].tolist()
            # Create tuples of unique combinations
            unique_comparisons = sorted(list(set(comparator_list)))
            new_npi_list = []

            # Merge the interventions that were on the same day
            for i in range(len(unique_comparisons)):
                string = ''
                for j in range(len(comparator_list)):
                    if unique_comparisons[i] == comparator_list[j]:
                        if string:
                            string += ' & '
                        string += npi_list[j]
                new_npi_list.append(string)

            # Populate the new dictionary with the reformatted intervention data
            # and convert to dataframe
            updated_county_list = [unique_comparisons[i][0]
                                   for i in range(len(unique_comparisons))]
            updated_date_list = [unique_comparisons[i][1]
                                 for i in range(len(unique_comparisons))]
            dict_intervention = {id_string: updated_county_list,
                                 'date': updated_date_list,
                                 'Intervention': new_npi_list}
            new_df_intervention = pd.DataFrame.from_dict(dict_intervention)

            # Combine the intervention dataframe with the main required table
            df_1 = df_required.set_index(['date', id_string])
            df_2 = new_df_intervention.set_index(['date', id_string])
            df_required = df_1.combine_first(df_2).reset_index()
            df_required = df_required.sort_values(
                by=['sub_region_1', 'sub_region_2', 'date'])

        # Rename the columns of the required table
        df_required.rename(columns={
            'index': 'Index',
            'country_region': 'Country',
            'sub_region_1': 'State',
            'sub_region_2': 'County',
            'date': 'date',
            'retail_and_recreation_percent_change_from_baseline': 'Retail & recreation',
            'grocery_and_pharmacy_percent_change_from_baseline': 'Grocery & pharmacy',
            'parks_percent_change_from_baseline': 'Parks',
            'transit_stations_percent_change_from_baseline': 'Transit stations',
            'workplaces_percent_change_from_baseline': 'Workplace',
            'residential_percent_change_from_baseline': 'Residential'
        }, inplace=True)

        # Keep only the useful columns in the dataframe
        if paramdict['counties'] is None or 'all' not in paramdict['counties']:
            required_keys.append('Intervention')

        if paramdict['counties'] is None and paramdict['states'] is not None:
            df_required = df_required.astype({'date': 'str'})
            state_test_df, keys_added = get_data.get_testing_state_data()
            df_required = df_required.merge(state_test_df, left_on='date',
                                            right_on='date', how='left')
            required_keys.extend(keys_added)
    else:
        # In the case it is not United states, then load from a new data source
        df_required = get_data.get_country_data()
        df_required.rename(columns={
            'index': 'Index',
            'cases_total': 'Cases',
            'census_fips_code': 'fips',
            'stats_population': 'Population',
            'deaths_total': 'Deaths',
            'country_name': 'Country',
            'DATE': 'date',
            'mobility_retail_recreation': 'Retail & recreation',
            'mobility_grocery_pharmacy': 'Grocery & pharmacy',
            'mobility_parks': 'Parks',
            'mobility_transit_stations': 'Transit stations',
            'mobility_workplaces': 'Workplace',
            'mobility_residential': 'Residential'
        }, inplace=True)
        npi_list = [x for x in df_required.keys() if 'npi' in x]
        required_keys += npi_list

    length_list = [len(value) if value is not None else 0 for value in paramdict.values()]
    if max(length_list) == 1 and 'all' not in paramdict['counties']:
        df_required = df_required.drop_duplicates(subset=['date'], keep='last')
    df_required = df_required[required_keys].reset_index()
    if verbose:
        print(df_required.tail(20))
        df_required.to_csv("formatted_all_data.csv")
    return df_required

