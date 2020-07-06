import pandas as pd
import urllib.request
import numpy as np
import warnings

import parameters as pm
warnings.filterwarnings("ignore")


def read_csv(path):
    return pd.read_csv(urllib.request.urlopen(path), low_memory=False)

# Make the dates consistent and filling in the missing days
def fill_missing_days(df):
    # The final date in the df
    end_date = pd.to_datetime(df['date'][df.index[-1]])

    # Fill in the missing days in the mobility data
    start_date = pd.to_datetime((df['date'][df.index[0]]))

    # Create a string day list
    total_no_days = int(str(end_date - start_date).split(" ")[0])
    actual_days_list = list(pd.to_datetime(np.arange(1, total_no_days + 1, 1), unit='D',
                                           origin=pd.Timestamp(df['date'][df.index[0]])).astype(str))
    # Get the actual dates in the list
    days_list = list(df['date'].values)

    # Temporary variables
    counter = 0
    counter2 = 0

    # Create a new dataframe with similar columns as the original
    new_df = pd.DataFrame(columns=df.columns)
    new_df['date'] = actual_days_list

    # Populate the dataframe and fill the other columns with NaNs
    for day in actual_days_list:
        if day in days_list:
            new_df.iloc[counter] = df.iloc[counter]
            counter += 1
        else:
            new_df.iloc[counter + counter2] = pd.Series(np.NaN)
            counter2 += 1

    return new_df


# Extending the dataframe to match the length with multiple sources
def extend_required_df(df):

    keylist = ['country_region', 'sub_region_1', 'sub_region_2']
    state_cases_df = read_csv(
        "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv")
    # Get the end date in the cases data
    date_list = state_cases_df['date'].unique().tolist()
    date2 = pd.to_datetime(date_list[-1])

    # Get the end date in the mobility dataframe
    date1 = pd.to_datetime(df['date'][df.index[-1]])

    # Find the diff
    no_days = int((str(date2-date1)).split(" ")[0])

    # Converting to list
    temp = list(pd.to_datetime(np.arange(1, no_days + 1, 1), unit='D',
                                    origin=pd.Timestamp(df['date'][df.index[-1]])).astype(str))
    # Resetting the index
    index = 0-no_days
    for i in range(no_days):
        df = df.append(pd.Series([np.NaN]), ignore_index=True)  # Appending zeros

    # Fill the empty values with the actual ones
    for i in range(no_days):
        df['date'][df.index[index]] = temp[i]
        index += 1

    # Performing the similar for the selected county and state columns
    for keys in df:
        if df[keys].nunique() == 1 and keys in keylist:
            # Extending the values of the columns ( replacing na with the actual values )
            df[keys] = df[keys].fillna(list(df[keys].unique())[0])

    print (len(list(df['date'].values)))

    return df

# Extend the dataframe for all the required counties
def apply_extension(df, region):
    new_df_list = []
    unique_list = df[region].unique().tolist()
    # Extending the df for all the required counties
    for county in unique_list:
        print (county)
        df = extend_required_df(df[df[region]==county])
        new_df_list.append(df)
    return pd.concat(new_df_list, sort=True)


# Seting up the LUT for extracting the census data
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


def reorganize_case_data(df, df_county_cases):
    # Intializing temporary and copy variables
    new_temp_df = {}
    new_county_df_list = []
    all_case_counties = []

    # Get the list of the counties provided when
    if (pm.params['counties'] is not None):

        # All the counties are required
        if 'all' in pm.params['counties']:
            # Getting the list of counties available in mobiliy and case data sources
            all_case_counties = list(df_county_cases['county'].unique())
            full_counties = list(df['sub_region_2'].unique())
            all_case_counties = [i + " County" for i in all_case_counties]
            # If the county list doesn't match
            if (len(all_case_counties) != len(full_counties)):
                new_counties = full_counties
            else:
                new_counties = all_case_counties
        # Specifically provided
        else:
            all_case_counties = pm.params['counties']
            new_counties = all_case_counties
    # Only for the state as a total
    else:
        new_counties = pm.params['states']

    # Obtain the case data for the counties or states
    county_list = list(df['sub_region_2'].values)
    for county in new_counties:
        if pm.params['counties'] is not None:
            # Check if the case data for counties required are present in the case dataframe
            if (county in all_case_counties):
                temp_df = df_county_cases[df_county_cases['county']==' '.join(county.split(" ")[:-1])]
                county_name_list = list(temp_df['county'].values)
                new_county_name_list = []
                # Matching the county names in cases and mobility dfs
                for val in county_name_list:
                    if "County" not in val:
                        new_val = val+" County"
                        new_county_name_list.append(new_val)
                temp_df['county'] = new_county_name_list

            # Fill the ones with no case data with zeros
            else:
                temp_df = df[df['sub_region_2']==county]
                county_length = len(temp_df)
                # Create a new dictionary with the required length
                temp_df['cases'] = [0]*county_length
                temp_df['deaths'] = [0]*county_length
                temp_df['county'] = [county]*county_length
                temp_df['fips'] = [0]*county_length
                temp_df['state'] = temp_df['sub_region_1'].tolist()
                temp_df = temp_df[['fips','date','county','state','cases','deaths']]
        # In the case of state data
        else:
            temp_df = df_county_cases[df_county_cases['state']==county.split(" ")[0]]

        length = county_list.count(county)
        # Extend the case list to map with the mobility and population data
        # extend = lambda x,y,z: list(x[y].unique())*z
        # Create a dictionary for cases and deaths in the selected region
        #print (temp_df)
        case_list = list(temp_df['cases'].values)
        death_list = list(temp_df['deaths'].values)
        fips_list = list(temp_df['fips'].values)
        print (temp_df.keys(),"\n",fips_list)
        fips_val = fips_list[0]
        for _ in range(length-len(temp_df['cases'].tolist())):
            case_list.insert(0,0)
            death_list.insert(0,0)
            fips_list.insert(0,fips_val)

        if (len(temp_df['cases'].tolist())< length):
            #print ("Entered this condition")
            # Extend other columns in the table
            new_temp_df['state'] = df.loc[df['sub_region_2']==county]['sub_region_1'].tolist()
            if (pm.params['counties'] is not None):
                new_temp_df['county'] = df.loc[df['sub_region_2'] == county]['sub_region_2'].tolist()

            # Fill in the dictionary
            new_temp_df['fips'] = fips_list
            new_temp_df['date'] = df.loc[df['sub_region_2']==county]['date'].tolist()
            new_temp_df['cases'] = case_list
            new_temp_df['deaths'] = death_list

            new_county_df = pd.DataFrame.from_dict(new_temp_df)

            # Append the dataframes
            new_county_df_list.append(new_county_df)
        else:
            new_county_df_list.append(temp_df)

    return pd.concat(new_county_df_list,sort=True)
