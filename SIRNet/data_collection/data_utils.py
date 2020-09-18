import urllib.request
import warnings

import numpy as np
import pandas as pd

from . import parameters as pm

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
	total_no_days = int(str(end_date - start_date).split(' ')[0])
	actual_days_list = list(pd.to_datetime(
		np.arange(1, total_no_days + 1, 1), unit='D',
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
	state_cases_df = read_csv(pm.COUNTY_CASE_DATA_SOURCE)
	# Get the end date in the cases data
	date_list = state_cases_df['date'].unique().tolist()
	date2 = pd.to_datetime(date_list[-1])
	# Get the end date in the mobility dataframe
	date1 = pd.to_datetime(df['date'][df.index[-1]])

	# Find the diff
	no_days = int((str(date2 - date1)).split(' ')[0])

	# Converting to list
	temp = list(pd.to_datetime(
		np.arange(1, no_days + 1, 1), unit='D',
		origin=pd.Timestamp(df['date'][df.index[-1]])).astype(str))
	# Resetting the index
	index = 0 - no_days
	for _ in range(no_days):
		df = df.append(pd.Series([np.NaN]), ignore_index=True)  # Append zeroes

	# Fill the empty values with the actual ones
	for i in range(no_days):
		df['date'][df.index[index]] = temp[i]
		index += 1

	# Performing the similar for the selected county and state columns
	for keys in df:
		if df[keys].nunique() == 1 and keys in keylist:
			# Extending the values of the columns (replacing na with the actual
			# values)
			df[keys] = df[keys].fillna(list(df[keys].unique())[0])

	return df


# Extend the dataframe for all the required counties
def apply_extension(df, region):
	new_df_list = []
	unique_list = df[region].unique().tolist()
	# Extending the df for all the required counties
	for county in unique_list:
		new_df = extend_required_df(df[df[region] == county])
		new_df_list.append(new_df)
	return pd.concat(new_df_list, sort=True)


def state_lookup(state):
	return pm.STATE_LUT[state]


def filter_mobility_data(mobility):
	'''
	Filtering the mobility data before sending in the network to get correct predictions.

	:param mobility: The input mobility to the network with some corrupt data.
	:return: Modified mobility with cleared out NaNs and replaced with the average mobility.
	'''
	mobility = np.where(mobility == '', -200.0, mobility)
	flat_mobility = mobility.ravel()
	for i in range(1, flat_mobility.size - 1):
		if flat_mobility[i] == -200.0:
			if flat_mobility[i + 1] == flat_mobility[i]:
				count = i
				val = flat_mobility[count]
				while val == flat_mobility[i]:
					val = flat_mobility[count + 1]
					count += 1
					if count == flat_mobility.size - 1:
						val = flat_mobility[i - 1]
						break
			else:
				val = flat_mobility[i + 1]
			flat_mobility[i] = (flat_mobility[i - 1] + val) / 2
	return flat_mobility.reshape(mobility.shape)


def reorganize_case_data(df, df_county_cases):
	'''
	Aligning the case data with the mobility data and performing the specific join
	:param df: The main conflated dataframe with mobility
	:param df_county_cases: The county cases dataframe
	:return: The updated dataframe with the organized county and data list
	'''
	# Initializing temporary and copy variables
	new_temp_df = {}
	new_county_df_list = []
	all_case_counties = []

	# Get the list of the counties provided when
	if pm.params['counties'] is not None:
		# All the counties are required
		if 'all' in pm.params['counties']:
			# Get the list of counties available in mobility and case data
			# sources
			all_case_counties = list(df_county_cases['county'].unique())
			full_counties = list(df['sub_region_2'].unique())
			all_case_counties = [i + " County" for i in all_case_counties]
			# If the county list doesn't match
			if len(all_case_counties) != len(full_counties):
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
			# Check if the case data for counties required are present in the
			# case dataframe
			if county in all_case_counties:
				temp_df = df_county_cases[
					df_county_cases['county'] == county.rsplit(' ', 1)[0]]
				county_name_list = list(temp_df['county'].values)
				new_county_name_list = []
				# Matching the county names in cases and mobility dfs
				for val in county_name_list:
					if 'County' not in val:
						new_val = val + ' County'
						new_county_name_list.append(new_val)
				temp_df['county'] = new_county_name_list

			# Fill the ones with no case data with zeros
			else:
				temp_df = df[df['sub_region_2'] == county]
				county_length = len(temp_df)
				# Create a new dictionary with the required length
				temp_df['cases'] = [0] * county_length
				temp_df['deaths'] = [0] * county_length
				temp_df['county'] = [county] * county_length
				temp_df['fips'] = [0] * county_length
				temp_df['state'] = temp_df['sub_region_1'].tolist()
				temp_df = temp_df[['fips', 'date', 'county', 'state',
				                   'cases', 'deaths']]
			# In the case of state data
			length = county_list.count(county)

		else:
			temp_df = df_county_cases[
				df_county_cases['state'] == county]
			length = len(list(df['sub_region_1'].values))
		# Extend the case list to map with the mobility and population data
		# Create a dictionary for cases and deaths in the selected region
		case_list = list(temp_df['cases'].values)
		death_list = list(temp_df['deaths'].values)
		fips_list = list(temp_df['fips'].values)
		fips_val = fips_list[0]
		for _ in range(length - len(temp_df['cases'].tolist())):
			case_list.insert(0, 0)
			death_list.insert(0, 0)
			fips_list.insert(0, fips_val)

		if len(temp_df['cases'].tolist()) < length:
			# Extend other columns in the table

			if pm.params['counties'] is not None:
				new_temp_df['county'] = df.loc[
					df['sub_region_2'] == county]['sub_region_2'].tolist()
				new_temp_df['state'] = df.loc[
					df['sub_region_2'] == county]['sub_region_1'].tolist()
				new_temp_df['date'] = df.loc[
					df['sub_region_2'] == county]['date'].tolist()

			else:
				new_temp_df['state'] = df.loc[
					df['sub_region_1'] == county]['sub_region_1'].tolist()
				new_temp_df['date'] = df.loc[
					df['sub_region_1'] == county]['date'].tolist()
			# Fill in the dictionary
			new_temp_df['fips'] = fips_list

			new_temp_df['cases'] = case_list
			new_temp_df['deaths'] = death_list

			new_county_df = pd.DataFrame.from_dict(new_temp_df)

			# Append the dataframes
			new_county_df_list.append(new_county_df)
		else:
			new_county_df_list.append(temp_df)

	return pd.concat(new_county_df_list, sort=True)
