import urllib.request
from urllib.error import HTTPError

import numpy as np
import pandas as pd

from . import data_utils
from . import parameters as pm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)
np.set_printoptions(threshold=np.inf)


def _filtering_func(x, y):
	# Function to filter the required data from the global mobility data.
	z = x.where(x['sub_region_2'].isin(y))
	z.dropna(inplace=True)
	z.reset_index(inplace=True)
	return z


# Retrieves the mobility data for the respective counties or states or country
def get_mobility_data():
	# Retrieve mobility data from Google's mobility reports.
	df = data_utils.read_csv(pm.MOBILITY_DATA_SOURCE)

	# Check if data is required for only the country
	if pm.params['country'] is not None and pm.params['states'] is None:
		df_country = df[df['country_region'].isin(pm.params['country'])]
		df_country.dropna(how='all', inplace=True)
		df_country.reset_index(inplace=True)
		# If want all the county data also
		if (pm.params['counties'] is not None and
				'all' not in pm.params['counties']):
			df = _filtering_func(df_country, pm.params['counties'])
			df = data_utils.apply_extension(df, 'sub_region_2')
		else:
			df = df_country
	else:
		# Get the state mobility data
		df_state = df.where(df['sub_region_1'].isin(pm.params['states']))
		df_state.dropna(how='all', inplace=True)
		df_state.fillna('', inplace=True)
		df_state.reset_index(inplace=True)
		# The state total mobility
		if pm.params['counties'] is None:
			df = df_state[df_state['sub_region_2'] == '']
			df.reset_index(inplace=True)
			df = data_utils.apply_extension(df, 'sub_region_1')
		# All the county mobility in the state
		elif 'all' in pm.params['counties']:
			df = df_state[df_state['sub_region_2'] != '']
			df.reset_index(inplace=True)
			df = data_utils.apply_extension(df, 'sub_region_2')
		# Mobility data for given counties
		else:
			df = _filtering_func(df_state, pm.params['counties'])
			df = data_utils.apply_extension(df, 'sub_region_2')

	return df


# Filter the required population data
def get_population_data(df):
	state_list = df['sub_region_1'].dropna().unique().tolist()

	new_state_list = ['_'.join(state.split(' ')) for state in state_list]

	# retrieve the population data based on the state provided
	base_path = [pm.CENSUS_DATA_SOURCE_TEMPLATE.format(
		state=data_utils.state_lookup(state)) for state in new_state_list]
	i = 0
	final_pop_df = pd.DataFrame()
	# Iterate over the given paths for the states required
	count = 0
	county = 'Geographic Area'
	for paths in base_path:
		try:
			pop_df = pd.read_excel(paths, skiprows=2, skipfooter=5)
		except HTTPError:
			pop_df = pd.read_excel(urllib.request.urlopen(paths), skiprows=2,
			                       skipfooter=5)

		pop_df = pop_df[[county, 'Unnamed: 12']].iloc[1:].reset_index()
		area_list = [i.split(',')[0].replace('.', '') for i in pop_df[county]]
		pop_df[county] = area_list

		def get_state_arr(state_list_, pop_df_):
			return [state_list_[count]] * len(pop_df_[county].tolist())

		# Filter out the data required for the counties
		if pm.params['counties'] is not None:
			pop_df = pop_df.where(
				pop_df[county].isin(df[df['sub_region_1'] == new_state_list[count]
				                       ]['sub_region_2'].unique()))
		# Just get the population for the required state
		else:
			pop_df = pop_df.where(pop_df[county] == state_list[i])
		pop_df.dropna(how='all', inplace=True)
		pop_df.reset_index(inplace=True)
		pop_df['State'] = get_state_arr(state_list, pop_df)
		print(pop_df['State'])
		count += 1
		final_pop_df = final_pop_df.append(pop_df, sort=True)
		i += 1

	return final_pop_df


# Retrieve the temporal active cases and deaths information for counties
def get_cases_data(df):
	def state_cases_update(state_cases_df_):
		return state_cases_df_[
			state_cases_df_['state'].isin(pm.params['states']) &
			state_cases_df_['date'].isin(df['date'].unique().tolist())]

	# If require county level cases data
	if pm.params['counties'] is not None:
		state_cases_df = data_utils.read_csv(pm.COUNTY_CASE_DATA_SOURCE)
		state_cases_df = state_cases_update(state_cases_df)

		# Special Case for New York City
		state_cases_df = state_cases_df.replace('New York City', 'New York')
		state_cases_df.loc[
			state_cases_df['county'] == 'New York', 'fips'] = 36061

		unique_counties = state_cases_df['county'].unique().tolist()
		county_list = []

		for county in unique_counties:
			county_df = state_cases_df[state_cases_df['county'] == county]
			county_name = county + ' County'
			required_cdf = df[df['sub_region_2'] == county_name]
			county_df = county_df[
				county_df['date'].isin(required_cdf['date'].unique().tolist())]
			county_list.append(county_df)
		state_cases_df = pd.concat(county_list, sort=True)

		# For all the counties in the state
		if 'all' in pm.params['counties']:
			county_cases_df = state_cases_df.sort_values(
				by=['county', 'date']).reset_index()
		# For selected counties in the state
		else:
			new_counties = [county.rsplit(' ', 1)[0]
			                for county in pm.params['counties']]
			county_cases_df = state_cases_df[
				state_cases_df['county'].isin(new_counties)].sort_values(
				by=['county', 'date'])

		county_cases_df = county_cases_df[
			['fips', 'date', 'county', 'state', 'cases', 'deaths']]

		return data_utils.reorganize_case_data(df, county_cases_df)

	# If require only state data
	else:
		state_cases_df = data_utils.read_csv(pm.STATE_CASE_DATA_SOURCE)
		state_cases_df = state_cases_update(state_cases_df)
		state_cases_df = state_cases_df[['fips', 'date', 'state',
		                                 'cases', 'deaths']]
		return data_utils.reorganize_case_data(df, state_cases_df.sort_values(by=['state', 'date']).reset_index())


# Get intervention setting dates for the different counties and states in USA
def get_intervention_data():
	df_data = pd.read_csv(urllib.request.urlopen(pm.INTERVENTION_DATA_SOURCE),
	                      sep=',', error_bad_lines=False)

	# Filter the dataframe based on whether it is required for the counties or
	# states
	if pm.params['counties'] is not None:
		# Separate case in the case required for all the counties
		if 'all' not in pm.params['counties']:
			new_counties = [county.rsplit(' ', 1)[0]
			                for county in pm.params['counties']]
			df_intervention = df_data[
				df_data['county'].isin(new_counties) &
				df_data['state'].isin(pm.params['states'])]
		else:
			df_intervention = df_data[
				df_data['state'].isin(pm.params['states']) &
				(~df_data['county'].isnull())]

	# Keep only the state intervention dates
	else:
		df_intervention = df_data[
			df_data['state'].isin(pm.params['states']) & df_data[
				'county'].isnull()]

	return df_intervention


# Read the country level data from the RS-DELVE data source
def get_country_data():
	df = pd.read_csv(pm.COUNTRY_DATA_SOURCE, parse_dates=['DATE'])
	df2 = pd.read_csv(pm.TESTING_COUNTRY_DATA_SOURCE)

	df_country = df.loc[df['country_name'].isin(pm.params['country'])]

	df_country.reset_index(inplace=True)
	df_country.merge(df2, how='inner', left_on='ISO',
	                 right_on='iso_code')
	# Setting all the required columns
	temp = df_country.keys()
	required_keylist = list(filter(lambda x: 'mobility' in x, temp))
	npi_keylist = [x for x in temp if 'npi' in x]
	add_cols = ['cases_total', 'deaths_total', 'DATE', 'country_name',
	            'census_fips_code', 'stats_population']
	required_keylist = required_keylist + add_cols + npi_keylist
	new_df = df_country[required_keylist]
	date_vals = df_country['DATE'].dropna()
	date_vals = date_vals.apply(lambda x: x.strftime('%Y-%m-%d'))
	new_df.DATE = date_vals
	redundant_cols = np.empty(len(new_df['DATE'].values.tolist()))
	new_df['State'] = redundant_cols.fill(np.NaN)
	new_df['County'] = redundant_cols.fill(np.NaN)
	return new_df


def get_testing_state_data():
	df = pd.read_csv(pm.TESTING_STATE_DATA_SOURCE)
	required_keys = ['totalTestsViral', 'positiveTestsViral',
	                 'negativeTestsViral', 'dataQualityGrade']
	df = df[required_keys + ['date', 'state']]
	name_list = [pm.NAME_LUT[i] for i in pm.params['states']]
	df = df.loc[df.state.isin(name_list)]
	df.date = pd.to_datetime(df.date, format='%Y%m%d')
	df = df.astype({'date': 'str'})
	return df[::-1], required_keys
