params = {
    # Can enter multiple countries
    'country': ['United States'],
    # Can enter either one or multiple states (please specify United States
    # also)
    'states': ['Texas'],
    # Can enter one or more counties - If all counties are required, fill in
    # ['all']
    'counties': None
}


def update_params(paramdict):
    params.update(paramdict)


COUNTY_CASE_DATA_SOURCE = (
    'https://raw.githubusercontent.com/'
    'nytimes/covid-19-data/master/'
    'us-counties.csv'
)
MOBILITY_DATA_SOURCE = (
    'https://www.gstatic.com/'
    'covid19/mobility/'
    'Global_Mobility_Report.csv'
)
CENSUS_DATA_SOURCE_TEMPLATE = (
    'https://www2.census.gov/'
    'programs-surveys/popest/tables/2010-2019/counties/totals/'
    'co-est2019-annres-{state}.xlsx'
)
INTERVENTION_DATA_SOURCE = (
    'https://github.com/'
    'Keystone-Strategy/covid19-intervention-data/raw/master/'
    'complete_npis_inherited_policies.csv'
)
STATE_CASE_DATA_SOURCE = (
    'https://raw.githubusercontent.com/'
    'nytimes/covid-19-data/master/'
    'us-states.csv'
)
COUNTRY_DATA_SOURCE = (
    'https://raw.githubusercontent.com/'
    'rs-delve/covid19_datasets/master/dataset/'
    'combined_dataset_latest.csv'
)
TESTING_STATE_DATA_SOURCE = (
    'https://covidtracking.com/'
    'api/v1/states/'
    'daily.csv'
)
TESTING_COUNTRY_DATA_SOURCE = (
    'https://covid.ourworldindata.org/'
    'data/'
    'owid-covid-data.csv'
)

# The LUT for extracting the census data
STATE_LUT = {
    'Alabama': '01',
    'Alaska': '02',
    'Arizona': '04',
    'Arkansas': '05',
    'California': '06',
    'Colorado': '08',
    'Connecticut': '09',
    'Delaware': '10',
    'Florida': '12',
    'Georgia': '13',
    'Hawaii': '15',
    'Idaho': '16',
    'Illinois': '17',
    'Indiana': '18',
    'Iowa': '19',
    'Kansas': '20',
    'Kentucky': '21',
    'Louisiana': '22',
    'Maine': '23',
    'Maryland': '24',
    'Massachusetts': '25',
    'Michigan': '26',
    'Minnesota': '27',
    'Mississippi': '28',
    'Missouri': '29',
    'Montana': '30',
    'Nebraska': '31',
    'Nevada': '32',
    'New_Hampshire': '33',
    'New_Jersey': '34',
    'New_Mexico': '35',
    'New_York': '36',
    'North_Carolina': '37',
    'North_Dakota': '38',
    'Ohio': '39',
    'Oklahoma': '40',
    'Oregon': '41',
    'Pennsylvania': '42',
    'Rhode_Island': '44',
    'South_Carolina': '45',
    'South_Dakota': '46',
    'Tennessee': '47',
    'Texas': '48',
    'Utah': '49',
    'Vermont': '50',
    'Virginia': '51',
    'Washington': '53',
    'West_Virginia': '54',
    'Wisconsin': '55',
    'Wyoming': '56'
}

NAME_LUT = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District Of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}
