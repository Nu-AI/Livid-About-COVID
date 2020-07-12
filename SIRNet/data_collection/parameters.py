# @formatter:off
params = {
    'country': 'United States',   # Can be only one country
    'states': ['Texas'],          # Can enter either one or multiple states
    'counties': ['Bexar County']  # Can enter one or more counties - If all
                                  #   counties are required, fill in ['all']
}
# @formatter:on


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

"""
Below was generated with the following code:

def get_lookup_table():
    states = "Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada New_Hampshire New_Jersey New_Mexico New_York North_Carolina North_Dakota Ohio Oklahoma Oregon Pennsylvania Rhode_Island South_Carolina South_Dakota Tennessee Texas Utah Vermont Virginia Washington West_Virginia Wisconsin Wyoming"
    states_list = states.split(" ")
    keys = "01 02 04 05 06 08 09 10 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56"
    key_list = keys.split(" ")
    LUT = {}
    i = 0
    for states in states_list:
        LUT[states] = key_list[i]
        i += 1
    return LUT
"""
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
