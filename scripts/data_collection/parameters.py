
global params

params = {
    'country': 'United States',         # Can be only one country
    'states' : ['Texas'],               # Can enter either one or multiple states
    'counties' : ['all'] # Can enter multiple or one county. If all counties are required, fill in ['all']
}

COUNTY_CASE_DATA_SOURCE  = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
MOBILITY_DATA_SOURCE     = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
CENSUS_DATA_SOURCE       = "https://www2.census.gov/programs-surveys/popest/tables/2010-2019/counties/totals/co-est2019-annres"
INTERVENTION_DATA_SOURCE = "https://github.com/Keystone-Strategy/covid19-intervention-data/raw/master/complete_npis_inherited_policies.csv"
STATE_CASE_DATA_SOURCE   = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
COUNTRY_DATA_SOURCE      = "https://raw.githubusercontent.com/rs-delve/covid19_datasets/master/dataset/combined_dataset_latest.csv"