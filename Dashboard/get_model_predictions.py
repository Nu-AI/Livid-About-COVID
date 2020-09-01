import os
from os import path
import sys


import pandas as pd
import datetime as dt

basepath = os.path.join(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(basepath, '..'))
SCRIPT_DIR = os.path.abspath(os.path.join(ROOT_DIR,'scripts'))
sys.path.append(SCRIPT_DIR)

DASH_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Dashboard'))
PREDS_DIR = os.path.join(DASH_DIR,'model_predictions')
if not os.path.exists(PREDS_DIR):
	os.mkdir(PREDS_DIR)

import parameters as param


import fit_bexar_mask
filepath = path.abspath(path.join(basepath, 'GEOJSONs'))

# Get the data from the data collection module
df = pd.read_csv('formatted_all_data.csv', dtype={'fips': str})
print (os.path.dirname(__file__))
print (df.keys())
print (df.shape)
county_list = df.County.unique().tolist()
print (county_list)
prediction_dict = {}
for county in county_list:
	county_name = county.split(' ')[0]
	try:
		actives, totals = fit_bexar_mask.pipeline(
			param, data=df[df['County'] == county].reset_index(),
			county=county)
		prediction_dict[county_name] = {}
		prediction_dict[county_name]['active'] = actives
		prediction_dict[county_name]['total'] = totals
	except RuntimeError:
		pass


pred_df = pd.DataFrame.from_dict(prediction_dict)
timestamp = dt.datetime.now().strftime('%Y_%m_%d_%H_%M')

pred_df.to_csv(os.path.join(PREDS_DIR,'model_predictions_{}.csv'.format(timestamp)))

print ("Done", prediction_dict.keys())
