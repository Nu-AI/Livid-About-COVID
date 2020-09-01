import os
import sys
import pandas as pd
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(ROOT_DIR)
from SIRNet.data_collection import retrieve_data
import parameters as params
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)
# Dict to define the requirement in the data
paramdict = {}
paramdict['country'] = ['United States']
paramdict['states'] = ['Texas']
paramdict['counties'] = None
print(paramdict['country'])
# Retrieve the data and save to the csv
df = retrieve_data.conflate_data(paramdict, verbose=0)
print (len(df[df.Cases>0]), len(df))
# df.to_csv("formatted_all_data.csv")
print (df.tail(10))
print("Finished generating data ...")
SCRIPT_DIR = os.path.abspath(os.path.join(ROOT_DIR,'scripts'))
sys.path.append(SCRIPT_DIR)
import fit_bexar_mask

actives, totals = fit_bexar_mask.pipeline(
			params, data=df,
			county=paramdict['counties'], country=paramdict['country'][0], state=paramdict['states'][0])

