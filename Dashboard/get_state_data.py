import os
import sys

import pandas as pd

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(ROOT_DIR)
from SIRNet.data_collection import retrieve_data

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)
# Dict to define the requirement in the data
paramdict = {}
paramdict['country'] = ['United States']
paramdict['states'] = ['Texas']
paramdict['counties'] = ['Bexar County']
print(paramdict['country'])
# Retrieve the data and save to the csv
df = retrieve_data.conflate_data(paramdict, verbose=1)
print(len(df[df.Cases > 0]), len(df))
# df.to_csv("formatted_all_data.csv")
print(df.tail(10))
print("Finished generating data ...")
