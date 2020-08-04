import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(ROOT_DIR)
from SIRNet.data_collection import retrieve_data

# Dict to define the requirement in the data
paramdict = {}
paramdict['country'] = 'United States'
paramdict['states'] = ['Texas']
paramdict['counties'] = ['all']

# Retrieve the data and save to the csv
df = retrieve_data.conflate_data(paramdict, verbose=1)
df.to_csv("formatted_all_data.csv")

print("Finished generating data ...")
