import sys
import os

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(ROOT_DIR)

# sys.path.insert(1,'Livid-About-COVID\SIRNet\util.py')
from SIRNet.data_collection import retrieve_data

############## Simplified Data ####################
###################################################
paramdict = {}
paramdict['country'] = 'United States'
paramdict['states'] = ['Texas']
paramdict['counties'] = ['all']

df = retrieve_data.conflate_data(paramdict, verbose=1)
df.to_csv("formatted_all_data.csv")
print("finished generating data")
