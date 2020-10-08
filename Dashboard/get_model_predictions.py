import os
from os import path
import sys
import traceback
import shutil
import tempfile
import glob

import pandas as pd
import datetime as dt

basepath = os.path.join(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(basepath, '..'))
SCRIPT_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'scripts'))
sys.path.append(SCRIPT_DIR)
sys.path.append(ROOT_DIR)
DASH_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Dashboard'))
PREDS_DIR = os.path.join(DASH_DIR, 'model_predictions')
if not os.path.exists(PREDS_DIR):
    os.mkdir(PREDS_DIR)

import parameters as param
import forecast

filepath = path.abspath(path.join(basepath, 'GEOJSONs'))

# Get the data from the data collection module
df = pd.read_csv('formatted_all_data.csv', dtype={'fips': str})
print(os.path.dirname(__file__))
print(df.keys())
print(df.shape)
county_list = df.County.unique().tolist()
print(county_list)
prediction_dict = {}
successful = 0
tmp_dirname = os.path.dirname(param.weights_dir)
with tempfile.TemporaryDirectory(dir=tmp_dirname) as tmp_dir:
    final_weights_dir = param.weights_dir
    # store weights here, copy to final destination upon successful training
    param.weights_dir = tmp_dir
    for county in county_list:
        county_name = county.split(' ')[0]
        try:
            actives, totals = forecast.pipeline(
                param, data=df[df['County'] == county].reset_index(),
                county=county)
            prediction_dict[county_name] = {}
            prediction_dict[county_name]['active'] = actives
            prediction_dict[county_name]['total'] = totals
            for keys in actives.keys():
                date_list = actives[keys]['date']
                new_list = list(map(lambda x: x.strftime('%d-%b %Y'), date_list))
                actives[keys]['date'] = new_list
                totals[keys]['date'] = new_list
            # No errors, great
            successful += 1
            # Move over all the weights
            for src_file in glob.iglob(os.path.join(tmp_dir, '*.pt')):
                dst_file = os.path.join(final_weights_dir,
                                        os.path.basename(src_file))
                shutil.move(src_file, dst_file)
        except RuntimeError:
            traceback.print_exc()

print('%d / %d counties succeeded in training (%.2f%%)\n' %
      (successful, len(county_list), successful / len(county_list) * 100))

pred_df = pd.DataFrame.from_dict(prediction_dict)
timestamp = dt.datetime.now().strftime('%Y_%m_%d_%H_%M')

pred_df.to_json(os.path.join(PREDS_DIR, 'model_predictions_{}.json'.format(timestamp)))
print("Done", prediction_dict.keys())
