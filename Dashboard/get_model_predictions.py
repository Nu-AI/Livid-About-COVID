import numpy as np
import pandas as pd
import importlib
import retrieve_data
import torch
import datetime as dt
import sys


# sys.path.insert(1, 'SIRNet')
from SIRNet import util, trainer


########### ASSUMPTIONS ##############################
######################################################
reporting_rate = 0.1     # Portion of cases that are actually detected
delay_days = 10          # Days between becoming infected / positive confirmation (due to incubation period / testing latency
start_model = 23         # The day where we begin our fit
mask_modifier = False    #
mask_day = 65            # Day of mask order

## Cases ##
actives = {}
totals = {}

############## Simplified Data ####################
###################################################
paramdict = {}
paramdict['country'] = 'United States'
paramdict['states'] = ['Texas']
paramdict['counties'] = ['Bexar County']
df = retrieve_data.conflate_data(paramdict)

def get_model_preds(df):
    for reporting_rate in [0.05, 0.1, 0.3]:
      mobility = df[['Retail & recreation', 'Grocery & pharmacy', 'Parks', 'Transit stations', 'Workplace', 'Residential']]
      cases = df['Cases']
      day0 = df['date'][0]
      population = df['Population'][0]

      # offset case data by delay days (treat it as though it was recorded earlier)
      cases = np.array(cases[delay_days:])
      mobility = np.array(mobility[:-delay_days])
      county_names = [x.split(" ")[0] for x in paramdict['counties']]
      county_name = county_names[0]
      ###################### Formatting Data ######################
      #############################################################
      # print(mobility)
      mobility[-1][-1] = 17.0
      mobility = np.asarray(mobility).astype(np.float32)
      mobility[:, :6] = (1.0 + mobility[:, :6] / 100.0)  # convert percentages of change to fractions of activity

      # Turn the last column into 1-hot social-distancing enforcement
      mobility[:, 5] = 0

      if mask_modifier: mobility[mask_day:,5] = 1.0

      # Initial conditions
      i0 = float(cases[start_model-1]) / population / reporting_rate
      e0 = 2.2*i0/5.0
      mobility = mobility[start_model:]  # start with delay
      cases = cases[start_model:]        # delay days

      # Split into input and output data
      X, Y = mobility, cases

      # divide out population of county, reporting rate
      Y = (Y / population) / reporting_rate

      # To Torch on device
      X = torch.from_numpy(X.astype(np.float32))
      Y = torch.from_numpy(Y.astype(np.float32))

      # Add batch dimension
      X = X.reshape(X.shape[0], 1, X.shape[1])  # time x batch x channels
      Y = Y.reshape(Y.shape[0], 1, 1)  # time x batch x channels

      #################### Training #######################
      #####################################################
      # weights_name ='{}_weights.pt'.format(county_name)
      weights_name = 'Bexar_weights.pt'
      trnr = trainer.Trainer(weights_name)
      model = trnr.build_model(e0,i0)
      # trnr.train(model, X, Y, 200)

      ################ Forecasting #######################
      ####################################################
      active = {}
      total = {}
      cases = [25, 50, 75, 100]
      for case in cases:
          xN = torch.ones((1, 6), dtype=torch.float32) * case/100
          xN[0,5] = 0
          rX = xN.expand(200, *xN.shape)  # 200 x 1 x 6
          rX = torch.cat((X, rX), dim=0)
          if mask_modifier:
            rX[mask_day:,0,5] = 1.0
          sir_state, total_cases = model(rX)
          s = util.to_numpy(sir_state)
          active[case] = s[:,0] * reporting_rate * population
          total[case] = (s[:,0] + s[:,1]) * reporting_rate * population
      actives[reporting_rate] = active
      totals[reporting_rate] = total

      ############## Forecast Dates ####################
      ##################################################
      yy, mm, dd = day0.split('-')
      date0 = dt.datetime(int(yy),int(mm),int(dd))
      days = np.arange(rX.shape[0])
      dates = [date0 + dt.timedelta(days=int(d + delay_days + start_model)) for d in days]

      ############### Reporting #########################
      ###################################################
      print('\n#########################################\n\n')
      timestamp = dt.datetime.now().strftime('%Y_%m_%d')
      for case in cases:
          M = np.max(active[case])
          idx = np.argmax(active[case])
          print ('Case: {}%'.format(case))
          print ('  Max value: {}'.format(M))
          print ('  Day: {}, {}'.format(idx, dates[idx]))

    for key in totals.keys():
        totals[key]['date'] = dates
        actives[key]['date'] = dates
    print(totals.keys())

    return actives, totals


actives,totals = get_model_preds(df)
# ############### Plotting ##########################
# ###################################################
# gt = np.squeeze(Y.numpy()) * reporting_rate * population

