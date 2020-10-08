# Check the `forecast` file to understand the role of these parameters.
# These are the mappings of the params used in the argparse interface.
import os.path as _p

weights_dir = _p.realpath(_p.join(_p.dirname(__file__), 'model_weights'))
del _p
country = 'United States'
state = 'Texas'
county = 'Bexar County'
tensorboard = 0
forecast_days = 200
reporting_rates = [0.05, 0.1, 0.3]
mobility_cases = [25, 50, 75, 100]
n_epochs = 15
b_model = 'linear'
lr_step_size = 4000
delay_days = 10
start_model = 23
incubation_days = 5
estimated_r0 = 2.2
mask_modifier = 1
mask_day = 65
train = 0
no_plot = 1
