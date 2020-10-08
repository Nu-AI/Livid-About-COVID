[![DOI](https://zenodo.org/badge/248896172.svg)](https://zenodo.org/badge/latestdoi/248896172)

# Livid About COVID
COVID-19 has impacted the State of Texas and the world in unprecedented ways. The goal of
our work is to help the City of San Antonio and the State of Texas in understanding the spread 
of the COVID-19 pandemic. We have designed a generic AI model for forecasting the spread of the
COVID-19 pandemic across the world. The region-specific models are designed to help local
government and the public better plan during the crisis, so as to facilitate a speedy recovery.
Because population dynamics differ between metropolitan areas, region-specific models can offer
additional insights. We present some preliminary results on forecasting the spread of the
COVID-19 disease by the SARS-CoV-2 virus on various scales.

You can view the live dashboard of this work with case forecasts for Texas & mobility data 
visualization on [our site here](https://livid-about-covid19.nuai.utsa.edu/).

*If you would like to use this work*, please reference this GitHub repository by the Zenodo
citation in the top of this README and the preprint reference below.

_N Soures, D Chambers, Z Carmichael, A Daram, DP Shah, K Clark, L Potter, and D Kudithipudi.
"SIRNet: Understanding Social Distancing Measures with Hybrid Neural Network Model for COVID-19
Infectious Spread." Proceedings of the International Joint Conference on Artificial Intelligence,
(IJCAI), Disease Computational Modeling Workshop (2020)_ -
[arxiv.org/abs/2004.10376](https://arxiv.org/abs/2004.10376)

<img src="images/tx_case_counts_updated_2.png" width="750px" />

This figure illustrates the trajectories for the total number of active cases in different
counties in Texas, where _Bexar County cases are doubling every ~2-3 days_. It is critical
to flatten these curves by continuing and reinforcing the social distancing measures.
The stars in the plot indicate the specific days when social distancing was implemented 
in the respective county. Data sources: CDC, European CDC, NYTimes, and Texas DSHS.

## Installation and Running
The code in this repository was developed for Python 3.5 and above. To install dependency
packages, run the following. 

```bash
pip install -r requirements.txt
```

Change the `tensorflow-x` line in `requirements.txt` depending on whether you will be running on
CPU or GPU.

For the most up-to-date forecasting employing SIRNet, run the script `scripts/forecast.py` after
installation. Parameters within can be modified for different counties, states, countries, etc.
Stay tuned for a more generalized and unified script-based interface.

Example usage:

```bash
./forecast.py --country "United States" --state "Texas" --county "Bexar County"
```

For help on all command line options:

```bash
./forecast.py --help
```

## Forecasting the Reach of the COVID-19 Disease
We use the worldwide daily case, mobility, hospital, population, and other data factors
to forecast the number of cases. One model is a hybrid Susceptible-Infected-Recovered (SIR)
& deep learning architecture and the other is a custom Long Short-Term Memory (LSTM)
architecture. Our code can be run from the scripts in the `scripts` directory. Some code is
available in the `notebooks` directory.

## Types of Data
1. Confirmed Cases
2. Deaths
3. Recoveries
    1. By country
    2. By province/state (for few)
    3. By county (for US)
