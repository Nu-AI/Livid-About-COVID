import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'zach_data')

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

# df_mob = pd.read_csv('top5_countries_mobility.csv')
df_mob = pd.read_csv('data/Mobility data/World_mobility.csv')
df_cas = pd.read_csv('time-series-19-covid-combined.csv')

df_mob.loc[:, 'date'] = pd.to_datetime(df_mob['date'])
df_cas.rename(columns={'Date': 'date'}, inplace=True)
df_cas.loc[:, 'date'] = pd.to_datetime(df_cas['date'])

df_cas.loc[:, 'Province/State'].fillna('total', inplace=True)
df_mob.loc[:, 'region'].fillna('total', inplace=True)

my_code_is_dumb = False
for (cty, prs), df_cty in df_cas.groupby(['Country/Region', 'Province/State']):
    if prs != 'total':
        continue
    df_cty = df_cty.loc[df_cty['Confirmed'] >= 10].copy()
    if df_cty.empty:
        print('Not enough cases in', cty)
        continue

    df_cty['Active'] = df_cty['Confirmed'] - df_cty['Recovered'] - df_cty[
        'Deaths']
    df_cty.reset_index(inplace=True, drop=True)

    dt_beg = pd.to_datetime(df_cty.iloc[0]['date'])
    dt_end = pd.to_datetime(df_cty.iloc[-1]['date'])

    df_cty_mob = df_mob.loc[df_mob['country'] == cty]
    df_cty_mob = df_cty_mob.loc[(dt_beg <= df_cty_mob['date']) &
                                (df_cty_mob['date'] <= dt_end)]
    df_cty = df_cty.loc[(df_cty_mob['date'].min() <= df_cty['date']) &
                        (df_cty['date'] <= df_cty_mob['date'].max())]

    if df_cty.empty:
        print('My code is too dumb to work with {} now'.format(cty))
        continue

    df_cty.date = df_cty.date.astype(str)
    for (mob_cat, mob_reg), df_mob_cat in df_cty_mob.groupby(
            ['category', 'region']):
        if mob_reg != 'total':
            continue
        if len(df_cty) != len(df_mob_cat):
            print('My code is too dumb to work with {} now'.format(cty))
            my_code_is_dumb = True
            continue
        df_cty[mob_cat] = df_mob_cat['value'].values
        df_cty.dropna(inplace=True, how='all')
    if my_code_is_dumb:
        my_code_is_dumb = False  # not a true statement
        continue
    df_cty.reset_index(inplace=True, drop=True)
    try:
        df_cty = df_cty[['retail/recreation', 'grocery/pharmacy', 'parks',
                         'transit_stations', 'workplace', 'residential',
                         'Confirmed', 'Recovered', 'Deaths', 'Active']]
    except KeyError as e:
        print('Missing mobility values in', cty, ':', e)
        continue
    df_cty.rename(columns={
        'retail/recreation': 'Retail & recreation',
        'grocery/pharmacy': 'Grocery & pharmacy',
        'parks': 'Parks',
        'transit_stations': 'Transit stations',
        'workplace': 'Workplace',
        'residential': 'Residential',
        'Confirmed': 'Total'
    }, inplace=True)

    df_cty.to_csv(os.path.join(DATA_DIR, cty + '_data.csv'), index=False)
