import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Defining constants
MCD_DIR = BASE_DIR / 'data' / 'raw' / 'mcd'
RAW_DIR = BASE_DIR / 'data' / 'raw'
NIGHT_HOURS = {18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5}
CONV_THRESHOLD = 5.0      # m/s, an hour counts as "convective" above this

# Load the REMS outputs 
daily_df = pd.read_csv(RAW_DIR / 'gale_late_winter_daily.csv')
indicators_df = pd.read_csv(RAW_DIR / 'gale_late_winter_indicators.csv')

# Load the three MCD files and match each to its REMS sol by solar longitude
mcd_files = sorted(MCD_DIR.glob('mcd_gale_ls*.csv'))

daily_rows, indicator_rows = [], []
for f in mcd_files:
    m = pd.read_csv(f)
    ls = m['LS'].iloc[0]
    rems = daily_df.iloc[(daily_df['LS'] - ls).abs().argmin()]
    night = m[m['LOCAL_HOUR'].isin(NIGHT_HOURS)]
    day = m[~m['LOCAL_HOUR'].isin(NIGHT_HOURS)]
    values = {
        'MCD_WIND_NIGHT_MEAN': (night['WIND_MS'].mean(), 'm/s'),
        'MCD_WIND_DAY_MEAN': (day['WIND_MS'].mean(), 'm/s'),
        'MCD_WIND_MAX': (m['WIND_MS'].max(), 'm/s'),
        'MCD_WIND_MAX_HOUR': (int(m.loc[m['WIND_MS'].idxmax(), 'LOCAL_HOUR']), 'LMST h'),
        'MCD_WIND_MIN': (m['WIND_MS'].min(), 'm/s'),
        'MCD_WIND_MIN_HOUR': (int(m.loc[m['WIND_MS'].idxmin(), 'LOCAL_HOUR']), 'LMST h'),
        'MCD_CONV_UP_MAX': (m['CONV_UP_MS'].max(), 'm/s'),
        'MCD_CONV_HOURS': (int((m['CONV_UP_MS'] > CONV_THRESHOLD).sum()), 'h per sol'),
        'MCD_DUST_TAU_MEAN': (m['DUST_TAU'].mean(), 'optical depth'),
        'MCD_RHO_MEAN': (m['RHO_KGM3'].mean(), 'kg/m3'),
        'MCD_RHO_MIN': (m['RHO_KGM3'].min(), 'kg/m3'),
    }
    daily_rows.append({'SOL': rems['SOL'], **{k: v[0] for k, v in values.items()}})
    for k, (v, unit) in values.items():
        indicator_rows.append({'LABEL': rems['LABEL'], 'SOL': rems['SOL'], 'LS': rems['LS'],
                               'INDICATOR': k, 'VALUE': v, 'UNIT': unit})

# Add to the daily table (one row per sol) and the indicator table (one row per sol and indicator)
daily_all_df = daily_df.merge(pd.DataFrame(daily_rows), on='SOL', how='left')
indicators_all_df = pd.concat([indicators_df, pd.DataFrame(indicator_rows)], ignore_index=True)

# Summarise + save
print(daily_all_df.filter(regex='^(SOL|MCD_)').round(4).to_string(index=False))
daily_all_df.to_csv(RAW_DIR / 'gale_late_winter_daily_part2.csv', index=False)
indicators_all_df.to_csv(RAW_DIR / 'gale_late_winter_indicators_part2.csv', index=False)
print('MCD indicators added to gale_late_winter and saved')