import sys
import time
import calendar
import numpy as np
import pandas as pd
import requests
import earthaccess
import xarray as xr
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Defining constants
HOURLY_URL = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
COMMUNITY = 'AG'
MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
PARAM_LIST = ['T2M', 'T2M_MIN', 'T2M_RANGE', 'WS2M', 'PS']
MIN_INTERVAL_S = 2.5             # seconds between POWER requests
RATE_LIMIT_WAIT_S = 60           # pause when POWER answers "429 Too Many Requests"
RETRIES = 5

# Match rules for Method A: (weight, value that scores 0, value that scores 1). One-sided: harsher than Mars still scores 1.
RULES = {
    'COLD':  (25, -35.0, -65.0),    # night minimum air temperature, deg C
    'RANGE': (20, 5.0, 20.0),       # day-night temperature range, K (Earth's POWER data tops out at about 20 K)
    'WIND':  (25, 2.0, 8.0),        # mean 2 m wind speed, m/s
}
NIGHT_HOURS = {18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5}
YEARS = (2022, 2023, 2024)       # years averaged for the hourly wind and the dust climatology
SHORTLIST_ANTARCTIC = 5          # shortlist slots for Antarctic sites (LAT <= -60)
SHORTLIST_OTHER = 12             # shortlist slots for everywhere else, so the terrain step has non-Antarctic options
MIN_RANGE_K = 8.0               # gate: a month needs a real day-night cycle (rules out polar night)
MAX_NIGHT_MIN_C = RULES['COLD'][1]   # gate: a month must be cold enough to score on the cold indicator
MIN_SEPARATION_KM = 800          # candidates at least this far apart, so one region can't fill the shortlist
MERRA_SHORT_NAME, MERRA_VERSION = 'M2TMNXAER', '5.12.4'    # monthly MERRA-2 aerosol diagnostics

# Mars reference values (to flag overshoot and for the density tie-breaker)
MARS_COLD_C, MARS_RANGE_K, MARS_WIND_MS = -76.5, 66.6, 11.1
MARS_WIND_RATIO, MARS_DUST_TAU = 2.0, 0.5   # Mars night/day wind ratio (MCD) and typical dust optical depth, reported as gaps
MARS_DENSITY = 0.0163            # kg/m3, mean REMS density over the three sols
R_EARTH_AIR = 287.0              # J/kg/K

RAW_DIR = BASE_DIR / 'data' / 'raw'
CLIM_FILE = RAW_DIR / 'power_climatology_raw.csv'
HOURLY_FILE = RAW_DIR / 'power_hourly_summary.csv'

last_call = [0.0]


def polite_get(url, params):
    """One POWER request, paced to stay under the rate limit, waiting and retrying on 429."""
    for attempt in range(RETRIES):
        wait = MIN_INTERVAL_S - (time.time() - last_call[0])
        if wait > 0:
            time.sleep(wait)
        last_call[0] = time.time()
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            try:
                pause = float(r.headers.get('Retry-After', RATE_LIMIT_WAIT_S))
            except ValueError:
                pause = RATE_LIMIT_WAIT_S
            print(f'  429 Too Many Requests - waiting {pause:.0f} s')
            time.sleep(pause)
        elif r.status_code == 422:
            raise RuntimeError(f'POWER rejected the request: {r.text[:300]}')
        else:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'POWER request failed after {RETRIES} tries: {r.status_code} {r.text[:200]}')



# ---------------------------------------------------------------- STAGE 1: score the saved climate table
if not CLIM_FILE.exists():
    raise RuntimeError(f'{CLIM_FILE.name} not found - run earth_climate_download.py first')
clim_df = pd.read_csv(CLIM_FILE)
# Score every point-month, then keep each point's best month (all indicators from the same month)
valid_all = clim_df[(clim_df[PARAM_LIST] > -990).all(axis=1)].copy()
for name, col in (('COLD', 'T2M_MIN'), ('RANGE', 'T2M_RANGE'), ('WIND', 'WS2M')):
    _, zero, full = RULES[name]
    valid_all['SCORE_' + name] = ((valid_all[col] - zero) / (full - zero)).clip(0, 1)
stage1_weight = sum(r[0] for r in RULES.values())
valid_all['STAGE1_POINTS'] = sum(RULES[n][0] * valid_all['SCORE_' + n] for n in RULES)   # out of 70
valid_all['SCORE'] = valid_all['STAGE1_POINTS'] / stage1_weight

# Reference list: polar sites with no day-night cycle required (kept for comparison, not used further)
polar = valid_all[valid_all['LAT'] <= -60]
polar_df = polar.loc[polar.groupby(['LAT', 'LON'])['STAGE1_POINTS'].idxmax()].sort_values('SCORE', ascending=False)

# Main list: only months with a real day-night cycle that are cold enough to count
valid = valid_all[(valid_all['T2M_RANGE'] >= MIN_RANGE_K) & (valid_all['T2M_MIN'] <= MAX_NIGHT_MIN_C)]
print(f'{valid.groupby(["LAT", "LON"]).ngroups} of {clim_df.groupby(["LAT", "LON"]).ngroups} points have a month passing the day-night-cycle and cold gates')
best_df = valid.loc[valid.groupby(['LAT', 'LON'])['STAGE1_POINTS'].idxmax()].copy()
best_df['DENSITY_KGM3'] = best_df['PS'] * 1000 / (R_EARTH_AIR * (best_df['T2M'] + 273.15))
best_df['DENSITY_VS_MARS'] = best_df['DENSITY_KGM3'] / MARS_DENSITY
best_df['OVERSHOOT'] = [','.join(n for n, hit in (('COLD', r.T2M_MIN < MARS_COLD_C), ('RANGE', r.T2M_RANGE > MARS_RANGE_K),
                                                  ('WIND', r.WS2M > MARS_WIND_MS)) if hit) for r in best_df.itertuples()]
best_df = best_df.rename(columns={'MONTH': 'BEST_MONTH', 'T2M_MIN': 'T2M_MIN_C', 'T2M_RANGE': 'T2M_RANGE_K',
                                  'WS2M': 'WS2M_MS', 'PS': 'PS_KPA'})
stage1_df = best_df.sort_values(['SCORE', 'DENSITY_VS_MARS'], ascending=[False, True]).reset_index(drop=True)
print('\nSTAGE 1 - top 10 (cold, range, wind only):')
print(stage1_df[['LAT', 'LON', 'ELEV_M', 'BEST_MONTH', 'SCORE', 'T2M_MIN_C', 'T2M_RANGE_K', 'WS2M_MS', 'OVERSHOOT']].head(10).round(3).to_string(index=False))


# ---------------------------------------------------------------- STAGE 2: wind day-night ratio and dust, reported as gaps (not scored)
def dust_by_month(lat_values, lon_values):
    """MERRA-2 dust extinction optical depth (550 nm), monthly climatology over YEARS at each point. Returns array (12, n)."""
    earthaccess.login(persist=True)
    granules = earthaccess.search_data(short_name=MERRA_SHORT_NAME, version=MERRA_VERSION,
                                       temporal=(f'{YEARS[0]}-01-01', f'{YEARS[-1]}-12-31'))
    ds = xr.open_mfdataset(earthaccess.open(granules), engine='h5netcdf')
    da = ds['DUEXTTAU'].sel(lat=xr.DataArray(lat_values, dims='pt'), lon=xr.DataArray(lon_values, dims='pt'),
                            method='nearest').load()
    return da.groupby('time.month').mean('time').values




def pick_spread(df, n, min_km):
    """Walk down the ranking and keep a site only if it is at least min_km from every site already kept."""
    kept, rows = [], []
    for row in df.itertuples():
        la, lo = np.radians(row.LAT), np.radians(row.LON)
        far = all(2 * 6371 * np.arcsin(np.sqrt(np.sin((la - a) / 2) ** 2 + np.cos(la) * np.cos(a) * np.sin((lo - b) / 2) ** 2)) >= min_km
                  for a, b in kept)
        if far:
            kept.append((la, lo))
            rows.append(row.Index)
        if len(rows) == n:
            break
    return df.loc[rows]


# Antarctic and non-Antarctic sites are shortlisted separately, so Antarctica can't take every slot
is_ant = stage1_df['LAT'] <= -60
top_df = pd.concat([pick_spread(stage1_df[is_ant], SHORTLIST_ANTARCTIC, MIN_SEPARATION_KM),
                    pick_spread(stage1_df[~is_ant], SHORTLIST_OTHER, MIN_SEPARATION_KM)]).copy()
top_df['REGION'] = np.where(top_df['LAT'] <= -60, 'ANTARCTIC', 'OTHER')

# Hourly wind: night and day means for each candidate's best month, one row per point-month-year
hourly_df = pd.read_csv(HOURLY_FILE) if HOURLY_FILE.exists() else pd.DataFrame(columns=['LAT', 'LON', 'MONTH', 'YEAR', 'NIGHT_MEAN', 'DAY_MEAN'])
done_h = set(zip(hourly_df['LAT'], hourly_df['LON'], hourly_df['MONTH'], hourly_df['YEAR']))
print(f'\nSTAGE 2 - hourly wind for {len(top_df)} spread-out candidates x {len(YEARS)} years')
for row in top_df.itertuples():
    month = MONTHS.index(row.BEST_MONTH) + 1
    new_rows = []
    for y in YEARS:
        if (row.LAT, row.LON, row.BEST_MONTH, y) in done_h:
            continue
        last = calendar.monthrange(y, month)[1]
        data = polite_get(HOURLY_URL, {'parameters': 'WS2M', 'community': COMMUNITY, 'latitude': row.LAT, 'longitude': row.LON,
                                       'format': 'JSON', 'start': f'{y}{month:02d}01', 'end': f'{y}{month:02d}{last}',
                                       'time-standard': 'LST'})
        s = pd.Series(data['properties']['parameter']['WS2M'])
        s = s[s > -990]
        hours = np.array([int(k[-2:]) for k in s.index])
        night = np.isin(hours, list(NIGHT_HOURS))
        new_rows.append({'LAT': row.LAT, 'LON': row.LON, 'MONTH': row.BEST_MONTH, 'YEAR': y,
                         'NIGHT_MEAN': s.values[night].mean(), 'DAY_MEAN': s.values[~night].mean()})
    if new_rows:
        hourly_df = pd.concat([hourly_df, pd.DataFrame(new_rows)], ignore_index=True)
        hourly_df.to_csv(HOURLY_FILE, index=False)

ratios = []
for row in top_df.itertuples():
    h = hourly_df[(hourly_df['LAT'] == row.LAT) & (hourly_df['LON'] == row.LON) & (hourly_df['MONTH'] == row.BEST_MONTH)]
    ratios.append(h['NIGHT_MEAN'].mean() / h['DAY_MEAN'].mean())
top_df['WIND_NIGHT_DAY_RATIO'] = ratios

# Dust
dust = dust_by_month(top_df['LAT'].values, top_df['LON'].values)
top_df['DUST_TAU'] = [dust[MONTHS.index(m), i] for i, m in enumerate(top_df['BEST_MONTH'])]

# Score is stage 1 only (cold, range, wind); ratio, dust and density are gap columns
top_df['FINAL_SCORE'] = top_df['SCORE']
final_df = top_df.sort_values(['FINAL_SCORE', 'DENSITY_VS_MARS'], ascending=[False, True]).reset_index(drop=True)

# Summarise + save
cols = ['LAT', 'LON', 'ELEV_M', 'BEST_MONTH', 'FINAL_SCORE', 'T2M_MIN_C', 'T2M_RANGE_K', 'WS2M_MS', 'WIND_NIGHT_DAY_RATIO',
        'DUST_TAU', 'DENSITY_VS_MARS', 'OVERSHOOT']
for region in ('ANTARCTIC', 'OTHER'):
    print(f'\nFINAL - {region} shortlist (score = cold, range, wind; ratio, dust, density are gaps to Mars):')
    print(final_df[final_df['REGION'] == region][cols].round(3).to_string(index=False))
stage1_df.to_csv(RAW_DIR / 'earth_climate_stage1.csv', index=False)
polar_df.head(20).to_csv(RAW_DIR / 'earth_climate_polar_reference.csv', index=False)
final_df.to_csv(RAW_DIR / 'earth_climate_final.csv', index=False)
print('Earth climate search complete')