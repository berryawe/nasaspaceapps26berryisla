import sys
import json
import time
import calendar
import numpy as np
import pandas as pd
import requests
import earthaccess
import xarray as xr
from pathlib import Path
from global_land_mask import globe

BASE_DIR = Path(__file__).resolve().parent.parent

# Defining constants
GRID_STEP_DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0     # spacing of the Earth search grid (degrees)
LAT_RANGE = (-85, 85)
CLIM_URL = 'https://power.larc.nasa.gov/api/temporal/climatology/point'
HOURLY_URL = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
PARAM_LIST = ['T2M', 'T2M_MIN', 'T2M_RANGE', 'WS2M', 'PS']
COMMUNITY = 'AG'
MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
MIN_INTERVAL_S = 2.5             # seconds between requests. POWER has reported a cap of 30 requests/min on the climatology endpoint
RATE_LIMIT_WAIT_S = 60           # pause when POWER answers "429 Too Many Requests"
RETRIES = 5
CHECKPOINT_EVERY = 25            # points between saves of the raw table

# Match rules for Method A: (weight, value that scores 0, value that scores 1). One-sided: harsher than Mars still scores 1.
RULES = {
    'COLD':  (25, -35.0, -65.0),    # night minimum air temperature, deg C
    'RANGE': (20, 15.0, 45.0),      # day-night temperature range, K
    'WIND':  (25, 2.0, 8.0),        # mean 2 m wind speed, m/s
}
SHAPE_WEIGHT, DUST_WEIGHT = 10, 20
SHAPE_FULL = (1.6, 2.3)          # night/day wind ratio that scores 1
SHAPE_ZERO = (1.0, 3.0)          # ratios at or beyond these score 0
DUST_ZERO, DUST_FULL = 0.05, 0.30
NIGHT_HOURS = {18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5}
YEARS = (2022, 2023, 2024)       # years averaged for the hourly wind and the dust climatology
STAGE2_N = 50                    # stage-1 candidates that get the wind-shape and dust indicators
MERRA_SHORT_NAME, MERRA_VERSION = 'M2TMNXAER', '5.12.4'    # monthly MERRA-2 aerosol diagnostics

# Mars reference values (to flag overshoot and for the density tie-breaker)
MARS_COLD_C, MARS_RANGE_K, MARS_WIND_MS = -76.5, 66.6, 11.1
MARS_DENSITY = 0.0163            # kg/m3, mean REMS density over the three sols
R_EARTH_AIR = 287.0              # J/kg/K

RAW_DIR = BASE_DIR / 'data' / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)
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


# ---------------------------------------------------------------- STAGE 1: climate scan (cold, range, wind)
lats = np.arange(LAT_RANGE[0], LAT_RANGE[1] + 0.01, GRID_STEP_DEG)
lons = np.arange(-180, 180, GRID_STEP_DEG)
points = [(float(la), float(lo)) for la in lats for lo in lons if globe.is_land(la, lo)]

# One-off: bring in the individual JSON files saved by the earlier version of this script, so that progress isn't lost
OLD_JSON_DIR = RAW_DIR / 'power_climatology'
if not CLIM_FILE.exists() and OLD_JSON_DIR.exists():
    old_rows = []
    for f in OLD_JSON_DIR.glob('*.json'):
        d = json.loads(f.read_text())
        lon, lat, elev = d['geometry']['coordinates']
        par = d['properties']['parameter']
        for m in MONTHS:
            old_rows.append({'LAT': lat, 'LON': lon, 'ELEV_M': elev, 'MONTH': m, **{p: par[p][m] for p in PARAM_LIST}})
    if old_rows:
        pd.DataFrame(old_rows).to_csv(CLIM_FILE, index=False)
        print(f'Imported {len(old_rows) // 12} points from the old JSON files')

clim_df = pd.read_csv(CLIM_FILE) if CLIM_FILE.exists() else pd.DataFrame()
done = set(zip(clim_df['LAT'], clim_df['LON'])) if len(clim_df) else set()
todo = [p for p in points if p not in done]
print(f'{len(points)} land points on a {GRID_STEP_DEG} degree grid, {len(done)} already downloaded, {len(todo)} to go '
      f'(about {len(todo) * MIN_INTERVAL_S / 60:.0f} min)')

new_rows = []
for i, (lat, lon) in enumerate(todo, 1):
    data = polite_get(CLIM_URL, {'parameters': ','.join(PARAM_LIST), 'community': COMMUNITY,
                                 'latitude': lat, 'longitude': lon, 'format': 'JSON'})
    par = data['properties']['parameter']
    elev = data['geometry']['coordinates'][2]
    for m in MONTHS:
        new_rows.append({'LAT': lat, 'LON': lon, 'ELEV_M': elev, 'MONTH': m, **{p: par[p][m] for p in PARAM_LIST}})
    if i % CHECKPOINT_EVERY == 0 or i == len(todo):
        clim_df = pd.concat([clim_df, pd.DataFrame(new_rows)], ignore_index=True)
        new_rows = []
        clim_df.to_csv(CLIM_FILE, index=False)
        print(f'  {i}/{len(todo)} points saved')

# Score every point-month, then keep each point's best month (all indicators from the same month)
pset = set(points)
clim_df = clim_df[[(a, b) in pset for a, b in zip(clim_df['LAT'], clim_df['LON'])]]
valid = clim_df[(clim_df[PARAM_LIST] > -990).all(axis=1)].copy()
for name, col in (('COLD', 'T2M_MIN'), ('RANGE', 'T2M_RANGE'), ('WIND', 'WS2M')):
    _, zero, full = RULES[name]
    valid['SCORE_' + name] = ((valid[col] - zero) / (full - zero)).clip(0, 1)
stage1_weight = sum(r[0] for r in RULES.values())
valid['STAGE1_POINTS'] = sum(RULES[n][0] * valid['SCORE_' + n] for n in RULES)           # out of 70
best_df = valid.loc[valid.groupby(['LAT', 'LON'])['STAGE1_POINTS'].idxmax()].copy()
best_df['SCORE'] = best_df['STAGE1_POINTS'] / stage1_weight
best_df['DENSITY_KGM3'] = best_df['PS'] * 1000 / (R_EARTH_AIR * (best_df['T2M'] + 273.15))
best_df['DENSITY_VS_MARS'] = best_df['DENSITY_KGM3'] / MARS_DENSITY
best_df['OVERSHOOT'] = [','.join(n for n, hit in (('COLD', r.T2M_MIN < MARS_COLD_C), ('RANGE', r.T2M_RANGE > MARS_RANGE_K),
                                                  ('WIND', r.WS2M > MARS_WIND_MS)) if hit) for r in best_df.itertuples()]
best_df = best_df.rename(columns={'MONTH': 'BEST_MONTH', 'T2M_MIN': 'T2M_MIN_C', 'T2M_RANGE': 'T2M_RANGE_K',
                                  'WS2M': 'WS2M_MS', 'PS': 'PS_KPA'})
stage1_df = best_df.sort_values(['SCORE', 'DENSITY_VS_MARS'], ascending=[False, True]).reset_index(drop=True)
print('\nSTAGE 1 - top 10 (cold, range, wind only):')
print(stage1_df[['LAT', 'LON', 'ELEV_M', 'BEST_MONTH', 'SCORE', 'T2M_MIN_C', 'T2M_RANGE_K', 'WS2M_MS', 'OVERSHOOT']].head(10).round(3).to_string(index=False))


# ---------------------------------------------------------------- STAGE 2: wind day-night shape and dust (top candidates only)
def shape_score(r):
    if r < SHAPE_FULL[0]:
        return float(np.clip((r - SHAPE_ZERO[0]) / (SHAPE_FULL[0] - SHAPE_ZERO[0]), 0, 1))
    if r > SHAPE_FULL[1]:
        return float(np.clip((SHAPE_ZERO[1] - r) / (SHAPE_ZERO[1] - SHAPE_FULL[1]), 0, 1))
    return 1.0


def dust_by_month(lat_values, lon_values):
    """MERRA-2 dust extinction optical depth (550 nm), monthly climatology over YEARS at each point. Returns array (12, n)."""
    earthaccess.login(persist=True)
    granules = earthaccess.search_data(short_name=MERRA_SHORT_NAME, version=MERRA_VERSION,
                                       temporal=(f'{YEARS[0]}-01-01', f'{YEARS[-1]}-12-31'))
    ds = xr.open_mfdataset(earthaccess.open(granules), engine='h5netcdf')
    da = ds['DUEXTTAU'].sel(lat=xr.DataArray(lat_values, dims='pt'), lon=xr.DataArray(lon_values, dims='pt'),
                            method='nearest').load()
    return da.groupby('time.month').mean('time').values


top_df = stage1_df.head(STAGE2_N).copy()

# Hourly wind: night and day means for each candidate's best month, one row per point-month-year
hourly_df = pd.read_csv(HOURLY_FILE) if HOURLY_FILE.exists() else pd.DataFrame(columns=['LAT', 'LON', 'MONTH', 'YEAR', 'NIGHT_MEAN', 'DAY_MEAN'])
done_h = set(zip(hourly_df['LAT'], hourly_df['LON'], hourly_df['MONTH'], hourly_df['YEAR']))
print(f'\nSTAGE 2 - hourly wind for {len(top_df)} candidates x {len(YEARS)} years')
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
top_df['SCORE_SHAPE'] = [shape_score(r) for r in ratios]

# Dust
dust = dust_by_month(top_df['LAT'].values, top_df['LON'].values)
top_df['DUST_TAU'] = [dust[MONTHS.index(m), i] for i, m in enumerate(top_df['BEST_MONTH'])]
top_df['SCORE_DUST'] = ((top_df['DUST_TAU'] - DUST_ZERO) / (DUST_FULL - DUST_ZERO)).clip(0, 1)

# Final score out of 100
top_df['FINAL_SCORE'] = (top_df['STAGE1_POINTS'] + SHAPE_WEIGHT * top_df['SCORE_SHAPE'] + DUST_WEIGHT * top_df['SCORE_DUST']) / 100
final_df = top_df.sort_values(['FINAL_SCORE', 'DENSITY_VS_MARS'], ascending=[False, True]).reset_index(drop=True)

# Summarise + save
print('\nFINAL - top 15 (all five indicators):')
print(final_df[['LAT', 'LON', 'ELEV_M', 'BEST_MONTH', 'FINAL_SCORE', 'SCORE_COLD', 'SCORE_RANGE', 'SCORE_WIND', 'SCORE_SHAPE',
                'SCORE_DUST', 'T2M_MIN_C', 'T2M_RANGE_K', 'WS2M_MS', 'WIND_NIGHT_DAY_RATIO', 'DUST_TAU',
                'OVERSHOOT']].head(15).round(3).to_string(index=False))
stage1_df.to_csv(RAW_DIR / 'earth_climate_stage1.csv', index=False)
final_df.to_csv(RAW_DIR / 'earth_climate_final.csv', index=False)
print('Earth climate search complete')