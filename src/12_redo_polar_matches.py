import time
import getpass
import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Checks the ASTER terrain of the Antarctic matches against REMA (Reference Elevation Model of Antarctica),
# using the OpenTopography Point Elevation API: one request per point, with a small daily limit.
# Defining constants
RAW_DIR = BASE_DIR / 'data' / 'raw'
A_BEST_FILE = RAW_DIR / 'earth_terrain_best.csv'            # Method A: best terrain window per site
A_ELEV_FILE = RAW_DIR / 'earth_terrain_elevation.csv'       # Method A: ASTER 1 km elevation table
B_BEST_FILE = RAW_DIR / 'earth_b_final.csv'                 # Method B: best terrain window per region
B_ELEV_FILE = RAW_DIR / 'earth_b_elevation.csv'             # Method B: ASTER 1 km elevation table
POINTS_FILE = RAW_DIR / 'rema_points_raw.csv'               # every REMA point queried so far (resumes across days)
CHECK_FILE = RAW_DIR / 'rema_check.csv'                     # one row per Antarctic window: ASTER against REMA

API_URL = 'https://portal.opentopography.org/API/v1/elevation'
DATASET = 'REMA32m'              # REMA mosaic, 32 m cells (other options: REMA10m, REMA2m)
DAILY_LIMIT = 50                 # free key: 50 requests a day (250 for .edu academic accounts or OpenTopography Plus members)
SAFETY_MARGIN = 2                # stop a couple of requests short of the limit
MIN_INTERVAL_S = 1.0             # pause between requests
RETRIES = 3
WINDOW_KM = 80.0                 # same Mount Sharp-sized window as the terrain scripts
OFFSET_KM = 20.0                 # extra points this far north, south, east and west of the window centre
M_PER_DEG = 111195.0
TOLERANCE_M = 300.0              # ASTER and REMA differing by more than this at a point = ASTER is wrong there
                                 # (ASTER is a 1 km average and REMA a point value, so small differences are expected)
ANTARCTIC_LAT = -60.0            # sites south of this are checked

# Key: from the environment if set (export OPENTOPO_API_KEY=...), otherwise typed in (never saved)
api_key = os.environ.get('OPENTOPO_API_KEY') or getpass.getpass('OpenTopography API key: ')

# ---------------------------------------------------------------- The Antarctic windows to check, best combined score first
targets = []
for best_file, elev_file, method, label_col in ((A_BEST_FILE, A_ELEV_FILE, 'A', None), (B_BEST_FILE, B_ELEV_FILE, 'B', 'REGION_NAME')):
    best = pd.read_csv(best_file)
    elev = pd.read_csv(elev_file)
    for r in best[best['LAT'] <= ANTARCTIC_LAT].itertuples():
        label = f'B {r.REGION_NAME}' if label_col else f'A {r.LAT}, {r.LON}'
        grid = elev[(elev['SITE_LAT'] == r.LAT) & (elev['SITE_LON'] == r.LON)].dropna(subset=['ELEV_M'])
        targets.append({'LABEL': label, 'METHOD': method, 'LAT': r.LAT, 'LON': r.LON, 'WIN_LAT': r.WIN_LAT, 'WIN_LON': r.WIN_LON,
                        'COMBINED_SCORE': r.COMBINED_SCORE, 'ASTER_RANGE_200M': r.ELEV_RANGE, 'GRID': grid})
targets.sort(key=lambda t: -t['COMBINED_SCORE'])
print(f'{len(targets)} Antarctic windows to check against {DATASET}')

# Points to query in each window: ASTER's highest and lowest cell, the window centre, and four points around it
points = []
for t in targets:
    half_lat = WINDOW_KM / 2 / (M_PER_DEG / 1000)
    half_lon = half_lat / np.cos(np.radians(t['WIN_LAT']))
    g = t['GRID']
    w = g[(abs(g['LAT'] - t['WIN_LAT']) <= half_lat) & (abs(g['LON'] - t['WIN_LON']) <= half_lon)]
    if w.empty:
        raise RuntimeError(f'No ASTER cells found in the window for {t["LABEL"]}')
    d_lat = OFFSET_KM / (M_PER_DEG / 1000)
    d_lon = d_lat / np.cos(np.radians(t['WIN_LAT']))
    wanted = [('ASTER highest cell', w.loc[w['ELEV_M'].idxmax()]), ('ASTER lowest cell', w.loc[w['ELEV_M'].idxmin()])]
    for kind, dla, dlo in (('window centre', 0, 0), ('north', d_lat, 0), ('south', -d_lat, 0), ('east', 0, d_lon), ('west', 0, -d_lon)):
        la, lo = t['WIN_LAT'] + dla, t['WIN_LON'] + dlo
        nearest = w.iloc[((w['LAT'] - la) ** 2 + ((w['LON'] - lo) * np.cos(np.radians(la))) ** 2).argmin()]
        wanted.append((kind, pd.Series({'LAT': la, 'LON': lo, 'ELEV_M': nearest['ELEV_M']})))
    for kind, cell in wanted:
        points.append({'LABEL': t['LABEL'], 'SITE_LAT': t['LAT'], 'SITE_LON': t['LON'], 'POINT': kind,
                       'LAT': round(float(cell['LAT']), 4), 'LON': round(float(cell['LON']), 4), 'ASTER_M': float(cell['ELEV_M'])})

# Points already queried on earlier days are not asked again
done_df = pd.read_csv(POINTS_FILE) if POINTS_FILE.exists() else pd.DataFrame(
    columns=['LABEL', 'SITE_LAT', 'SITE_LON', 'POINT', 'LAT', 'LON', 'ASTER_M', 'REMA_M', 'QUERY_DATE'])
done_keys = set(zip(done_df['LABEL'], done_df['POINT']))
today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
used_today = int((done_df['QUERY_DATE'] == today).sum())
todo = [p for p in points if (p['LABEL'], p['POINT']) not in done_keys]
budget = max(0, DAILY_LIMIT - SAFETY_MARGIN - used_today)
print(f'{len(points)} points in total, {len(points) - len(todo)} already done, {len(todo)} to go; '
      f'{used_today} used today, {budget} left in today\'s budget')


def find_elevation(obj):
    """Pull the elevation value out of the JSON reply (the exact layout isn't documented, so look for a number under an 'elev' key)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if 'elev' in k.lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        for v in obj.values():
            found = find_elevation(v)
            if found is not None:
                return found
    if isinstance(obj, list):
        for v in obj:
            found = find_elevation(v)
            if found is not None:
                return found
    return None


last_call = [0.0]
printed_example = [False]


def query_rema(lat, lon):
    for attempt in range(RETRIES):
        wait = MIN_INTERVAL_S - (time.time() - last_call[0])
        if wait > 0:
            time.sleep(wait)
        last_call[0] = time.time()
        r = requests.get(API_URL, params={'latitude': lat, 'longitude': lon, 'dataset': DATASET, 'API_Key': api_key}, timeout=60)
        if r.status_code == 200:
            if not printed_example[0]:
                print(f'  example reply: {r.text[:300]}')
                printed_example[0] = True
            value = find_elevation(r.json())
            if value is None:
                raise RuntimeError(f'Could not find an elevation in the reply: {r.text[:400]}')
            return value
        if r.status_code in (401, 403):
            raise RuntimeError(f'OpenTopography refused the request ({r.status_code}): check the API key. {r.text[:200]}')
        if r.status_code == 429:
            raise RuntimeError(f'Daily limit reached (429). Run the script again tomorrow - finished points are saved. {r.text[:200]}')
        if r.status_code == 400:
            raise RuntimeError(f'OpenTopography rejected the request (400): {r.text[:300]}')
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f'Request failed after {RETRIES} tries: {r.status_code} {r.text[:200]}')


new_rows = []
for p in todo[:budget]:
    try:
        rema = query_rema(p['LAT'], p['LON'])
    except RuntimeError as e:
        print(f'  stopped: {e}')
        break
    new_rows.append({**p, 'REMA_M': rema, 'QUERY_DATE': today})
    pd.DataFrame(new_rows).pipe(lambda d: pd.concat([done_df, d], ignore_index=True)).to_csv(POINTS_FILE, index=False)   # save after every point
    print(f'  {p["LABEL"]:<28} {p["POINT"]:<20} ASTER {p["ASTER_M"]:>7.0f} m   REMA {rema:>7.0f} m')
if len(todo) > budget:
    print(f'\n{len(todo) - budget} points left over: today\'s limit is used up. Run the script again tomorrow to finish them.')

# ---------------------------------------------------------------- ASTER against REMA, one row per window
all_points = pd.read_csv(POINTS_FILE) if POINTS_FILE.exists() else done_df
rows = []
for t in targets:
    pts = all_points[all_points['LABEL'] == t['LABEL']].dropna(subset=['REMA_M'])
    if len(pts) < len(points) // len(targets):
        rows.append({'LABEL': t['LABEL'], 'POINTS_DONE': len(pts), 'VERDICT': 'Not finished (daily limit): run again tomorrow'})
        continue
    diff = pts['ASTER_M'] - pts['REMA_M']
    rows.append({'LABEL': t['LABEL'], 'POINTS_DONE': len(pts),
                 'ASTER_RELIEF_AT_POINTS_M': pts['ASTER_M'].max() - pts['ASTER_M'].min(),
                 'REMA_RELIEF_AT_POINTS_M': pts['REMA_M'].max() - pts['REMA_M'].min(),
                 'REMA_LOWEST_M': pts['REMA_M'].min(), 'REMA_HIGHEST_M': pts['REMA_M'].max(),
                 'MEDIAN_ASTER_MINUS_REMA_M': diff.median(), 'MAX_ABS_DIFF_M': diff.abs().max(),
                 'VERDICT': 'ASTER agrees with REMA' if diff.abs().max() <= TOLERANCE_M else 'ASTER disagrees with REMA (do not trust its terrain)'})
check_df = pd.DataFrame(rows)

# Summarise + save
print(f'\nASTER AGAINST {DATASET} (tolerance {TOLERANCE_M:.0f} m), per Antarctic window:')
print(check_df.round(0).to_string(index=False))
check_df.to_csv(CHECK_FILE, index=False)
print(f'Saved {POINTS_FILE.name} and {CHECK_FILE.name}')