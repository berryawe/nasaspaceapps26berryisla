import re
import time
import zipfile
import calendar
import tempfile
import warnings
import numpy as np
import pandas as pd
import rasterio
import requests
import earthaccess
from scipy.ndimage import uniform_filter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Method B: find Mount Sharp-like terrain first, then check the climate there.
# Defining constants
RAW_DIR = BASE_DIR / 'data' / 'raw'
MARS_TERRAIN_FILE = RAW_DIR / 'terrain_mount_sharp.csv'    # Mount Sharp indicators from the Mars terrain script
WINDOWS_FILE = RAW_DIR / 'earth_b_windows.csv'             # indicators of every window; saved region by region, so a rerun resumes
ELEV_FILE = RAW_DIR / 'earth_b_elevation.csv'              # elevation of every search box at about 1 km (plain CSV)
FINAL_FILE = RAW_DIR / 'earth_b_final.csv'                 # best window per region with terrain and climate scores

# Mountain regions to scan: (name, latitude, longitude of the search box centre)
REGIONS = [
    ('Himalaya, southern Tibet', 28.5, 86.0),
    ('Karakoram', 36.0, 76.5),
    ('Pamirs', 38.5, 73.0),
    ('Tien Shan', 42.0, 80.0),
    ('Kunlun', 36.4, 87.4),
    ('Altiplano-Puna volcanoes, Andes', -23.0, -67.8),
    ('St Elias Mountains, Alaska-Yukon', 60.5, -140.5),
    ('Ellesmere Island, Arctic Canada', 79.5, -78.0),
    ('Transantarctic Mountains, Dry Valleys', -77.5, 162.0),
]
SEARCH_HALF_KM = 100.0                       # search box around each region centre = a square of 2 x this width
WINDOW_KM = 80.0                             # Mount Sharp comparison window
STRIDE_KM = 20.0                             # window step
TARGET_RES_M = 200.0                         # cell size used for slope/roughness (same as the Mars DEM)
SAVE_RES_KM = 1.0                            # cell size of the saved elevation table
ROUGHNESS_WINDOWS_M = (1000, 5000)
MIN_VALID_SHARE = 0.8                        # skip windows with more than 20% missing elevation
PCT_LOW, PCT_HIGH = 0.5, 99.5                # relief = these percentiles, so single-pixel DEM spikes don't count
ASTER_SHORT_NAME, ASTER_VERSION = 'ASTGTM', '003'   # NASA/METI ASTER Global DEM v3, 1 arc-second (~30 m), 83N to 83S
ASTER_CELL_M = 30.87                         # 1 arc-second of latitude
M_PER_DEG = 111195.0                         # Earth sphere
TILE_PIXELS = 3600                           # cells per 1-degree tile side (the extra overlap row/column is dropped)

# Terrain score: closeness to Mount Sharp on each indicator, weight out of 100 (same as Method A)
WEIGHTS = {'ELEV_RANGE': 30, 'SLOPE_MEAN': 25, 'ROUGHNESS_1000M': 20, 'ROUGHNESS_5000M': 25}
CLIMATE_WEIGHT = 0.5                         # combined score = this x climate score + (1 - this) x terrain score

# Climate rules (same as Method A): (weight, value that scores 0, value that scores 1)
RULES = {'COLD': (25, -35.0, -65.0), 'RANGE': (20, 5.0, 20.0), 'WIND': (25, 2.0, 8.0)}
MIN_RANGE_K = 8.0                            # gate: a month needs a real day-night cycle
MAX_NIGHT_MIN_C = RULES['COLD'][1]           # gate: a month must be cold enough to score on the cold indicator
CLIM_URL = 'https://power.larc.nasa.gov/api/temporal/climatology/point'
HOURLY_URL = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
COMMUNITY = 'AG'
PARAM_LIST = ['T2M', 'T2M_MIN', 'T2M_RANGE', 'WS2M', 'PS']
MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
NIGHT_HOURS = {18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5}
YEARS = (2022, 2023, 2024)                   # years averaged for the hourly wind
MIN_INTERVAL_S = 2.5                         # seconds between POWER requests
RATE_LIMIT_WAIT_S = 60
RETRIES = 5
MARS_COLD_C, MARS_RANGE_K, MARS_WIND_MS = -76.5, 66.6, 11.1
MARS_DENSITY = 0.0163                        # kg/m3, mean REMS density over the three sols
R_EARTH_AIR = 287.0                          # J/kg/K

last_call = [0.0]

# Mount Sharp targets
mars_df = pd.read_csv(MARS_TERRAIN_FILE).set_index('INDICATOR')['VALUE']
targets = {k: float(mars_df[k]) for k in WEIGHTS}
print(f'{len(REGIONS)} mountain regions to scan, {SEARCH_HALF_KM * 2:.0f} km box each, {WINDOW_KM:.0f} km windows')

# Regions already finished in an earlier run are skipped
done = set()
if WINDOWS_FILE.exists():
    prev = pd.read_csv(WINDOWS_FILE)
    done = set(zip(prev['LAT'], prev['LON']))
earthaccess.login(persist=True)


def on_ice(lat, lon):
    """Antarctica or Greenland: ASTER elevations are unreliable over smooth ice (the Method A check found impossible heights)."""
    return lat <= -60 or (59 <= lat <= 84 and -74 <= lon <= -11)


def tile_name(lat, lon):
    return f'ASTGTMV003_{"N" if lat >= 0 else "S"}{abs(lat):02d}{"E" if lon >= 0 else "W"}{abs(lon):03d}'


def reduce_tile(file_path, fy, fx):
    """Open a downloaded ASTER tile and average it down to about TARGET_RES_M cells. Returns (tile name, small array)."""
    name = re.search(r'ASTGTMV003_[NS]\d{2}[EW]\d{3}', file_path.name).group()
    with open(file_path, 'rb') as fh:
        head = fh.read(16)
    if head.lstrip().startswith(b'<'):
        raise RuntimeError(f'{file_path.name} is a web page, not data - the Earthdata login or the LP DAAC approval failed')
    if zipfile.is_zipfile(file_path):                        # tile delivered as a zip holding the _dem.tif
        with zipfile.ZipFile(file_path) as zf:
            zf.extract(f'{name}_dem.tif', file_path.parent)
        file_path = file_path.parent / f'{name}_dem.tif'
    with rasterio.open(file_path) as src:                    # (a plain GeoTIFF tile is opened directly)
        z = src.read(1)[:TILE_PIXELS, :TILE_PIXELS].astype('float32')
    z[z <= -9000] = np.nan                                  # voids
    ny, nx = TILE_PIXELS // fy, TILE_PIXELS // fx
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        small = np.nanmean(z[:ny * fy, :nx * fx].reshape(ny, fy, nx, fx), axis=(1, 3))
    return name, small.astype('float32')


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


# ---------------------------------------------------------------- PART 1: terrain first, scan each mountain region
for name, lat, lon in REGIONS:
    if (lat, lon) in done:
        print(f'\n{name}: already done, skipped')
        continue
    # Search box in degrees
    half_lat = SEARCH_HALF_KM * 1000 / M_PER_DEG
    half_lon = half_lat / np.cos(np.radians(lat))
    south, north, west, east = lat - half_lat, lat + half_lat, lon - half_lon, lon + half_lon
    if west < -180 or east >= 180:
        raise RuntimeError(f'Search box for {name} crosses the date line - not handled')
    # Tiles needed, and how much to average each so cells are about TARGET_RES_M in both directions
    fy = max(1, round(TARGET_RES_M / ASTER_CELL_M))
    fx = max(1, round(TARGET_RES_M / (ASTER_CELL_M * np.cos(np.radians(lat)))))
    lat_tiles = list(range(int(np.floor(north)), int(np.floor(south)) - 1, -1))      # north to south
    lon_tiles = list(range(int(np.floor(west)), int(np.floor(east)) + 1))            # west to east
    names = {(la, lo): tile_name(la, lo) for la in lat_tiles for lo in lon_tiles}
    ny, nx = TILE_PIXELS // fy, TILE_PIXELS // fx
    print(f'\n{name} ({lat}, {lon}): {len(names)} tiles needed')

    # Download into a temporary folder (deleted afterwards) and average each tile down
    with tempfile.TemporaryDirectory() as tmp:
        granules = earthaccess.search_data(short_name=ASTER_SHORT_NAME, version=ASTER_VERSION,
                                           bounding_box=(west, south, east, north))
        # Only the elevation file of each tile (skip the _num.tif quality files and previews)
        urls = [u for g in granules for u in g.data_links()
                if any(n in u for n in names.values()) and (u.endswith('_dem.tif') or u.endswith('.zip'))]
        print(f'  downloading {len(urls)} files, e.g. {urls[0].split("/")[-1] if urls else "none found"}')
        files = earthaccess.download(urls, tmp)
        if not urls or len(files) < len(urls):
            raise RuntimeError(f'Only {len(files)} of {len(urls)} tiles downloaded - check the Earthdata login and that '
                               f'"LP DAAC Data Pool" is approved under Applications -> Authorized Apps')
        tiles = dict(reduce_tile(Path(f), fy, fx) for f in files)

    # Mosaic the tiles (tiles with no ASTER data, e.g. open sea, stay empty)
    mosaic = np.full((len(lat_tiles) * ny, len(lon_tiles) * nx), np.nan, dtype='float32')
    for i, la in enumerate(lat_tiles):
        for j, lo in enumerate(lon_tiles):
            if names[(la, lo)] in tiles:
                mosaic[i * ny:(i + 1) * ny, j * nx:(j + 1) * nx] = tiles[names[(la, lo)]]
    # Cell centres: row 0 is the top of the northernmost tile, column 0 the west edge of the westernmost tile
    top_lat, left_lon = lat_tiles[0] + 1, lon_tiles[0]
    row_lat = top_lat - (np.arange(mosaic.shape[0]) + 0.5) / ny
    col_lon = left_lon + (np.arange(mosaic.shape[1]) + 0.5) / nx
    r0, r1 = np.searchsorted(-row_lat, -north), np.searchsorted(-row_lat, -south)
    c0, c1 = np.searchsorted(col_lon, west), np.searchsorted(col_lon, east)
    z, row_lat, col_lon = mosaic[r0:r1, c0:c1], row_lat[r0:r1], col_lon[c0:c1]
    if not np.isfinite(z).any():
        raise RuntimeError(f'No elevation data for {name} - the tile download did not work')
    cell_y = M_PER_DEG / ny
    cell_x = M_PER_DEG * np.cos(np.radians(lat)) / nx
    print(f'  elevation grid {z.shape[0]} x {z.shape[1]} cells of {cell_x:.0f} x {cell_y:.0f} m, {np.isfinite(z).mean():.0%} valid')

    # Slope and roughness over the whole box (same definitions as the Mount Sharp script)
    gy, gx = np.gradient(z, cell_y, cell_x)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    z_fill = np.nan_to_num(z, nan=float(np.nanmean(z)))
    rough = {}
    for L in ROUGHNESS_WINDOWS_M:
        k = max(3, int(round(L / cell_x)))
        rough[L] = np.sqrt(np.maximum(uniform_filter(z_fill ** 2, k) - uniform_filter(z_fill, k) ** 2, 0))

    # Slide a Mount Sharp-sized window across the box and measure each position
    wy, wx = int(round(WINDOW_KM * 1000 / cell_y)), int(round(WINDOW_KM * 1000 / cell_x))
    sy, sx = max(1, int(round(STRIDE_KM * 1000 / cell_y))), max(1, int(round(STRIDE_KM * 1000 / cell_x)))
    region_windows = []
    for a in range(0, z.shape[0] - wy + 1, sy):
        for b in range(0, z.shape[1] - wx + 1, sx):
            zw = z[a:a + wy, b:b + wx]
            if np.isfinite(zw).mean() < MIN_VALID_SHARE:
                continue
            c_lat, c_lon = row_lat[a + wy // 2], col_lon[b + wx // 2]
            region_windows.append({'REGION_NAME': name, 'LAT': lat, 'LON': lon,
                                   'WIN_LAT': round(float(c_lat), 3), 'WIN_LON': round(float(c_lon), 3),
                                   'ELEV_RANGE': float(np.nanpercentile(zw, PCT_HIGH) - np.nanpercentile(zw, PCT_LOW)),
                                   'SLOPE_MEAN': float(np.nanmean(slope[a:a + wy, b:b + wx])),
                                   'ROUGHNESS_1000M': float(np.nanmean(rough[1000][a:a + wy, b:b + wx])),
                                   'ROUGHNESS_5000M': float(np.nanmean(rough[5000][a:a + wy, b:b + wx]))})
    print(f'  {len(region_windows)} windows measured')

    # Save: elevation at about 1 km (plain CSV), then the windows (this marks the region as done)
    ky, kx = max(1, round(SAVE_RES_KM * 1000 / cell_y)), max(1, round(SAVE_RES_KM * 1000 / cell_x))
    hb, wb = z.shape[0] // ky * ky, z.shape[1] // kx * kx
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        z_small = np.nanmean(z[:hb, :wb].reshape(hb // ky, ky, wb // kx, kx), axis=(1, 3))
    lat_small = row_lat[:hb].reshape(-1, ky).mean(axis=1)
    lon_small = col_lon[:wb].reshape(-1, kx).mean(axis=1)
    elev_small_df = pd.DataFrame({'SITE_LAT': lat, 'SITE_LON': lon,
                                  'LAT': np.repeat(lat_small, len(lon_small)).round(4),
                                  'LON': np.tile(lon_small, len(lat_small)).round(4),
                                  'ELEV_M': z_small.ravel().round(0)})
    elev_small_df.to_csv(ELEV_FILE, mode='a', header=not ELEV_FILE.exists(), index=False)
    if region_windows:
        pd.DataFrame(region_windows).to_csv(WINDOWS_FILE, mode='a', header=not WINDOWS_FILE.exists(), index=False)
    else:
        print('  no usable windows here (too much missing data)')

# Score every saved window against Mount Sharp (computed here, so weights can be changed without downloading again)
region_centres = pd.DataFrame(REGIONS, columns=['REGION_NAME', 'LAT', 'LON'])
windows_df = pd.read_csv(WINDOWS_FILE).merge(region_centres[['LAT', 'LON']], on=['LAT', 'LON'])
if windows_df.empty:
    raise RuntimeError('No windows had enough elevation data')
# Closeness: smaller value / larger value, so 1 = identical and 0 = nothing alike
windows_df['TERRAIN_SCORE'] = sum(w * np.minimum(windows_df[k], targets[k]) / np.maximum(windows_df[k], targets[k])
                                  for k, w in WEIGHTS.items()) / 100
best_df = windows_df.loc[windows_df.groupby(['LAT', 'LON'])['TERRAIN_SCORE'].idxmax()].reset_index(drop=True)

# Lowest and highest point of each best window from the saved 1 km grid (catches impossible ASTER heights), and a reliability flag
elev_all = pd.read_csv(ELEV_FILE)
half_deg = WINDOW_KM / 2 / (M_PER_DEG / 1000)
low, high = [], []
for r in best_df.itertuples():
    e = elev_all[(elev_all['SITE_LAT'] == r.LAT) & (elev_all['SITE_LON'] == r.LON)]
    w = e[(abs(e['LAT'] - r.WIN_LAT) <= half_deg) & (abs(e['LON'] - r.WIN_LON) <= half_deg / np.cos(np.radians(r.LAT)))]
    low.append(w['ELEV_M'].min())
    high.append(w['ELEV_M'].max())
best_df['ELEV_MIN_1KM'], best_df['ELEV_MAX_1KM'] = low, high
best_df['TERRAIN_DATA'] = ['Unverified (ASTER unreliable over ice)' if on_ice(la, lo) else 'Checked, plausible'
                           for la, lo in zip(best_df['LAT'], best_df['LON'])]
print('\nPART 1 - best Mount Sharp-like window in each region:')
print(best_df[['REGION_NAME', 'WIN_LAT', 'WIN_LON', 'TERRAIN_SCORE', 'ELEV_RANGE', 'SLOPE_MEAN', 'ROUGHNESS_1000M',
               'ROUGHNESS_5000M', 'ELEV_MIN_1KM', 'ELEV_MAX_1KM']].round(2).to_string(index=False))

# ---------------------------------------------------------------- PART 2: climate at each best window (NASA POWER)
print(f'\nPART 2 - climate at the {len(best_df)} best windows')
climate_rows = []
for r in best_df.itertuples():
    data = polite_get(CLIM_URL, {'parameters': ','.join(PARAM_LIST), 'community': COMMUNITY,
                                 'latitude': r.WIN_LAT, 'longitude': r.WIN_LON, 'format': 'JSON'})
    par = data['properties']['parameter']
    month_df = pd.DataFrame({'MONTH': MONTHS, **{p: [par[p][m] for m in MONTHS] for p in PARAM_LIST}})
    month_df = month_df[(month_df[PARAM_LIST] > -990).all(axis=1)].copy()
    for cname, col in (('COLD', 'T2M_MIN'), ('RANGE', 'T2M_RANGE'), ('WIND', 'WS2M')):
        _, zero, full = RULES[cname]
        month_df['SCORE_' + cname] = ((month_df[col] - zero) / (full - zero)).clip(0, 1)
    month_df['SCORE'] = sum(RULES[n][0] * month_df['SCORE_' + n] for n in RULES) / sum(v[0] for v in RULES.values())
    # Best month among those with a real day-night cycle and cold enough nights; if none passes, best month overall and flagged
    passing = month_df[(month_df['T2M_RANGE'] >= MIN_RANGE_K) & (month_df['T2M_MIN'] <= MAX_NIGHT_MIN_C)]
    pool = passing if len(passing) else month_df
    m = pool.loc[pool['SCORE'].idxmax()]
    # Hourly wind in that month: night and day means (local solar time), averaged over YEARS
    month_num = MONTHS.index(m['MONTH']) + 1
    night_means, day_means = [], []
    for y in YEARS:
        last = calendar.monthrange(y, month_num)[1]
        hourly = polite_get(HOURLY_URL, {'parameters': 'WS2M', 'community': COMMUNITY, 'latitude': r.WIN_LAT, 'longitude': r.WIN_LON,
                                         'format': 'JSON', 'start': f'{y}{month_num:02d}01', 'end': f'{y}{month_num:02d}{last}',
                                         'time-standard': 'LST'})
        s = pd.Series(hourly['properties']['parameter']['WS2M'])
        s = s[s > -990]
        hours = np.array([int(k[-2:]) for k in s.index])
        night = np.isin(hours, list(NIGHT_HOURS))
        night_means.append(s.values[night].mean())
        day_means.append(s.values[~night].mean())
    density = m['PS'] * 1000 / (R_EARTH_AIR * (m['T2M'] + 273.15)) / MARS_DENSITY
    climate_rows.append({'LAT': r.LAT, 'LON': r.LON, 'CLIMATE_ELEV_M': data['geometry']['coordinates'][2],
                         'BEST_MONTH': m['MONTH'], 'GATES_PASSED': len(passing) > 0, 'T2M_MIN_C': m['T2M_MIN'],
                         'T2M_RANGE_K': m['T2M_RANGE'], 'WS2M_MS': m['WS2M'],
                         'WIND_NIGHT_DAY_RATIO': np.mean(night_means) / np.mean(day_means), 'DENSITY_VS_MARS': density,
                         'CLIMATE_SCORE': m['SCORE'],
                         'OVERSHOOT': ','.join(n for n, hit in (('COLD', m['T2M_MIN'] < MARS_COLD_C), ('RANGE', m['T2M_RANGE'] > MARS_RANGE_K),
                                                                ('WIND', m['WS2M'] > MARS_WIND_MS)) if hit)})

final_df = best_df.merge(pd.DataFrame(climate_rows), on=['LAT', 'LON'])
final_df['COMBINED_SCORE'] = CLIMATE_WEIGHT * final_df['CLIMATE_SCORE'] + (1 - CLIMATE_WEIGHT) * final_df['TERRAIN_SCORE']
# Regions whose terrain could be checked rank first; unverified ones follow
final_df['UNVERIFIED'] = final_df['TERRAIN_DATA'].str.startswith('Unverified')
final_df = final_df.sort_values(['UNVERIFIED', 'COMBINED_SCORE'], ascending=[True, False]).reset_index(drop=True)

# Summarise + save
print('\nMount Sharp targets:', {k: round(v, 1) for k, v in targets.items()})
print('\nMETHOD B - terrain first, then climate (best combined score first):')
print(final_df[['REGION_NAME', 'WIN_LAT', 'WIN_LON', 'TERRAIN_SCORE', 'CLIMATE_SCORE', 'COMBINED_SCORE', 'BEST_MONTH', 'GATES_PASSED',
                'T2M_MIN_C', 'T2M_RANGE_K', 'WS2M_MS', 'WIND_NIGHT_DAY_RATIO', 'DENSITY_VS_MARS',
                'TERRAIN_DATA']].round(2).to_string(index=False))
final_df.drop(columns='UNVERIFIED').to_csv(FINAL_FILE, index=False)
print('Method B search complete')