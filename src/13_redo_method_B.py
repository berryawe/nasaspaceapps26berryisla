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
import xarray as xr
from scipy.ndimage import uniform_filter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Method B, round 2: find Mount Sharp-like terrain first (more regions, several windows each), then check the climate.
# The same run also scores known Mars analogues and non-analogue controls, to test the method itself.
# Defining constants
RAW_DIR = BASE_DIR / 'data' / 'raw'
CLEAN_DIR = BASE_DIR / 'data' / 'clean'
MARS_TERRAIN_FILE = RAW_DIR / 'terrain_mount_sharp.csv'    # Mount Sharp indicators from the Mars terrain script
WINDOWS_FILE = RAW_DIR / 'earth_b2_windows.csv'            # every window of every region; saved region by region, so a rerun resumes
CLIMATE_FILE = RAW_DIR / 'earth_b2_climate.csv'            # climate of each chosen window; saved window by window, so a rerun resumes
TOP_WINDOWS_FILE = CLEAN_DIR / 'earth_b2_top_windows.csv'  # the chosen windows (top terrain matches in each region) with all scores
FINAL_FILE = CLEAN_DIR / 'earth_b2_regions.csv'            # best window per region, candidates + analogues + controls together
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Regions to scan: (name, latitude, longitude of the search box centre, kind)
#   Candidate = a cold, high, dry mountain region that could be the answer
#   Analogue  = a place already used as a Mars analogue (validation: should score well)
#   Control   = a place that is clearly NOT like Mars (validation: should score badly)
REGIONS = [
    ('Himalaya, southern Tibet', 28.5, 86.0, 'Candidate'),
    ('Karakoram', 36.0, 76.5, 'Candidate'),
    ('Pamirs', 38.5, 73.0, 'Candidate'),
    ('Tien Shan, eastern', 42.0, 80.0, 'Candidate'),
    ('Kunlun', 36.4, 87.4, 'Candidate'),
    ('Altiplano-Puna volcanoes, Andes', -23.0, -67.8, 'Candidate'),
    ('St Elias Mountains, Alaska-Yukon', 60.5, -140.5, 'Candidate'),
    ('Ellesmere Island, Arctic Canada', 79.5, -78.0, 'Candidate'),
    ('Altiplano, northern Andes (Sajama)', -18.0, -69.0, 'Candidate'),
    ('High Andes (Ojos del Salado)', -27.1, -68.5, 'Candidate'),
    ('Hindu Kush', 36.5, 71.0, 'Candidate'),
    ('Ladakh', 34.0, 77.5, 'Candidate'),
    ('Zanskar-Spiti', 31.5, 78.5, 'Candidate'),
    ('Changtang plateau, northern Tibet', 34.5, 84.0, 'Candidate'),
    ('Qilian Shan', 38.5, 98.0, 'Candidate'),
    ('Altun Shan', 37.5, 88.5, 'Candidate'),
    ('Tien Shan, Kyrgyzstan interior', 41.0, 75.0, 'Candidate'),
    ('Bogda Shan', 43.5, 88.0, 'Candidate'),
    ('Aksai Chin', 35.5, 79.5, 'Candidate'),
    ('Mongolian Altai', 48.5, 90.0, 'Candidate'),
    ('Khangai Mountains, Mongolia', 47.0, 99.0, 'Candidate'),
    ('Gobi Altai', 44.5, 100.5, 'Candidate'),
    ('Colorado Rockies', 39.0, -106.5, 'Candidate'),
    ('Wind River Range, Wyoming', 43.0, -109.5, 'Candidate'),
    ('Sierra Nevada, California', 37.0, -118.5, 'Candidate'),
    ('Brooks Range, Alaska', 68.0, -152.0, 'Candidate'),
    ('Alaska Range', 63.0, -150.0, 'Candidate'),
    ('Central Andes (Aconcagua)', -32.6, -70.0, 'Candidate'),
    ('Caucasus', 42.7, 43.0, 'Candidate'),
    ('Alborz (Damavand)', 35.9, 52.1, 'Candidate'),
    ('Chersky Range, Siberia', 65.0, 143.0, 'Candidate'),
    ('Putorana Plateau, Siberia', 68.5, 94.0, 'Candidate'),
    ('Polar Urals', 66.0, 65.0, 'Candidate'),
    ('Dry Valleys, Antarctica', -77.5, 162.0, 'Analogue'),
    ('Devon Island, Arctic Canada', 75.4, -89.7, 'Analogue'),
    ('Atacama Desert, Yungay', -24.1, -69.9, 'Analogue'),
    ('Qaidam Basin, China', 37.8, 91.5, 'Analogue'),
    ('Mars Desert Research Station, Utah', 38.4, -110.8, 'Analogue'),
    ('Salar de Uyuni, Bolivia', -20.2, -67.5, 'Analogue'),
    ('Tibesti Mountains, Chad', 21.0, 17.5, 'Analogue'),
    ('Death Valley, California', 36.5, -117.0, 'Analogue'),
    ('Amazon basin', -3.0, -60.0, 'Control'),
    ('Kansas plains', 38.5, -98.0, 'Control'),
    ('Congo basin', 0.0, 23.0, 'Control'),
    ('Swiss Alps', 46.5, 9.0, 'Control'),
]
TOP_PER_REGION = 4                           # windows kept per region (the best terrain matches)
MIN_SEPARATION_KM = 80.0                     # chosen windows in one region must be at least this far apart (no overlap)
SEARCH_HALF_KM = 100.0                       # search box around each region centre = a square of 2 x this width
WINDOW_KM = 80.0                             # Mount Sharp comparison window
STRIDE_KM = 20.0                             # window step
TARGET_RES_M = 200.0                         # cell size used for slope/roughness (same as the Mars DEM)
ROUGHNESS_WINDOWS_M = (1000, 5000)
MIN_VALID_SHARE = 0.8                        # skip windows with more than 20% missing elevation
PCT_LOW, PCT_HIGH = 0.5, 99.5                # relief = these percentiles, so single-pixel DEM spikes don't count
SANE_ELEV_M = (-450.0, 8900.0)               # outside this range a window's elevation is impossible on Earth (DEM error)
ASTER_SHORT_NAME, ASTER_VERSION = 'ASTGTM', '003'   # NASA/METI ASTER Global DEM v3, 1 arc-second (~30 m), 83N to 83S
ASTER_CELL_M = 30.87                         # 1 arc-second of latitude
M_PER_DEG = 111195.0                         # Earth sphere
TILE_PIXELS = 3600                           # cells per 1-degree tile side (the extra overlap row/column is dropped)

# Terrain score: closeness to Mount Sharp on each indicator, weight out of 100 (same as Method A)
WEIGHTS = {'ELEV_RANGE': 30, 'SLOPE_MEAN': 25, 'ROUGHNESS_1000M': 20, 'ROUGHNESS_5000M': 25}
CLIMATE_WEIGHT = 0.6                         # combined score = this x climate score + (1 - this) x terrain score (same as the combiner)

# Climate rules (same as Method A): (weight, value that scores 0, value that scores 1)
RULES = {'COLD': (25, -35.0, -65.0), 'RANGE': (20, 5.0, 20.0), 'WIND': (25, 2.0, 8.0)}
MIN_RANGE_K = 8.0                            # gate: a month needs a real day-night cycle
MAX_NIGHT_MIN_C = RULES['COLD'][1]           # gate: a month must be cold enough to score on the cold indicator
CLIM_URL = 'https://power.larc.nasa.gov/api/temporal/climatology/point'
HOURLY_URL = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
COMMUNITY = 'AG'
PARAM_LIST = ['T2M', 'T2M_MIN', 'T2M_RANGE', 'WS2M', 'PS', 'PRECTOTCORR']
MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
NIGHT_HOURS = {18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5}
YEARS = (2022, 2023, 2024)                   # years averaged for the hourly wind and the dust
MIN_INTERVAL_S = 2.5                         # seconds between POWER requests
RATE_LIMIT_WAIT_S = 60
RETRIES = 5
MERRA_SHORT_NAME, MERRA_VERSION = 'M2TMNXAER', '5.12.4'   # monthly MERRA-2 aerosol diagnostics (dust optical depth)
MARS_COLD_C, MARS_RANGE_K, MARS_WIND_MS, MARS_DUST = -76.5, 66.6, 11.1, 0.49
MARS_DENSITY = 0.0163                        # kg/m3, mean REMS density over the three sols
R_EARTH_AIR = 287.0                          # J/kg/K

# Validation tests, fixed before the run (nothing here is tuned to the result)
#   V1: no control site may pass the climate gates
#   V2: the analogues' median combined score must beat the controls' median
NEAR_TIE = 0.01

last_call = [0.0]

# Mount Sharp targets
mars_df = pd.read_csv(MARS_TERRAIN_FILE).set_index('INDICATOR')['VALUE']
targets = {k: float(mars_df[k]) for k in WEIGHTS}
kinds = pd.Series([r[3] for r in REGIONS]).value_counts()
print(f'{len(REGIONS)} regions to scan ({kinds.get("Candidate", 0)} candidates, {kinds.get("Analogue", 0)} analogues, '
      f'{kinds.get("Control", 0)} controls), {SEARCH_HALF_KM * 2:.0f} km box each, {WINDOW_KM:.0f} km windows, top {TOP_PER_REGION} per region')

# Regions already finished in an earlier run are skipped
done = set()
if WINDOWS_FILE.exists():
    prev = pd.read_csv(WINDOWS_FILE)
    done = set(zip(prev['LAT'], prev['LON']))
earthaccess.login(persist=True)


def on_ice(lat, lon):
    """Antarctica or Greenland: ASTER elevations are unreliable over smooth ice (the Method A check found impossible heights)."""
    return lat <= -60 or (59 <= lat <= 84 and -74 <= lon <= -11)


def km_between(lat1, lon1, lat2, lon2):
    """Approximate distance in km (flat-earth, fine for short distances)."""
    dy = (lat2 - lat1) * M_PER_DEG / 1000
    dx = (lon2 - lon1) * M_PER_DEG / 1000 * np.cos(np.radians((lat1 + lat2) / 2))
    return float(np.hypot(dx, dy))


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


# ---------------------------------------------------------------- PART 1: terrain first, scan each region
for name, lat, lon, kind in REGIONS:
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
    print(f'\n{name} ({lat}, {lon}) [{kind}]: {len(names)} tiles needed')

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
            low, high = np.nanpercentile(zw, [PCT_LOW, PCT_HIGH])
            region_windows.append({'REGION_NAME': name, 'KIND': kind, 'LAT': lat, 'LON': lon,
                                   'WIN_LAT': round(float(row_lat[a + wy // 2]), 3), 'WIN_LON': round(float(col_lon[b + wx // 2]), 3),
                                   'ELEV_RANGE': float(high - low), 'ELEV_LOW_M': float(low), 'ELEV_HIGH_M': float(high),
                                   'SLOPE_MEAN': float(np.nanmean(slope[a:a + wy, b:b + wx])),
                                   'ROUGHNESS_1000M': float(np.nanmean(rough[1000][a:a + wy, b:b + wx])),
                                   'ROUGHNESS_5000M': float(np.nanmean(rough[5000][a:a + wy, b:b + wx]))})
    print(f'  {len(region_windows)} windows measured')
    if region_windows:                                                     # saving marks the region as done
        pd.DataFrame(region_windows).to_csv(WINDOWS_FILE, mode='a', header=not WINDOWS_FILE.exists(), index=False)
    else:
        print('  no usable windows here (too much missing data)')

# Score every saved window against Mount Sharp (computed here, so weights can be changed without downloading again)
region_centres = pd.DataFrame([(r[1], r[2]) for r in REGIONS], columns=['LAT', 'LON'])
windows_df = pd.read_csv(WINDOWS_FILE).merge(region_centres, on=['LAT', 'LON'])
# Closeness: smaller value / larger value, so 1 = identical and 0 = nothing alike
windows_df['TERRAIN_SCORE'] = sum(w * np.minimum(windows_df[k], targets[k]) / np.maximum(windows_df[k], targets[k])
                                  for k, w in WEIGHTS.items()) / 100

# Keep the best TOP_PER_REGION windows of each region, at least MIN_SEPARATION_KM apart
chosen = []
for _, group in windows_df.groupby(['LAT', 'LON']):
    picked = []
    for r in group.sort_values('TERRAIN_SCORE', ascending=False).itertuples():
        if all(km_between(r.WIN_LAT, r.WIN_LON, p.WIN_LAT, p.WIN_LON) >= MIN_SEPARATION_KM - 1 for p in picked):
            picked.append(r)
        if len(picked) == TOP_PER_REGION:
            break
    chosen += [group.loc[p.Index] for p in picked]
top_df = pd.DataFrame(chosen).reset_index(drop=True)
top_df['TERRAIN_RANK_IN_REGION'] = top_df.groupby(['LAT', 'LON'])['TERRAIN_SCORE'].rank(ascending=False, method='first').astype(int)
top_df['TERRAIN_DATA'] = ['Unverified (ASTER unreliable over ice)' if on_ice(la, lo)
                          else 'Unverified (impossible elevation)' if (lo_ < SANE_ELEV_M[0] or hi_ > SANE_ELEV_M[1])
                          else 'Checked, plausible'
                          for la, lo, lo_, hi_ in zip(top_df['LAT'], top_df['LON'], top_df['ELEV_LOW_M'], top_df['ELEV_HIGH_M'])]
print(f'\nPART 1 - {len(top_df)} windows chosen ({TOP_PER_REGION} best terrain matches per region); best terrain match in each region:')
print(top_df[top_df['TERRAIN_RANK_IN_REGION'] == 1].sort_values('TERRAIN_SCORE', ascending=False)[
    ['KIND', 'REGION_NAME', 'WIN_LAT', 'WIN_LON', 'TERRAIN_SCORE', 'ELEV_RANGE', 'SLOPE_MEAN', 'ROUGHNESS_1000M',
     'ROUGHNESS_5000M']].round(2).to_string(index=False))

# ---------------------------------------------------------------- PART 2: climate at each chosen window (NASA POWER, modelled)
climate_done = pd.read_csv(CLIMATE_FILE) if CLIMATE_FILE.exists() else pd.DataFrame()
have = set(zip(climate_done['WIN_LAT'], climate_done['WIN_LON'])) if len(climate_done) else set()
todo = [r for r in top_df.itertuples() if (r.WIN_LAT, r.WIN_LON) not in have]
print(f'\nPART 2 - climate at {len(top_df)} windows: {len(have)} already done, {len(todo)} to go (a rerun resumes where it stopped)')
for n, r in enumerate(todo, 1):
    data = polite_get(CLIM_URL, {'parameters': ','.join(PARAM_LIST), 'community': COMMUNITY,
                                 'latitude': r.WIN_LAT, 'longitude': r.WIN_LON, 'format': 'JSON'})
    par = data['properties']['parameter']
    month_df = pd.DataFrame({'MONTH': MONTHS, **{p: [par[p][m] for m in MONTHS] for p in PARAM_LIST}})
    month_df = month_df[(month_df[PARAM_LIST] > -990).all(axis=1)].copy()
    for cname, col in (('COLD', 'T2M_MIN'), ('RANGE', 'T2M_RANGE'), ('WIND', 'WS2M')):
        _, zero, full = RULES[cname]
        month_df['SCORE_' + cname] = ((month_df[col] - zero) / (full - zero)).clip(0, 1)
    month_df['SCORE'] = sum(RULES[k][0] * month_df['SCORE_' + k] for k in RULES) / sum(v[0] for v in RULES.values())
    # Best month among those with a real day-night cycle and cold enough nights; if none passes, best month overall and flagged
    passing = month_df[(month_df['T2M_RANGE'] >= MIN_RANGE_K) & (month_df['T2M_MIN'] <= MAX_NIGHT_MIN_C)]
    pool = passing if len(passing) else month_df
    m = pool.loc[pool['SCORE'].idxmax()]
    # Hourly wind shape (night/day ratio) only for windows that pass the gates: it is a gap column, not scored
    ratio = np.nan
    if len(passing):
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
        ratio = np.mean(night_means) / np.mean(day_means)
    density = m['PS'] * 1000 / (R_EARTH_AIR * (m['T2M'] + 273.15)) / MARS_DENSITY
    row = {'WIN_LAT': r.WIN_LAT, 'WIN_LON': r.WIN_LON, 'CLIMATE_ELEV_M': data['geometry']['coordinates'][2],
           'BEST_MONTH': m['MONTH'], 'GATES_PASSED': len(passing) > 0, 'T2M_MIN_C': m['T2M_MIN'], 'T2M_RANGE_K': m['T2M_RANGE'],
           'WS2M_MS': m['WS2M'], 'WIND_NIGHT_DAY_RATIO': ratio, 'DENSITY_VS_MARS': density,
           'PRECIP_MM_DAY': m['PRECTOTCORR'], 'CLIMATE_SCORE': m['SCORE']}
    pd.DataFrame([row]).to_csv(CLIMATE_FILE, mode='a', header=not CLIMATE_FILE.exists(), index=False)   # saved window by window
    print(f'  {n}/{len(todo)} {r.REGION_NAME} ({r.WIN_LAT}, {r.WIN_LON}): night min {m["T2M_MIN"]:.0f} C, range {m["T2M_RANGE"]:.0f} K, '
          f'gates {"passed" if len(passing) else "failed"}')
climate_df = pd.read_csv(CLIMATE_FILE).drop_duplicates(['WIN_LAT', 'WIN_LON'], keep='last')

# ---------------------------------------------------------------- PART 3: dust (MERRA-2, modelled), a gap column for every chosen window
print('\nPART 3 - MERRA-2 dust optical depth at the chosen windows')
all_df = top_df.merge(climate_df, on=['WIN_LAT', 'WIN_LON'])
granules = earthaccess.search_data(short_name=MERRA_SHORT_NAME, version=MERRA_VERSION, temporal=(f'{YEARS[0]}-01-01', f'{YEARS[-1]}-12-31'))
ds = xr.open_mfdataset(earthaccess.open(granules), engine='h5netcdf')
dust = ds['DUEXTTAU'].sel(lat=xr.DataArray(all_df['WIN_LAT'].values, dims='pt'), lon=xr.DataArray(all_df['WIN_LON'].values, dims='pt'),
                          method='nearest').load().groupby('time.month').mean('time').values          # (12, n)
all_df['DUST_TAU'] = [dust[MONTHS.index(mo), i] for i, mo in enumerate(all_df['BEST_MONTH'])]

# ---------------------------------------------------------------- PART 4: score, rank and validate
all_df['COMBINED_SCORE'] = CLIMATE_WEIGHT * all_df['CLIMATE_SCORE'] + (1 - CLIMATE_WEIGHT) * all_df['TERRAIN_SCORE']
all_df['GATES_OK'] = all_df['GATES_PASSED']
all_df['TERRAIN_OK'] = ~all_df['TERRAIN_DATA'].str.startswith('Unverified')
all_df['STATUS'] = np.where(~all_df['GATES_OK'], 'Fails climate gates',
                            np.where(~all_df['TERRAIN_OK'], 'Provisional (terrain unverified)', 'Eligible'))
all_df.sort_values(['COMBINED_SCORE'], ascending=False).round(3).to_csv(TOP_WINDOWS_FILE, index=False)

# One row per region: the chosen window with the best combined score (terrain first picked the windows, climate then decides)
reg = all_df.loc[all_df.groupby(['LAT', 'LON'])['COMBINED_SCORE'].idxmax()].copy()
reg = reg.sort_values(['COMBINED_SCORE', 'DENSITY_VS_MARS'], ascending=[False, True]).reset_index(drop=True)
reg['RANK_ALL'] = np.arange(1, len(reg) + 1)
cand = (reg['KIND'] == 'Candidate') & (reg['STATUS'] == 'Eligible')
reg['RANK_CANDIDATE'] = pd.Series(np.arange(1, cand.sum() + 1), index=reg.index[cand]).reindex(reg.index).astype('Int64')
reg['WHY'] = ['; '.join(x for x in (f'nights only {r.T2M_MIN_C:.0f} C' if r.T2M_MIN_C > MAX_NIGHT_MIN_C else '',
                                    f'range {r.T2M_RANGE_K:.1f} K' if r.T2M_RANGE_K < MIN_RANGE_K else '',
                                    'terrain unverified' if not r.TERRAIN_OK else '') if x) for r in reg.itertuples()]
cols = ['RANK_ALL', 'RANK_CANDIDATE', 'KIND', 'REGION_NAME', 'STATUS', 'WHY', 'WIN_LAT', 'WIN_LON', 'COMBINED_SCORE', 'CLIMATE_SCORE',
        'TERRAIN_SCORE', 'BEST_MONTH', 'T2M_MIN_C', 'T2M_RANGE_K', 'WS2M_MS', 'WIND_NIGHT_DAY_RATIO', 'DUST_TAU', 'PRECIP_MM_DAY',
        'DENSITY_VS_MARS', 'ELEV_RANGE', 'SLOPE_MEAN', 'ROUGHNESS_1000M', 'ROUGHNESS_5000M', 'TERRAIN_RANK_IN_REGION', 'TERRAIN_DATA']
reg[cols].round(3).to_csv(FINAL_FILE, index=False)

show = ['RANK_ALL', 'KIND', 'REGION_NAME', 'COMBINED_SCORE', 'CLIMATE_SCORE', 'TERRAIN_SCORE', 'T2M_MIN_C', 'T2M_RANGE_K', 'WS2M_MS',
        'DUST_TAU', 'PRECIP_MM_DAY', 'DENSITY_VS_MARS', 'STATUS']
head = {'RANK_ALL': 'Rank', 'KIND': 'Kind', 'REGION_NAME': 'Region', 'COMBINED_SCORE': 'Combined', 'CLIMATE_SCORE': 'Climate',
        'TERRAIN_SCORE': 'Terrain', 'T2M_MIN_C': 'Night min (C)', 'T2M_RANGE_K': 'Range (K)', 'WS2M_MS': 'Wind (m/s)',
        'DUST_TAU': 'Dust', 'PRECIP_MM_DAY': 'Rain (mm/d)', 'DENSITY_VS_MARS': 'Density (x Mars)', 'STATUS': 'Status'}
dec = {'COMBINED_SCORE': 2, 'CLIMATE_SCORE': 2, 'TERRAIN_SCORE': 2, 'T2M_MIN_C': 1, 'T2M_RANGE_K': 1, 'WS2M_MS': 1, 'DUST_TAU': 2,
       'PRECIP_MM_DAY': 2, 'DENSITY_VS_MARS': 0}


def table(df, columns=show):
    out = df[columns].copy()
    for c in columns:
        if c in dec:
            out[c] = out[c].map(lambda v: f'{v:.{dec[c]}f}' if pd.notna(v) else '')
    print(out.rename(columns=head).fillna('').to_string(index=False))


def heading(text):
    print(f'\n{text}\n{"-" * len(text)}')


print(f'\nMars (Gale) for reference: night min {MARS_COLD_C} C, range {MARS_RANGE_K} K, wind {MARS_WIND_MS} m/s, dust {MARS_DUST}, '
      f'rain 0, density 1x; Mount Sharp targets {({k: round(v, 1) for k, v in targets.items()})}')

heading('TOP 10 CANDIDATES (eligible: pass the gates, verified terrain)')
table(reg[cand].head(10))
heading('CANDIDATES NOT RANKED')
table(reg[(reg['KIND'] == 'Candidate') & ~cand].head(40), show + ['WHY'])

heading('FULL COMPARISON: candidates, analogues and controls on the same rules (all regions, best combined score first)')
table(reg)

# ---------------------------------------------------------------- Validation of the method
heading('VALIDATION: does the scoring behave sensibly on places we already know?')
analogues, controls = reg[reg['KIND'] == 'Analogue'], reg[reg['KIND'] == 'Control']
v1 = int(controls['GATES_OK'].sum()) == 0
v2 = analogues['COMBINED_SCORE'].median() > controls['COMBINED_SCORE'].median()
print(f'V1 no control passes the climate gates: {"PASS" if v1 else "FAIL"} ({int(controls["GATES_OK"].sum())} of {len(controls)} controls pass)')
print(f'V2 analogues beat controls (median combined score): {"PASS" if v2 else "FAIL"} '
      f'(analogues {analogues["COMBINED_SCORE"].median():.2f}, controls {controls["COMBINED_SCORE"].median():.2f})')
print(f'   analogues passing the gates: {int(analogues["GATES_OK"].sum())} of {len(analogues)}; '
      f'candidates passing the gates: {int(reg[reg["KIND"] == "Candidate"]["GATES_OK"].sum())} of {int((reg["KIND"] == "Candidate").sum())}')
best = reg[cand].head(1)
if len(best):
    b = best.iloc[0]
    beaten = analogues[analogues['COMBINED_SCORE'] < b['COMBINED_SCORE'] - NEAR_TIE]
    print(f'Best candidate ({b["REGION_NAME"]}, {b["COMBINED_SCORE"]:.2f}) beats {len(beaten)} of {len(analogues)} analogues: '
          f'{", ".join(beaten["REGION_NAME"]) if len(beaten) else "none"}')
print('\nHow each analogue and control scored (rank out of all regions):')
table(pd.concat([analogues, controls]), ['RANK_ALL', 'KIND', 'REGION_NAME', 'COMBINED_SCORE', 'CLIMATE_SCORE', 'TERRAIN_SCORE', 'STATUS'])
print('\nReading the validation: an analogue is chosen for one Mars property (dust, dryness, cold, geology), so it need not match all')
print('of Gale. What matters is that analogues sit well above the controls and that the best candidate is competitive with them.')
print('Coastal windows include sea level in their relief; the Dry Valleys terrain is ice-adjacent (ASTER unverified, REMA agreed there earlier).')
print(f'\nSaved earth_b2_windows.csv and earth_b2_climate.csv in data/raw; earth_b2_regions.csv and earth_b2_top_windows.csv in data/clean')