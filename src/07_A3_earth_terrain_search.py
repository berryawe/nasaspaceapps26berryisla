import re
import zipfile
import tempfile
import warnings
import numpy as np
import pandas as pd
import rasterio
import earthaccess
from scipy.ndimage import uniform_filter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Defining constants
RAW_DIR = BASE_DIR / 'data' / 'raw'
CLIMATE_FILE = RAW_DIR / 'earth_climate_final.csv'         # climate shortlist from the climate script
MARS_TERRAIN_FILE = RAW_DIR / 'terrain_mount_sharp.csv'    # Mount Sharp indicators from the Mars terrain script
WINDOWS_FILE = RAW_DIR / 'earth_terrain_windows.csv'       # indicators of every window; saved site by site, so a rerun resumes
ELEV_FILE = RAW_DIR / 'earth_terrain_elevation.csv'        # elevation of every search box at about 1 km (plain CSV for checking and maps)
BEST_FILE = RAW_DIR / 'earth_terrain_best.csv'             # best window at each site, with scores

N_ANTARCTIC, N_OTHER = 4, 3                  # shortlist sites taken from each climate list (best climate score first)
FORCE_INCLUDE = [(30.0, 85.0)]               # extra sites always searched (the Tibetan plateau site)
SEARCH_HALF_KM = 100.0                       # search box around each site = a square of 2 x this width
WINDOW_KM = 80.0                             # Mount Sharp comparison window (same 80 km square as the Mars terrain script)
STRIDE_KM = 20.0                             # window step
TARGET_RES_M = 200.0                         # cell size used for slope/roughness (same as the Mars DEM)
SAVE_RES_KM = 1.0                            # cell size of the elevation table that is saved
ROUGHNESS_WINDOWS_M = (1000, 5000)
MIN_VALID_SHARE = 0.8                        # skip windows with more than 20% missing elevation
PCT_LOW, PCT_HIGH = 0.5, 99.5                # relief = these percentiles, so single-pixel DEM spikes don't count
ASTER_SHORT_NAME, ASTER_VERSION = 'ASTGTM', '003'   # NASA/METI ASTER Global DEM v3, 1 arc-second (~30 m), 83N to 83S
ASTER_CELL_M = 30.87                         # 1 arc-second of latitude
M_PER_DEG = 111195.0                         # Earth sphere
TILE_PIXELS = 3600                           # cells per 1-degree tile side (the extra overlap row/column is dropped)

# Terrain score: closeness to Mount Sharp on each indicator, weight out of 100
WEIGHTS = {'ELEV_RANGE': 30, 'SLOPE_MEAN': 25, 'ROUGHNESS_1000M': 20, 'ROUGHNESS_5000M': 25}
CLIMATE_WEIGHT = 0.5                         # combined score = this x climate score + (1 - this) x terrain score

# Mount Sharp targets
mars_df = pd.read_csv(MARS_TERRAIN_FILE).set_index('INDICATOR')['VALUE']
targets = {k: float(mars_df[k]) for k in WEIGHTS}

# Sites: best Antarctic and non-Antarctic climate sites, plus the forced ones
climate_df = pd.read_csv(CLIMATE_FILE)
sites = pd.concat([climate_df[climate_df['REGION'] == 'ANTARCTIC'].nlargest(N_ANTARCTIC, 'FINAL_SCORE'),
                   climate_df[climate_df['REGION'] == 'OTHER'].nlargest(N_OTHER, 'FINAL_SCORE'),
                   climate_df[[(la, lo) in FORCE_INCLUDE for la, lo in zip(climate_df['LAT'], climate_df['LON'])]]])
sites = sites.drop_duplicates(['LAT', 'LON']).reset_index(drop=True)
print(f'{len(sites)} sites to search, {SEARCH_HALF_KM * 2:.0f} km box each, {WINDOW_KM:.0f} km windows')

# Sites already finished in an earlier run are skipped
done = set()
if WINDOWS_FILE.exists():
    prev = pd.read_csv(WINDOWS_FILE)
    done = set(zip(prev['LAT'], prev['LON']))
earthaccess.login(persist=True)


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


for site in sites.itertuples():
    lat, lon = site.LAT, site.LON
    if (lat, lon) in done:
        print(f'\nSite {lat}, {lon}: already done, skipped')
        continue
    # Search box in degrees
    half_lat = SEARCH_HALF_KM * 1000 / M_PER_DEG
    half_lon = half_lat / np.cos(np.radians(lat))
    south, north, west, east = lat - half_lat, lat + half_lat, lon - half_lon, lon + half_lon
    if west < -180 or east >= 180:
        raise RuntimeError(f'Search box at {lat}, {lon} crosses the date line - not handled')
    # Tiles needed, and how much to average each so cells are about TARGET_RES_M in both directions
    fy = max(1, round(TARGET_RES_M / ASTER_CELL_M))
    fx = max(1, round(TARGET_RES_M / (ASTER_CELL_M * np.cos(np.radians(lat)))))
    lat_tiles = list(range(int(np.floor(north)), int(np.floor(south)) - 1, -1))      # north to south
    lon_tiles = list(range(int(np.floor(west)), int(np.floor(east)) + 1))            # west to east
    names = {(la, lo): tile_name(la, lo) for la in lat_tiles for lo in lon_tiles}
    ny, nx = TILE_PIXELS // fy, TILE_PIXELS // fx
    print(f'\nSite {lat}, {lon}: {len(names)} tiles needed')

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
        raise RuntimeError(f'No elevation data for the site at {lat}, {lon} - the tile download did not work')
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
    site_windows = []
    for a in range(0, z.shape[0] - wy + 1, sy):
        for b in range(0, z.shape[1] - wx + 1, sx):
            zw = z[a:a + wy, b:b + wx]
            if np.isfinite(zw).mean() < MIN_VALID_SHARE:
                continue
            c_lat, c_lon = row_lat[a + wy // 2], col_lon[b + wx // 2]
            dist_km = np.hypot((c_lat - lat) * M_PER_DEG, (c_lon - lon) * M_PER_DEG * np.cos(np.radians(lat))) / 1000
            site_windows.append({'LAT': lat, 'LON': lon, 'REGION': site.REGION, 'CLIMATE_SCORE': site.FINAL_SCORE,
                                 'WIN_LAT': round(float(c_lat), 3), 'WIN_LON': round(float(c_lon), 3),
                                 'DIST_FROM_SITE_KM': round(float(dist_km), 1),
                                 'ELEV_RANGE': float(np.nanpercentile(zw, PCT_HIGH) - np.nanpercentile(zw, PCT_LOW)),
                                 'SLOPE_MEAN': float(np.nanmean(slope[a:a + wy, b:b + wx])),
                                 'ROUGHNESS_1000M': float(np.nanmean(rough[1000][a:a + wy, b:b + wx])),
                                 'ROUGHNESS_5000M': float(np.nanmean(rough[5000][a:a + wy, b:b + wx]))})
    print(f'  {len(site_windows)} windows measured')

    # Save: elevation at about 1 km (plain CSV), then the windows (this marks the site as done)
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
    if site_windows:
        pd.DataFrame(site_windows).to_csv(WINDOWS_FILE, mode='a', header=not WINDOWS_FILE.exists(), index=False)
    else:
        print('  no usable windows here (too much missing data, common over ice sheets)')

# Score every saved window against Mount Sharp (computed here, so weights can be changed without downloading again)
windows_df = pd.read_csv(WINDOWS_FILE).merge(sites[['LAT', 'LON']], on=['LAT', 'LON'])
if windows_df.empty:
    raise RuntimeError('No windows had enough elevation data')
# Closeness: smaller value / larger value, so 1 = identical and 0 = nothing alike
windows_df['TERRAIN_SCORE'] = sum(w * np.minimum(windows_df[k], targets[k]) / np.maximum(windows_df[k], targets[k])
                                  for k, w in WEIGHTS.items()) / 100

# Best window for each site, then combine with climate
best_df = windows_df.loc[windows_df.groupby(['LAT', 'LON'])['TERRAIN_SCORE'].idxmax()].copy()
best_df['COMBINED_SCORE'] = CLIMATE_WEIGHT * best_df['CLIMATE_SCORE'] + (1 - CLIMATE_WEIGHT) * best_df['TERRAIN_SCORE']
best_df = best_df.sort_values('COMBINED_SCORE', ascending=False).reset_index(drop=True)

# Summarise + save
print('\nMount Sharp targets:', {k: round(v, 1) for k, v in targets.items()})
print('\nBEST TERRAIN WINDOW AT EACH SITE (best combined score first):')
print(best_df[['LAT', 'LON', 'REGION', 'CLIMATE_SCORE', 'TERRAIN_SCORE', 'COMBINED_SCORE', 'WIN_LAT', 'WIN_LON',
               'DIST_FROM_SITE_KM', 'ELEV_RANGE', 'SLOPE_MEAN', 'ROUGHNESS_1000M', 'ROUGHNESS_5000M']].round(3).to_string(index=False))
best_df.to_csv(BEST_FILE, index=False)
print('Earth terrain search complete')