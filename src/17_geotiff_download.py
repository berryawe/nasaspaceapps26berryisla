import re
import tempfile
import numpy as np
import pandas as pd
import rasterio
import earthaccess
from pathlib import Path
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform, transform_bounds
from scipy.ndimage import uniform_filter

BASE_DIR = Path(__file__).resolve().parent.parent

# GeoTIFF elevation model (metres) of a 5 km square: the most rugged 5 km part of the best terrain match's window that is still no steeper
# than Mars. "Rugged" = relief and roughness after removing the overall tilt, plus mean slope; the square must stay within Mount Sharp's slopes.
# In the local UTM zone (metres), so it can be used directly for the SolidWorks terrain mesh.
# Defining constants
CLEAN_DIR = BASE_DIR / 'data' / 'clean'
RAW_DIR = BASE_DIR / 'data' / 'raw'
PICKS_FILE = CLEAN_DIR / 'final_results.csv'  
MARS_TERRAIN_FILE = RAW_DIR / 'terrain_mount_sharp.csv'                 # Mount Sharp indicators from the Mars terrain script
WINDOW_KM = 80.0                                  # the window the site was scored on (the part is searched inside it)
TILE_KM = 5.0                                     # side of the GeoTIFF
STRIDE_KM = 1.0                                   # step between the 5 km squares tried
GRID_RES_M = 25.0                                 # saved grid. ASTER is about 30 m; 25 m makes 8 cells = 200 m (the Mars DEM cell) and 200 cells = 5 km
SEARCH_FACTOR = 8                                 # squares are judged on a grid 8 x coarser = 200 m cells, the same scale as the Mount Sharp numbers
NODATA = -9999.0
MIN_VALID_SHARE = 0.95                            # skip squares with more than 5% missing elevation
PCT_LOW, PCT_HIGH = 0.5, 99.5                     # relief = these percentiles, so single-pixel DEM spikes don't count
ROUGHNESS_M = 1000                                # roughness = local standard deviation of elevation in a window this wide
MAX_MEAN_SLOPE_FROM = 'SLOPE_P95'                 # a square's mean slope may not exceed Mount Sharp's 95th-percentile slope (steepest 5% of Mount Sharp)
MAX_P99_SLOPE_FROM = 'SLOPE_MAX'                  # a square's own 99th-percentile slope may not exceed Mount Sharp's steepest slope (no cliffs)
WEIGHTS = {'RELIEF_M': 1, 'SLOPE_MEAN': 1, 'ROUGHNESS_M': 1}   # equal weight; each is divided by its best value among the allowed squares
ASTER_SHORT_NAME, ASTER_VERSION = 'ASTGTM', '003'  # NASA/METI ASTER GDEM v3, 1 arc-second (~30 m)


def utm_crs(lat, lon):
    """WGS 84 / UTM zone of a point."""
    zone = int((lon + 180) // 6) + 1
    return f'EPSG:{32600 + zone if lat >= 0 else 32700 + zone}'


def block_mean(a, f):
    """Average f x f blocks (ignoring missing cells)."""
    n = a.shape[0] // f * f
    with np.errstate(all='ignore'):
        blocks = a[:n, :n].reshape(n // f, f, n // f, f)
        return np.nanmean(blocks, axis=(1, 3))


# ---------------------------------------------------------------- Read the best terrain match and the Mount Sharp limits
picks = pd.read_csv(PICKS_FILE).set_index('Indicator')
column = [c for c in picks.columns if c.startswith('Best terrain match')][0]
place = picks.loc['Place', column]
lat, lon = float(picks.loc['Best window latitude', column]), float(picks.loc['Best window longitude', column])
if lat <= -60 or (59 <= lat <= 84 and -74 <= lon <= -11):
    raise RuntimeError(f'{place} is on ice, where ASTER is unreliable - this script only handles ASTER terrain')
mars = pd.read_csv(MARS_TERRAIN_FILE).set_index('INDICATOR')['VALUE']
max_mean_slope, max_p99_slope = float(mars[MAX_MEAN_SLOPE_FROM]), float(mars[MAX_P99_SLOPE_FROM])
print(f'Best terrain match: {place} ({lat:.3f}, {lon:.3f})')
print(f'Mount Sharp limits: a square\'s mean slope <= {max_mean_slope:.1f} deg and its 99th-percentile slope <= {max_p99_slope:.1f} deg')

# ---------------------------------------------------------------- Elevation of the whole window on an exact grid (UTM, metres)
crs = utm_crs(lat, lon)
cx, cy = (v[0] for v in transform('EPSG:4326', crs, [lon], [lat]))
half = WINDOW_KM * 1000 / 2
size = int(round(WINDOW_KM * 1000 / GRID_RES_M))
win_transform = from_origin(cx - half, cy + half, GRID_RES_M, GRID_RES_M)
west, south, east, north = transform_bounds(crs, 'EPSG:4326', cx - half, cy - half, cx + half, cy + half, densify_pts=21)

earthaccess.login(persist=True)
granules = earthaccess.search_data(short_name=ASTER_SHORT_NAME, version=ASTER_VERSION, bounding_box=(west, south, east, north))
urls = sorted({u for g in granules for u in g.data_links() if u.endswith('_dem.tif')})
print(f'downloading {len(urls)} ASTER tiles')
with tempfile.TemporaryDirectory() as tmp:
    files = earthaccess.download(urls, tmp)
    if not urls or len(files) < len(urls):
        raise RuntimeError(f'Only {len(files)} of {len(urls)} ASTER tiles downloaded - check the Earthdata login and the LP DAAC approval')
    sources = [rasterio.open(f) for f in files]
    mosaic, mosaic_transform = merge(sources, bounds=(west - 0.01, south - 0.01, east + 0.01, north + 0.01), nodata=NODATA)
    mosaic = mosaic[0].astype('float32')
    mosaic[mosaic <= -9000] = NODATA
    z = np.full((size, size), NODATA, dtype='float32')
    reproject(mosaic, z, src_transform=mosaic_transform, src_crs=sources[0].crs, dst_transform=win_transform, dst_crs=crs,
              resampling=Resampling.bilinear, src_nodata=NODATA, dst_nodata=NODATA)
    for s in sources:
        s.close()
z[z == NODATA] = np.nan
if not np.isfinite(z).any():
    raise RuntimeError('No elevation data in the window - the download did not cover it')
print(f'window elevation grid {size} x {size} cells of {GRID_RES_M:.0f} m, {np.isfinite(z).mean():.0%} valid')

# ---------------------------------------------------------------- Judge every 5 km square at the Mount Sharp scale (200 m cells)
coarse_res = GRID_RES_M * SEARCH_FACTOR
zc = block_mean(z, SEARCH_FACTOR)
gy, gx = np.gradient(zc, coarse_res)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
tile_c = int(round(TILE_KM * 1000 / coarse_res))          # square side in coarse cells
step_c = int(round(STRIDE_KM * 1000 / coarse_res))
k = max(3, int(round(ROUGHNESS_M / coarse_res)))
yy, xx = np.mgrid[0:tile_c, 0:tile_c] * coarse_res
design = np.column_stack([xx.ravel(), yy.ravel(), np.ones(tile_c * tile_c)])

rows = []
for a in range(0, zc.shape[0] - tile_c + 1, step_c):
    for b in range(0, zc.shape[1] - tile_c + 1, step_c):
        zt = zc[a:a + tile_c, b:b + tile_c]
        if np.isfinite(zt).mean() < MIN_VALID_SHARE:
            continue
        zt = np.where(np.isfinite(zt), zt, np.nanmean(zt))
        # remove the overall tilt: subtract the best-fitting plane, so relief and roughness measure the rugged surface, not a sloping mountainside
        coef = np.linalg.lstsq(design, zt.ravel(), rcond=None)[0]
        resid = zt - (design @ coef).reshape(tile_c, tile_c)
        low, high = np.percentile(resid, [PCT_LOW, PCT_HIGH])
        local_sd = np.sqrt(np.maximum(uniform_filter(resid ** 2, k) - uniform_filter(resid, k) ** 2, 0))
        st = slope[a:a + tile_c, b:b + tile_c]
        rows.append({'ROW': a, 'COL': b, 'RELIEF_M': high - low, 'ROUGHNESS_M': local_sd.mean(), 'SLOPE_MEAN': st.mean(),
                     'SLOPE_P95': np.percentile(st, 95), 'SLOPE_P99': np.percentile(st, 99), 'RELIEF_RAW_M': np.percentile(zt, PCT_HIGH) - np.percentile(zt, PCT_LOW),
                     'TILT_DEG': np.degrees(np.arctan(np.hypot(coef[0], coef[1])))})
squares = pd.DataFrame(rows)
if squares.empty:
    raise RuntimeError('No 5 km square had enough elevation data')
allowed = squares[(squares['SLOPE_MEAN'] <= max_mean_slope) & (squares['SLOPE_P99'] <= max_p99_slope)].copy()
print(f'{len(squares)} squares of {TILE_KM:.0f} km tried (every {STRIDE_KM:.0f} km); {len(allowed)} are within the Mount Sharp slope limits; '
      f'steepest square in the window has mean slope {squares["SLOPE_MEAN"].max():.0f} deg')
if allowed.empty:
    raise RuntimeError('No square is within the Mount Sharp slope limits - raise the limits in the constants or use another site')
# Score: each indicator as a share of its best value among the allowed squares, weighted
allowed['SCORE'] = sum(w * allowed[c] / allowed[c].max() for c, w in WEIGHTS.items()) / sum(WEIGHTS.values())
best = allowed.loc[allowed['SCORE'].idxmax()]
ci, cj = int(best['ROW']), int(best['COL'])
a, b = ci * SEARCH_FACTOR, cj * SEARCH_FACTOR            # the same square on the saved grid
tile = int(round(TILE_KM * 1000 / GRID_RES_M))

# ---------------------------------------------------------------- Save the best 5 km square
dem = np.where(np.isfinite(z[a:a + tile, b:b + tile]), z[a:a + tile, b:b + tile], NODATA).astype('float32')
tile_transform = win_transform * win_transform.translation(b, a)
tx, ty = tile_transform.c + tile / 2 * GRID_RES_M, tile_transform.f - tile / 2 * GRID_RES_M     # centre of the square
t_lon, t_lat = (v[0] for v in transform(crs, 'EPSG:4326', [tx], [ty]))
t_west, t_south, t_east, t_north = transform_bounds(crs, 'EPSG:4326', tx - TILE_KM * 500, ty - TILE_KM * 500, tx + TILE_KM * 500, ty + TILE_KM * 500)
out_file = CLEAN_DIR / f'dem_best_terrain_{re.sub(r"[^a-z0-9]+", "_", place.lower()).strip("_")}_{TILE_KM:.0f}km.tif'
with rasterio.open(out_file, 'w', driver='GTiff', height=tile, width=tile, count=1, dtype='float32', crs=crs, transform=tile_transform,
                   nodata=NODATA, compress='deflate') as dst:
    dst.write(dem, 1)
    dst.update_tags(DESCRIPTION='ASTER GDEM v3 (NASA/METI), heights above the EGM96 geoid, resampled to 25 m', UNITS='metres',
                    TILE_CENTRE=f'{t_lat:.5f}, {t_lon:.5f}')

valid = dem != NODATA
print(f'\nSaved {out_file.name} in data/clean')
print(f'  {tile} x {tile} cells of {GRID_RES_M:.0f} m = {TILE_KM:.0f} km square, CRS {crs} (metres), {valid.mean():.0%} valid')
print(f'  centre {abs(t_lat):.5f}°{"N" if t_lat >= 0 else "S"} {abs(t_lon):.5f}°{"E" if t_lon >= 0 else "W"}  |  '
      f'S {t_south:.4f}  N {t_north:.4f}  W {t_west:.4f}  E {t_east:.4f} (degrees)')
print(f'  elevation {dem[valid].min():.0f} to {dem[valid].max():.0f} m')
print(f'\n{"":<34}{"this 5 km square":>18}{"Mount Sharp (80 km)":>21}')
print(f'{"Mean slope (deg)":<34}{best["SLOPE_MEAN"]:>18.1f}{float(mars["SLOPE_MEAN"]):>21.1f}')
print(f'{"95th-percentile slope (deg)":<34}{best["SLOPE_P95"]:>18.1f}{float(mars["SLOPE_P95"]):>21.1f}')
print(f'{"Overall tilt of the square (deg)":<34}{best["TILT_DEG"]:>18.1f}{"":>21}')
print(f'{"Relief after removing tilt (m)":<34}{best["RELIEF_M"]:>18.0f}{"":>21}')
print(f'{"Relief, raw (m)":<34}{best["RELIEF_RAW_M"]:>18.0f}{float(mars["ELEV_RANGE"]):>21.0f}')
print(f'{"Roughness after removing tilt (m)":<34}{best["ROUGHNESS_M"]:>18.1f}{float(mars["ROUGHNESS_1000M"]):>21.1f}')
print('\n(Slopes are measured on 200 m cells, the same scale as the Mars DEM. A 5 km square cannot show 5 km of relief, so compare slope and')
print('roughness with Mount Sharp, not relief. Look at the file in QGIS before building a mesh from it: ASTER is unreliable over glaciers.)')