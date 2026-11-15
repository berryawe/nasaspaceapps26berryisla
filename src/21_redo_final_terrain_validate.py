import os
import getpass
import tempfile
import requests
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds
from scipy.ndimage import uniform_filter

BASE_DIR = Path(__file__).resolve().parent.parent

# Validate the ArcticDEM 2 m tiles (5 km at 2 m and 25 km at 5 m) against independent elevation models:
#   Copernicus GLO-30 (radar), ALOS AW3D30 (optical stereo), the ASTER tile used before (5 km only) and, if OpenTopography has it for this area,
#   USGS 3DEP 10 m (made from airborne radar in Alaska). None of them are ArcticDEM, so they are independent of the tile.
# The 30 m models cannot confirm 2 m detail, so every model is compared on a common coarser grid (20 m for the 5 km tile, 50 m for the 25 km tile):
# elevation difference, then slope, relief and roughness at the Mars 200 m scale.
# Defining constants
CLEAN_DIR = BASE_DIR / 'data' / 'clean'
TILES = [('dem_best_terrain_alaska_range_5km_2m.tif', 20.0), ('dem_best_terrain_alaska_range_25km_5m.tif', 50.0)]   # (tile file, common grid cell in m)
ASTER_TILE = CLEAN_DIR / 'dem_best_terrain_alaska_range_5km.tif'
OUT_FILE = CLEAN_DIR / 'dem_validation_2m.csv'
OT_URL = 'https://portal.opentopography.org/API/globaldem'         # OpenTopography global DEM API (free API key)
OT_SOURCES = [('COP30', 'Copernicus GLO-30 (radar, 30 m)'), ('AW3D30', 'ALOS AW3D30 (optical stereo, 30 m)')]
USGS_URL = 'https://portal.opentopography.org/API/usgsdem'         # OpenTopography USGS 3DEP API (same key)
USGS_DATASET = 'USGS10m'
USGS_LABEL = 'USGS 3DEP 10 m (airborne radar in Alaska)'
PAD_DEG = 0.01
SEARCH_M = 200.0                                                   # the scale of the Mount Sharp numbers
ROUGHNESS_M = 1000
NODATA = -9999.0
OUTLIER_M = 50.0
# Agreement thresholds, the same as for the ASTER tile
MAX_NMAD_M = 20.0
MAX_SLOPE_DIFF_DEG = 3.0
MAX_RELIEF_DIFF = 0.15
MAX_OUTLIER_SHARE = 0.05


def block_mean(a, f):
    n = a.shape[0] // f * f
    with np.errstate(all='ignore'):
        return np.nanmean(a[:n, :n].reshape(n // f, f, n // f, f), axis=(1, 3))


def measure(z, cell):
    """Slope, relief and roughness at the 200 m scale, tilt removed."""
    f = max(1, int(round(SEARCH_M / cell)))
    zc = block_mean(z, f)
    res = cell * f
    zc = np.where(np.isfinite(zc), zc, np.nanmean(zc))
    gy, gx = np.gradient(zc, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    n = zc.shape[0]
    yy, xx = np.mgrid[0:n, 0:n] * res
    design = np.column_stack([xx.ravel(), yy.ravel(), np.ones(n * n)])
    resid = zc - (design @ np.linalg.lstsq(design, zc.ravel(), rcond=None)[0]).reshape(n, n)
    k = max(3, int(round(ROUGHNESS_M / res)))
    local_sd = np.sqrt(np.maximum(uniform_filter(resid ** 2, k) - uniform_filter(resid, k) ** 2, 0))
    low, high = np.percentile(resid, [0.5, 99.5])
    return {'SLOPE_MEAN': slope.mean(), 'SLOPE_P95': np.percentile(slope, 95), 'RELIEF_M': high - low, 'ROUGHNESS_M': local_sd.mean()}


def to_grid(sources, crs, transform, shape):
    """Mosaic the source rasters and resample them onto the common grid. Missing cells are NaN."""
    bounds = rasterio.transform.array_bounds(shape[0], shape[1], transform)
    box = transform_bounds(crs, sources[0].crs, *bounds, densify_pts=21)
    pad = 3 * max(sources[0].res)
    mosaic, mosaic_transform = merge(sources, bounds=(box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), nodata=NODATA)
    src = mosaic[0].astype('float32')
    src[(src <= -500) | ~np.isfinite(src)] = np.nan
    out = np.full(shape, np.nan, dtype='float32')
    reproject(src, out, src_transform=mosaic_transform, src_crs=sources[0].crs, dst_transform=transform, dst_crs=crs,
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def download(url, params, label, crs, transform, shape):
    """Download a DEM for the grid's bounding box from OpenTopography and put it on the grid."""
    west, south, east, north = transform_bounds(crs, 'EPSG:4326', *rasterio.transform.array_bounds(shape[0], shape[1], transform))
    params = dict(params, south=south - PAD_DEG, north=north + PAD_DEG, west=west - PAD_DEG, east=east + PAD_DEG, outputFormat='GTiff')
    r = requests.get(url, params=params, timeout=300)
    if r.status_code in (401, 403):
        raise RuntimeError('OpenTopography rejected the API key')
    if r.status_code != 200 or r.content[:4] not in (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+'):
        raise RuntimeError(f'{label}: status {r.status_code}, {r.text[:150]!r}')
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'dem.tif'
        path.write_bytes(r.content)
        with rasterio.open(path) as src:
            return to_grid([src], crs, transform, shape)


key = os.environ.get('OPENTOPO_API_KEY') or getpass.getpass('OpenTopography API key (not saved): ')
all_rows = []
for tile_name, cell in TILES:
    tile_file = CLEAN_DIR / tile_name
    if not tile_file.exists():
        print(f'\n{tile_name}: not found in data/clean, skipping')
        continue
    with rasterio.open(tile_file) as tile:
        z = tile.read(1).astype('float32')
        z[z == tile.nodata] = np.nan
        crs, left, bottom, right, top = tile.crs, *tile.bounds
        tile_transform, tile_cell = tile.transform, tile.res[0]
    n = int(round((right - left) / cell))
    transform = from_origin(left, top, cell, cell)
    shape = (n, n)
    print(f'\n=== {tile_name}: {z.shape[1]} x {z.shape[0]} cells of {tile_cell:g} m, compared on a {n} x {n} grid of {cell:g} m ===')

    ref = np.full(shape, np.nan, dtype='float32')                    # the tile averaged down to the common grid
    reproject(z, ref, src_transform=tile_transform, src_crs=crs, dst_transform=transform, dst_crs=crs,
              resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)

    sources, skipped = {}, {}
    for demtype, label in OT_SOURCES:
        print(f'fetching {label} ...')
        sources[label] = download(OT_URL, {'demtype': demtype, 'API_Key': key}, label, crs, transform, shape)
    print(f'fetching {USGS_LABEL} ...')
    try:
        sources[USGS_LABEL] = download(USGS_URL, {'datasetName': USGS_DATASET, 'API_Key': key}, USGS_LABEL, crs, transform, shape)
    except Exception as e:
        skipped[USGS_LABEL] = f'Not available: {str(e)[:90]}'
    if ASTER_TILE.exists():
        with rasterio.open(ASTER_TILE) as a:
            az = a.read(1).astype('float32')
            az[az == a.nodata] = np.nan
            out = np.full(shape, np.nan, dtype='float32')
            reproject(az, out, src_transform=a.transform, src_crs=a.crs, dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
            sources['ASTER GDEM v3 (the earlier tile, 25 m)'] = out

    base = measure(ref, cell)
    rows = [{'TILE': tile_name, 'SOURCE': f'ArcticDEM 2 m tile (reference, averaged to {cell:g} m)', 'VALID': 1.0, 'OFFSET_M': 0.0, 'NMAD_M': 0.0,
             'RMSE_M': 0.0, 'OUTLIERS': 0.0, 'CORRELATION': 1.0, **base, 'VERDICT': 'reference'}]
    for label, other in sources.items():
        both = np.isfinite(ref) & np.isfinite(other)
        if both.mean() < 0.5:
            rows.append({'TILE': tile_name, 'SOURCE': label, 'VALID': both.mean(), 'VERDICT': 'Not enough overlap to compare'})
            continue
        diff = other[both] - ref[both]
        offset = np.median(diff)                                     # vertical reference and survey-date differences; removed below
        centred = diff - offset
        m = measure(other, cell)
        nmad = 1.4826 * np.median(np.abs(centred))
        outliers = np.mean(np.abs(centred) > OUTLIER_M)
        ok = nmad <= MAX_NMAD_M and outliers <= MAX_OUTLIER_SHARE and abs(m['SLOPE_MEAN'] - base['SLOPE_MEAN']) <= MAX_SLOPE_DIFF_DEG \
            and abs(m['RELIEF_M'] - base['RELIEF_M']) <= MAX_RELIEF_DIFF * base['RELIEF_M']
        rows.append({'TILE': tile_name, 'SOURCE': label, 'VALID': both.mean(), 'OFFSET_M': offset, 'NMAD_M': nmad, 'RMSE_M': np.sqrt(np.mean(centred ** 2)),
                     'OUTLIERS': outliers, 'CORRELATION': np.corrcoef(ref[both], other[both])[0, 1], **m,
                     'VERDICT': 'Agrees with the tile' if ok else 'Disagrees with the tile'})
    for label, why in skipped.items():
        rows.append({'TILE': tile_name, 'SOURCE': label, 'VERDICT': why})
    result = pd.DataFrame(rows)
    all_rows.append(result)

    show = result.drop(columns='TILE').rename(columns={'VALID': 'Overlap', 'OFFSET_M': 'Offset (m)', 'NMAD_M': 'NMAD (m)', 'RMSE_M': 'RMSE (m)',
                                                       'OUTLIERS': f'>{OUTLIER_M:.0f} m off', 'CORRELATION': 'Correlation', 'SLOPE_MEAN': 'Slope mean (deg)',
                                                       'SLOPE_P95': 'Slope p95 (deg)', 'RELIEF_M': 'Relief (m)', 'ROUGHNESS_M': 'Roughness (m)'})
    for c in ('Overlap', f'>{OUTLIER_M:.0f} m off'):
        show[c] = show[c].map(lambda v: f'{v:.0%}' if pd.notna(v) else '')
    for c in ('Offset (m)', 'NMAD (m)', 'RMSE (m)', 'Relief (m)', 'Roughness (m)', 'Slope mean (deg)', 'Slope p95 (deg)'):
        show[c] = show[c].map(lambda v: f'{v:.1f}' if pd.notna(v) else '')
    show['Correlation'] = show['Correlation'].map(lambda v: f'{v:.3f}' if pd.notna(v) else '')
    print(show.fillna('').to_string(index=False))
    agree = int((result['VERDICT'] == 'Agrees with the tile').sum())
    print(f'{agree} of {len(sources)} independent sources agree with this tile.')

print(f'\n(agree = difference spread (NMAD) <= {MAX_NMAD_M:.0f} m, no more than {MAX_OUTLIER_SHARE:.0%} of cells {OUTLIER_M:.0f} m off, mean slope within {MAX_SLOPE_DIFF_DEG:.0f} deg,'
      f' relief within {MAX_RELIEF_DIFF:.0%}; offsets are removed because of different vertical references and survey dates)')
print('The 30 m models cannot confirm detail finer than their own resolution, so this check covers the large-scale shape, slope, relief and roughness of the tile.')
if all_rows:
    pd.concat(all_rows).round(3).to_csv(OUT_FILE, index=False)
    print(f'\nSaved {OUT_FILE.name} in data/clean')