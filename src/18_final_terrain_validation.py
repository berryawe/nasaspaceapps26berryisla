import os
import getpass
import tempfile
import requests
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from pystac_client import Client
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, transform_bounds
from scipy.ndimage import uniform_filter

BASE_DIR = Path(__file__).resolve().parent.parent

# Validate the 5 km terrain tile (built from ASTER) against three independent elevation models:
#   Copernicus GLO-30 (radar, TanDEM-X), ALOS AW3D30 (optical stereo) and ArcticDEM (PGC stereo photogrammetry).
# Each is resampled onto the tile's own grid, then compared on elevation, slope (at the Mars 200 m scale), relief and roughness.
# Defining constants
CLEAN_DIR = BASE_DIR / 'data' / 'clean'
TILE_FILE = CLEAN_DIR / 'dem_best_terrain_alaska_range_5km.tif'   # the tile from earth_pick_dems.py
OUT_FILE = CLEAN_DIR / 'dem_validation.csv'                        # one row per source
OT_URL = 'https://portal.opentopography.org/API/globaldem'         # OpenTopography global DEM API (needs a free API key)
OT_SOURCES = [('COP30', 'Copernicus GLO-30 (radar, 30 m)'), ('AW3D30', 'ALOS AW3D30 (optical stereo, 30 m)')]
STAC_URL = 'https://stac.pgc.umn.edu/api/v1/'                      # Polar Geospatial Center STAC catalogue (open, no login)
ARCTICDEM_RESOLUTIONS = ('10m', '32m')                             # finest open ArcticDEM v4 mosaic first
PAD_DEG = 0.01                                                     # margin around the tile when downloading
SEARCH_FACTOR = 8                                                  # 8 x 25 m = 200 m cells, the scale of the Mount Sharp numbers
ROUGHNESS_M = 1000
NODATA = -9999.0
OUTLIER_M = 50.0                                                   # a cell this far from ASTER (after removing the offset) counts as an outlier
# Agreement thresholds (judgement calls, stated in the methodology note)
MAX_NMAD_M = 20.0                                                  # robust spread of the elevation difference
MAX_SLOPE_DIFF_DEG = 3.0                                           # mean slope at 200 m cells
MAX_RELIEF_DIFF = 0.15                                             # relief after removing tilt, as a share of ASTER's
MAX_OUTLIER_SHARE = 0.05                                           # share of cells more than OUTLIER_M from ASTER (after removing the offset)


def block_mean(a, f):
    n = a.shape[0] // f * f
    with np.errstate(all='ignore'):
        return np.nanmean(a[:n, :n].reshape(n // f, f, n // f, f), axis=(1, 3))


def to_tile_grid(sources, tile):
    """Mosaic the source rasters and resample them onto the tile's grid (same CRS, 25 m cells). Missing cells are NaN."""
    bounds = transform_bounds(tile.crs, sources[0].crs, *tile.bounds, densify_pts=21)
    pad = 3 * max(sources[0].res)
    mosaic, transform_ = merge(sources, bounds=(bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad), nodata=NODATA)
    src = mosaic[0].astype('float32')
    src[(src <= -500) | ~np.isfinite(src)] = np.nan                  # voids (the lowest land on Earth is far above -500 m here)
    out = np.full(tile.shape, np.nan, dtype='float32')
    reproject(src, out, src_transform=transform_, src_crs=sources[0].crs, dst_transform=tile.transform, dst_crs=tile.crs,
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def from_opentopography(demtype, tile, key):
    """Download a global DEM for the tile's bounding box from OpenTopography and put it on the tile grid."""
    west, south, east, north = transform_bounds(tile.crs, 'EPSG:4326', *tile.bounds)
    r = requests.get(OT_URL, params={'demtype': demtype, 'south': south - PAD_DEG, 'north': north + PAD_DEG, 'west': west - PAD_DEG,
                                     'east': east + PAD_DEG, 'outputFormat': 'GTiff', 'API_Key': key}, timeout=180)
    if r.status_code in (401, 403):
        raise RuntimeError('OpenTopography rejected the API key')
    if r.status_code != 200 or r.content[:4] not in (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+'):
        raise RuntimeError(f'OpenTopography {demtype}: status {r.status_code}, {r.text[:200]}')
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f'{demtype}.tif'
        path.write_bytes(r.content)
        with rasterio.open(path) as src:
            return to_tile_grid([src], tile)


def from_arcticdem(tile):
    """ArcticDEM v4 mosaic tiles over the tile (open, from the PGC STAC catalogue). Returns the array and a label."""
    west, south, east, north = transform_bounds(tile.crs, 'EPSG:4326', *tile.bounds)
    client = Client.open(STAC_URL)
    collections = [c.id for c in client.get_collections() if c.id.lower().startswith('arcticdem-mosaics-v4')]
    for resolution in ARCTICDEM_RESOLUTIONS:
        match = sorted(c for c in collections if c.lower().endswith(resolution))
        if not match:
            continue
        items = list(client.search(collections=[match[-1]], bbox=[west, south, east, north]).items())
        hrefs = []
        for item in items:
            key = 'dem' if 'dem' in item.assets else next((k for k, a in item.assets.items() if a.href.endswith('_dem.tif')), None)
            if key:
                hrefs.append(item.assets[key].href)
        if hrefs:
            sources = [rasterio.open(h) for h in hrefs]
            out = to_tile_grid(sources, tile)
            for s in sources:
                s.close()
            return out, f'ArcticDEM {match[-1].split("-")[-2]} {resolution} (stereo photogrammetry)'
    raise RuntimeError('No ArcticDEM v4 mosaic tiles found for the tile: collections seen: ' + ', '.join(collections))


def measure(z):
    """Slope (at 200 m cells), relief and roughness after removing the tilt, for one elevation grid on the tile grid."""
    zc = block_mean(z, SEARCH_FACTOR)
    res = 25.0 * SEARCH_FACTOR
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


# ---------------------------------------------------------------- The tile (ASTER) is the reference grid
with rasterio.open(TILE_FILE) as tile:
    aster = tile.read(1).astype('float32')
    aster[aster == tile.nodata] = np.nan
    print(f'Tile: {TILE_FILE.name}, {tile.width} x {tile.height} cells of {tile.res[0]:.0f} m, {tile.crs}')

    key = os.environ.get('OPENTOPO_API_KEY') or getpass.getpass('OpenTopography API key (not saved): ')
    sources = {}
    for demtype, label in OT_SOURCES:
        print(f'fetching {label} ...')
        sources[label] = from_opentopography(demtype, tile, key)
    print('fetching ArcticDEM ...')
    arctic, arctic_label = from_arcticdem(tile)
    sources[arctic_label] = arctic

# ---------------------------------------------------------------- Compare each source with ASTER
ref = measure(aster)
rows = [{'SOURCE': 'ASTER GDEM v3 (the tile, reference)', 'VALID': 1.0, 'OFFSET_M': 0.0, 'NMAD_M': 0.0, 'RMSE_M': 0.0, 'OUTLIERS': 0.0,
         'CORRELATION': 1.0, **ref, 'VERDICT': 'reference'}]
for label, z in sources.items():
    both = np.isfinite(aster) & np.isfinite(z)
    if both.mean() < 0.5:
        rows.append({'SOURCE': label, 'VALID': both.mean(), 'VERDICT': 'Not enough overlap to compare'})
        continue
    diff = z[both] - aster[both]
    offset = np.median(diff)                              # datum and glacier-surface differences show up here; it is removed below
    centred = diff - offset
    m = measure(np.where(np.isfinite(z), z, np.nan))
    nmad = 1.4826 * np.median(np.abs(centred))
    outliers = np.mean(np.abs(centred) > OUTLIER_M)
    ok = nmad <= MAX_NMAD_M and outliers <= MAX_OUTLIER_SHARE and abs(m['SLOPE_MEAN'] - ref['SLOPE_MEAN']) <= MAX_SLOPE_DIFF_DEG \
        and abs(m['RELIEF_M'] - ref['RELIEF_M']) <= MAX_RELIEF_DIFF * ref['RELIEF_M']
    rows.append({'SOURCE': label, 'VALID': both.mean(), 'OFFSET_M': offset, 'NMAD_M': nmad, 'RMSE_M': np.sqrt(np.mean(centred ** 2)),
                 'OUTLIERS': outliers, 'CORRELATION': np.corrcoef(aster[both], z[both])[0, 1], **m,
                 'VERDICT': 'Agrees with ASTER' if ok else 'Disagrees with ASTER'})
result = pd.DataFrame(rows)

print('\nASTER TILE AGAINST THREE INDEPENDENT ELEVATION MODELS')
print(f'(agree = difference spread (NMAD) <= {MAX_NMAD_M:.0f} m, no more than {MAX_OUTLIER_SHARE:.0%} of cells {OUTLIER_M:.0f} m off, mean slope within {MAX_SLOPE_DIFF_DEG:.0f} deg, relief within {MAX_RELIEF_DIFF:.0%};'
      f' slope, relief and roughness measured on 200 m cells with the tilt removed, as in the tile search)')
show = result.rename(columns={'VALID': 'Overlap', 'OFFSET_M': 'Offset (m)', 'NMAD_M': 'NMAD (m)', 'RMSE_M': 'RMSE (m)', 'OUTLIERS': f'>{OUTLIER_M:.0f} m off',
                              'CORRELATION': 'Correlation', 'SLOPE_MEAN': 'Slope mean (deg)', 'SLOPE_P95': 'Slope p95 (deg)',
                              'RELIEF_M': 'Relief (m)', 'ROUGHNESS_M': 'Roughness (m)'})
for c in ('Overlap', f'>{OUTLIER_M:.0f} m off'):
    show[c] = show[c].map(lambda v: f'{v:.0%}' if pd.notna(v) else '')
for c in ('Offset (m)', 'NMAD (m)', 'RMSE (m)', 'Relief (m)', 'Roughness (m)', 'Slope mean (deg)', 'Slope p95 (deg)'):
    show[c] = show[c].map(lambda v: f'{v:.1f}' if pd.notna(v) else '')
show['Correlation'] = show['Correlation'].map(lambda v: f'{v:.3f}' if pd.notna(v) else '')
print(show.fillna('').to_string(index=False))
agree = int((result['VERDICT'] == 'Agrees with ASTER').sum())
print(f'\n{agree} of {len(sources)} independent sources agree with the ASTER tile.')
print('Offsets come from different vertical references (geoids or the ellipsoid) plus real glacier change between survey dates, so only the spread (NMAD),')
print('outliers, slope, relief and roughness are used for the verdict. A glacier surface can genuinely change by tens of metres between surveys.')
result.round(3).to_csv(OUT_FILE, index=False)
print(f'\nSaved {OUT_FILE.name} in data/clean')