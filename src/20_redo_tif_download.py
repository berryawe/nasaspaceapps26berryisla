import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from pystac_client import Client
from rasterio.fill import fillnodata
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds
from scipy.ndimage import uniform_filter

BASE_DIR = Path(__file__).resolve().parent.parent

# Rebuild the 5 km terrain tile from the finest open elevation model, ArcticDEM v4.1 at 2 m (stereo photogrammetry from sub-metre satellite images).
# Two tiles with the same centre as the ASTER tile (same projection): the 5 km square at 2 m (for the simulation) and a 25 km square at 5 m (context).
# Then check it against the ASTER tile and against the Mount Sharp slope limits that were used to choose the square.
# Defining constants
RAW_DIR = BASE_DIR / 'data' / 'raw'
CLEAN_DIR = BASE_DIR / 'data' / 'clean'
ASTER_TILE = CLEAN_DIR / 'dem_best_terrain_alaska_range_5km.tif'          # defines the square and projection
MARS_TERRAIN_FILE = CLEAN_DIR / 'mount_sharp_terrain_final.csv'
TILES = [(5.0, 2.0), (25.0, 5.0)]                                         # (square size in km, cell size in m); both share the centre of the ASTER tile
STAC_URL = 'https://stac.pgc.umn.edu/api/v1/'                             # Polar Geospatial Center STAC catalogue (open, no login)
MAX_FILL_M = 50.0                                                         # small voids (shadow, snow, steep faces) are filled from their edges up to this width
MAX_VOID_SHARE = 0.02                                                     # warn if more than this share of cells is still empty after filling
SEARCH_M = 200.0                                                          # the scale of the Mount Sharp numbers
NODATA = -9999.0


def block_mean(a, f):
    n = a.shape[0] // f * f
    with np.errstate(all='ignore'):
        return np.nanmean(a[:n, :n].reshape(n // f, f, n // f, f), axis=(1, 3))


def slopes(z, cell):
    z = np.where(np.isfinite(z), z, np.nanmean(z))
    gy, gx = np.gradient(z, cell)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def detrended(z, cell):
    """Elevation after removing the best-fit plane (the tilt of the square)."""
    z = np.where(np.isfinite(z), z, np.nanmean(z))
    n = z.shape[0]
    yy, xx = np.mgrid[0:n, 0:n] * cell
    design = np.column_stack([xx.ravel(), yy.ravel(), np.ones(n * n)])
    return z - (design @ np.linalg.lstsq(design, z.ravel(), rcond=None)[0]).reshape(n, n)


def measure(z, cell):
    """Slope, relief and roughness of a square at the Mars 200 m scale, tilt removed."""
    f = int(round(SEARCH_M / cell))
    zc = block_mean(z, f)
    res = cell * f
    s = slopes(zc, res)
    resid = detrended(zc, res)
    k = max(3, int(round(1000 / res)))
    local_sd = np.sqrt(np.maximum(uniform_filter(resid ** 2, k) - uniform_filter(resid, k) ** 2, 0))
    low, high = np.percentile(resid, [0.5, 99.5])
    return {'SLOPE_MEAN': s.mean(), 'SLOPE_P95': np.percentile(s, 95), 'SLOPE_P99': np.percentile(s, 99), 'RELIEF_M': high - low, 'ROUGHNESS_M': local_sd.mean()}


def find_mosaic_tiles(bounds_ll):
    """Links to the ArcticDEM v4 2 m mosaic tiles that cover the square."""
    client = Client.open(STAC_URL)
    collections = [c.id for c in client.get_collections() if c.id.lower().startswith('arcticdem-mosaics-v4') and c.id.lower().endswith('-2m')]
    if not collections:
        raise RuntimeError('No ArcticDEM v4 2 m mosaic collection found in the catalogue')
    collection = sorted(collections)[-1]
    hrefs = []
    for item in client.search(collections=[collection], bbox=bounds_ll).items():
        key = 'dem' if 'dem' in item.assets else next((k for k, a in item.assets.items() if a.href.endswith('_dem.tif')), None)
        if key:
            hrefs.append(item.assets[key].href)
    if not hrefs:
        raise RuntimeError(f'No 2 m tiles found over the square in {collection}')
    return collection, hrefs


# ---------------------------------------------------------------- Centre of the picked 5 km square (from the ASTER tile)
with rasterio.open(ASTER_TILE) as aster_tile:
    crs = aster_tile.crs
    left0, bottom0, right0, top0 = aster_tile.bounds
    aster = aster_tile.read(1).astype('float32')
    aster[aster == aster_tile.nodata] = np.nan
    aster_transform, aster_cell = aster_tile.transform, aster_tile.res[0]
cx, cy = (left0 + right0) / 2, (bottom0 + top0) / 2
mars = pd.read_csv(MARS_TERRAIN_FILE).set_index('Indicator')['Value'].astype(float)

for km, CELL_M in TILES:
    half = km * 1000 / 2
    left, right, bottom, top = cx - half, cx + half, cy - half, cy + half
    size = int(round(km * 1000 / CELL_M))
    transform = from_origin(left, top, CELL_M, CELL_M)
    print(f'\n=== {km:g} km square, {size} x {size} cells of {CELL_M:g} m, {crs} ===')

    # Download the 2 m mosaic over the square (rasterio reads only what is needed, averaged to the cell size)
    collection, hrefs = find_mosaic_tiles(transform_bounds(crs, 'EPSG:4326', left, bottom, right, top))
    print(f'{collection}: {len(hrefs)} mosaic tile(s) over the square, reading ...')
    sources = [rasterio.open(h) for h in hrefs]
    box = transform_bounds(crs, sources[0].crs, left, bottom, right, top, densify_pts=21)
    pad = 3 * CELL_M
    mosaic, mosaic_transform = merge(sources, bounds=(box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), res=CELL_M,
                                     resampling=Resampling.average, nodata=NODATA)
    src_crs = sources[0].crs
    for s_ in sources:
        s_.close()
    src = mosaic[0].astype('float32')
    src[(src <= -500) | ~np.isfinite(src)] = np.nan
    z = np.full((size, size), np.nan, dtype='float32')
    reproject(src, z, src_transform=mosaic_transform, src_crs=src_crs, dst_transform=transform, dst_crs=crs,
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    del src, mosaic

    # Fill small voids, report what is left
    void_before = np.mean(~np.isfinite(z))
    filled = fillnodata(np.where(np.isfinite(z), z, NODATA), mask=np.isfinite(z).astype('uint8'), max_search_distance=MAX_FILL_M / CELL_M, smoothing_iterations=0)
    z = np.where(np.isfinite(z), z, np.where(filled == NODATA, np.nan, filled)).astype('float32')
    void_after = np.mean(~np.isfinite(z))
    print(f'Empty cells: {void_before:.2%} before filling, {void_after:.2%} after filling gaps up to {MAX_FILL_M:.0f} m wide')
    if void_after > MAX_VOID_SHARE:
        print(f'WARNING: more than {MAX_VOID_SHARE:.0%} of the square is still empty; check the picture before using this tile.')
    if not np.isfinite(z).any():
        raise RuntimeError('The grid is empty: nothing was read from the mosaic.')

    # Mars scale numbers at 200 m cells (the 5 km square is also compared with ASTER and the Mount Sharp limits)
    hi = measure(z, CELL_M)
    print(f'Elevation {np.nanmin(z):.0f} to {np.nanmax(z):.0f} m | at 200 m cells: mean slope {hi["SLOPE_MEAN"]:.1f} deg, p99 slope {hi["SLOPE_P99"]:.1f} deg, '
          f'relief {hi["RELIEF_M"]:.0f} m, roughness {hi["ROUGHNESS_M"]:.1f} m')
    if km == 5.0:
        aster_on_grid = np.full((size, size), np.nan, dtype='float32')
        reproject(aster, aster_on_grid, src_transform=aster_transform, src_crs=crs, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        both = np.isfinite(z) & np.isfinite(aster_on_grid)
        diff = z[both] - aster_on_grid[both]
        offset = np.median(diff)
        nmad = 1.4826 * np.median(np.abs(diff - offset))
        lo = measure(aster, aster_cell)
        rows = [('Mean slope at 2 m cells (deg)', f'{np.nanmean(slopes(z, CELL_M)):.1f}', '', 'detail the ASTER tile cannot show'),
                ('Mean slope at 20 m cells (deg)', f'{np.nanmean(slopes(block_mean(z, 10), 20.0)):.1f}', f'{np.nanmean(slopes(aster, aster_cell)):.1f}', 'ASTER at its own 25 m'),
                ('Mean slope at 200 m cells (deg)', f'{hi["SLOPE_MEAN"]:.1f}', f'{lo["SLOPE_MEAN"]:.1f}', f'Mars limit {float(mars["95th percentile slope"]):.1f} (Mount Sharp p95)'),
                ('99th-percentile slope, 200 m (deg)', f'{hi["SLOPE_P99"]:.1f}', f'{lo["SLOPE_P99"]:.1f}', f'Mars limit {float(mars["Maximum slope"]):.1f} (Mount Sharp max)'),
                ('Relief, tilt removed, 200 m (m)', f'{hi["RELIEF_M"]:.0f}', f'{lo["RELIEF_M"]:.0f}', ''),
                ('Roughness 1 km, 200 m cells (m)', f'{hi["ROUGHNESS_M"]:.1f}', f'{lo["ROUGHNESS_M"]:.1f}', f'Mount Sharp {float(mars["Roughness, 1 km window"]):.1f}')]
        print('\n2 m ARCTICDEM TILE AGAINST THE 25 m ASTER TILE')
        print(pd.DataFrame(rows, columns=['Indicator', 'ArcticDEM 2 m', 'ASTER 25 m', 'Note']).to_string(index=False))
        print(f'Height difference to ASTER: offset {offset:.1f} m (different vertical reference and survey dates, not an error), spread (NMAD) {nmad:.1f} m, overlap {both.mean():.0%}')
        inside = hi['SLOPE_MEAN'] <= float(mars['95th percentile slope']) and hi['SLOPE_P99'] <= float(mars['Maximum slope'])
        print('Inside the Mount Sharp slope limits at the 200 m scale: ' + ('yes' if inside else 'NO - the finer model puts this square outside the limits you set; tell me and we re-pick the square'))

    # Save
    out_file = CLEAN_DIR / f'dem_best_terrain_alaska_range_{km:g}km_{CELL_M:g}m.tif'
    z_out = np.where(np.isfinite(z), z, NODATA).astype('float32')
    with rasterio.open(out_file, 'w', driver='GTiff', height=size, width=size, count=1, dtype='float32', crs=crs, transform=transform,
                       nodata=NODATA, compress='deflate', predictor=3, tiled=True) as dst:
        dst.write(z_out, 1)
        dst.update_tags(SOURCE=f'ArcticDEM v4.1 2 m mosaic ({collection}), PGC/NSF', VERTICAL_DATUM='WGS84 ellipsoid (heights above the ellipsoid, as published)',
                        VOIDS_FILLED=f'gaps up to {MAX_FILL_M:.0f} m; {void_after:.2%} still empty', CELL_SIZE_M=str(CELL_M), SQUARE_KM=str(km))
        dst.set_band_description(1, 'elevation (m)')
    print(f'Saved {out_file.name} in data/clean')