import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from pathlib import Path
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter

BASE_DIR = Path(__file__).resolve().parent.parent

# Which 5 km square inside the 25 km tile looks most like Mount Sharp?
# Every 5 km square is judged at the Mars 200 m scale: slope shape and 1 km roughness, each scored as smaller/larger ratio against Mount Sharp
# (the same scoring as the terrain match). Elevation range and 5 km roughness are left out: a 5 km square cannot hold Mount Sharp's 5.4 km of relief.
# Defining constants
CLEAN_DIR = BASE_DIR / 'data' / 'clean'
RAW_DIR = BASE_DIR / 'data' / 'raw'
TILE_25 = CLEAN_DIR / 'dem_best_terrain_alaska_range_25km_5m.tif'
PREVIEW_25 = CLEAN_DIR / 'dem_best_terrain_alaska_range_25km_5m_preview.png'    # full size version (marked copy is saved if it matches the tile)
MARS_TERRAIN_FILE = CLEAN_DIR / 'mount_sharp_terrain_final.csv'
OUT_FILE = CLEAN_DIR / 'earth_best_mars_match_squares.csv'
SQUARE_KM = 5.0                   # the simulation domain
CELL_M = 200.0                    # the scale of the Mount Sharp numbers
STRIDE_CELLS = 2                  # squares start every 400 m
ROUGHNESS_M = 1000
TOP_N = 5
MIN_SEPARATION_KM = 5.0           # the top squares in the list do not overlap
# Indicator -> (Mount Sharp row in terrain_mount_sharp.csv, weight)
WEIGHTS = {'SLOPE_MEAN': ('Mean slope', 30), 'SLOPE_P95': ('95th percentile slope', 20), 'FLAT_5': ('Area flatter than 5 deg', 15),
           'STEEP_15': ('Area steeper than 15 deg', 15), 'ROUGH_1KM': ('Roughness, 1 km window', 20)}


def block_mean(a, f):
    n = a.shape[0] // f * f
    with np.errstate(all='ignore'):
        return np.nanmean(a[:n, :n].reshape(n // f, f, n // f, f), axis=(1, 3))


def ratio(a, b):
    return min(a, b) / max(a, b) if max(a, b) > 0 else 1.0


mars = pd.read_csv(MARS_TERRAIN_FILE).set_index('Indicator')['Value'].astype(float)
target = {k: float(mars[row]) for k, (row, w) in WEIGHTS.items()}

with rasterio.open(TILE_25) as src:
    z = src.read(1).astype('float64')
    z[z == src.nodata] = np.nan
    cell, transform_, crs, bounds = src.res[0], src.transform, src.crs, src.bounds
f = int(round(CELL_M / cell))
zc = block_mean(z, f)
zc = np.where(np.isfinite(zc), zc, np.nanmean(zc))
gy, gx = np.gradient(zc, CELL_M)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
side = int(round(SQUARE_KM * 1000 / CELL_M))                      # cells per square side
yy, xx = np.mgrid[0:side, 0:side] * CELL_M
design = np.column_stack([xx.ravel(), yy.ravel(), np.ones(side * side)])
k = max(3, int(round(ROUGHNESS_M / CELL_M)))

rows = []
for r in range(0, zc.shape[0] - side + 1, STRIDE_CELLS):
    for c in range(0, zc.shape[1] - side + 1, STRIDE_CELLS):
        s = slope[r:r + side, c:c + side]
        zz = zc[r:r + side, c:c + side]
        resid = zz - (design @ np.linalg.lstsq(design, zz.ravel(), rcond=None)[0]).reshape(side, side)
        local_sd = np.sqrt(np.maximum(uniform_filter(resid ** 2, k) - uniform_filter(resid, k) ** 2, 0))
        got = {'SLOPE_MEAN': s.mean(), 'SLOPE_P95': np.percentile(s, 95), 'FLAT_5': 100 * np.mean(s < 5), 'STEEP_15': 100 * np.mean(s > 15), 'ROUGH_1KM': local_sd.mean()}
        score = sum(w * ratio(got[key], target[key]) for key, (row, w) in WEIGHTS.items()) / sum(w for row, w in WEIGHTS.values())
        rows.append({'ROW': r, 'COL': c, 'SCORE': score, **got, 'ELEV_MIN': zz.min(), 'ELEV_MAX': zz.max()})
table = pd.DataFrame(rows).sort_values('SCORE', ascending=False).reset_index(drop=True)


def describe(row_):
    """Pixel box, UTM and lat/lon of a square (row_ is a table row)."""
    x0, y1 = transform_ * (row_['COL'] * f, row_['ROW'] * f)
    size_m = side * CELL_M
    cx, cy = x0 + size_m / 2, y1 - size_m / 2
    lon, lat = transform(crs, 'EPSG:4326', [cx], [cy])
    corners = transform(crs, 'EPSG:4326', [x0, x0 + size_m, x0, x0 + size_m], [y1, y1, y1 - size_m, y1 - size_m])
    return {'lat': lat[0], 'lon': lon[0], 'x': cx, 'y': cy, 'S': min(corners[1]), 'N': max(corners[1]), 'W': min(corners[0]), 'E': max(corners[0]),
            'px': tuple(int(v) for v in (row_['COL'] * f, row_['ROW'] * f, (row_['COL'] + side) * f, (row_['ROW'] + side) * f))}


# the square picked earlier (the centre of the tile) on the same scale
centre_r = (zc.shape[0] - side) // 2
centre_c = (zc.shape[1] - side) // 2
current = table[(table['ROW'] == centre_r - centre_r % STRIDE_CELLS) & (table['COL'] == centre_c - centre_c % STRIDE_CELLS)].iloc[0]

# top squares that do not overlap
chosen = []
for _, row_ in table.iterrows():
    d = describe(row_)
    if all(np.hypot(d['x'] - o[1]['x'], d['y'] - o[1]['y']) >= MIN_SEPARATION_KM * 1000 for o in chosen):
        chosen.append((row_, d))
    if len(chosen) == TOP_N:
        break

print(f'Mount Sharp targets: mean slope {target["SLOPE_MEAN"]:.1f} deg, p95 {target["SLOPE_P95"]:.1f} deg, flatter than 5 deg {target["FLAT_5"]:.1f}%, '
      f'steeper than 15 deg {target["STEEP_15"]:.1f}%, roughness 1 km {target["ROUGH_1KM"]:.1f} m')
print(f'{len(table)} squares of {SQUARE_KM:g} km scored at {CELL_M:g} m cells\n')
out = []
for i, (row_, d) in enumerate(chosen, 1):
    out.append({'RANK': i, 'SCORE': row_['SCORE'], 'CENTRE_LAT': d['lat'], 'CENTRE_LON': d['lon'], 'UTM32605_E': d['x'], 'UTM32605_N': d['y'],
                'SOUTH': d['S'], 'NORTH': d['N'], 'WEST': d['W'], 'EAST': d['E'], 'PIXEL_X0_Y0_X1_Y1': str(d['px']),
                'SLOPE_MEAN': row_['SLOPE_MEAN'], 'SLOPE_P95': row_['SLOPE_P95'], 'FLAT_5_PCT': row_['FLAT_5'], 'STEEP_15_PCT': row_['STEEP_15'],
                'ROUGH_1KM_M': row_['ROUGH_1KM'], 'ELEV_MIN_M': row_['ELEV_MIN'], 'ELEV_MAX_M': row_['ELEV_MAX']})
result = pd.DataFrame(out)
cd = describe(current)
result.loc[len(result)] = {'RANK': 'current', 'SCORE': current['SCORE'], 'CENTRE_LAT': cd['lat'], 'CENTRE_LON': cd['lon'], 'UTM32605_E': cd['x'], 'UTM32605_N': cd['y'],
                           'SOUTH': cd['S'], 'NORTH': cd['N'], 'WEST': cd['W'], 'EAST': cd['E'], 'PIXEL_X0_Y0_X1_Y1': str(cd['px']), 'SLOPE_MEAN': current['SLOPE_MEAN'],
                           'SLOPE_P95': current['SLOPE_P95'], 'FLAT_5_PCT': current['FLAT_5'], 'STEEP_15_PCT': current['STEEP_15'], 'ROUGH_1KM_M': current['ROUGH_1KM'],
                           'ELEV_MIN_M': current['ELEV_MIN'], 'ELEV_MAX_M': current['ELEV_MAX']}
show = result[['RANK', 'SCORE', 'CENTRE_LAT', 'CENTRE_LON', 'SLOPE_MEAN', 'SLOPE_P95', 'FLAT_5_PCT', 'STEEP_15_PCT', 'ROUGH_1KM_M', 'ELEV_MIN_M', 'ELEV_MAX_M']].copy()
show['SCORE'] = show['SCORE'].map('{:.2f}'.format)
for c in ('CENTRE_LAT', 'CENTRE_LON'):
    show[c] = show[c].map('{:.5f}'.format)
for c in ('SLOPE_MEAN', 'SLOPE_P95', 'FLAT_5_PCT', 'STEEP_15_PCT', 'ROUGH_1KM_M', 'ELEV_MIN_M', 'ELEV_MAX_M'):
    show[c] = show[c].map('{:.1f}'.format)
print(show.to_string(index=False))

best_row, best = chosen[0]
print(f'\nBEST MATCH TO MOUNT SHARP (score {best_row["SCORE"]:.2f}): centre {best["lat"]:.5f}, {best["lon"]:.5f}  (paste "{best["lat"]:.5f}, {best["lon"]:.5f}" into Google Maps)')
print(f'  5 km square: south {best["S"]:.5f}  north {best["N"]:.5f}  west {best["W"]:.5f}  east {best["E"]:.5f}')
print(f'  UTM 5N (EPSG:32605): easting {best["x"]:.1f}, northing {best["y"]:.1f}; pixels in the 25 km tif / full-size preview (x0, y0, x1, y1): {best["px"]}')
print(f'The square picked earlier scores {current["SCORE"]:.2f} on this scale (mean slope {current["SLOPE_MEAN"]:.1f} deg against Mount Sharp {target["SLOPE_MEAN"]:.1f}).')
result.round(5).to_csv(OUT_FILE, index=False)
print(f'Saved {OUT_FILE.name} in data/clean')

# marked copy of the full-size preview: best square in red, earlier square in white
if PREVIEW_25.exists():
    img = Image.open(PREVIEW_25).convert('RGB')
    if img.size == (z.shape[1], z.shape[0]):
        draw = ImageDraw.Draw(img)
        draw.rectangle(cd['px'], outline=(255, 255, 255), width=6)
        draw.rectangle(best['px'], outline=(255, 0, 0), width=6)
        marked = CLEAN_DIR / 'dem_best_terrain_alaska_range_25km_5m_preview_marked.png'
        img.save(marked)
        print(f'Saved {marked.name}: red box = best match to Mount Sharp, white box = the square picked earlier')
    else:
        print('The preview is not full size, so no marked copy was made (set MAX_SIDE_PX = None in make_preview_png_2m.py and rerun it).')