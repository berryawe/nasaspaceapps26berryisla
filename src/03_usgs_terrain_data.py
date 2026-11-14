import sys
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window, from_bounds
from scipy.ndimage import uniform_filter

BASE_DIR = Path(__file__).resolve().parent.parent

# Defining constants
# USGS Astrogeology MOLA-HRSC blended global DEM, 200 m/px 
DEM_URL = 'https://planetarymaps.usgs.gov/mosaic/Mars/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif'
DEM_SOURCE = sys.argv[1] if len(sys.argv) > 1 else DEM_URL       
AOI_NAME = sys.argv[2] if len(sys.argv) > 2 else 'mount_sharp'

CENTER_LAT, CENTER_LON = -5.08, 137.85   # Mount Sharp (Aeolis Mons) summit area, degrees (east positive)
HALF_WIDTH_KM = 40.0                     # area of interest = a square of 2 x this width
TARGET_RES_M = 200.0                     # cell size used for slope/roughness (use ~10 for the 1 m Gale DEM)
ROUGHNESS_WINDOWS_M = (1000, 5000)       # baselines for roughness
HAZARD_SLOPES_DEG = (15, 25)             # slopes steeper than these count as hazardous 
MARS_RADIUS_M = 3396190.0                # Mars reference sphere used by USGS map products
MARS_GRAVITY = 3.71                      # m/s2
R_MARS_AIR = 191.0                       # J/kg/K
COLD_AIR_K = 200.0                       # typical air temperature used for the scale height

OUT_DIR = BASE_DIR / 'data' / 'raw'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Area of interest in degrees (one degree of latitude on Mars is ~59.27 km)
m_per_deg = MARS_RADIUS_M * np.pi / 180
half_lat = HALF_WIDTH_KM * 1000 / m_per_deg
half_lon = half_lat / np.cos(np.radians(CENTER_LAT))
west, east = CENTER_LON - half_lon, CENTER_LON + half_lon
south, north = CENTER_LAT - half_lat, CENTER_LAT + half_lat

# Open the DEM and read only the area of interest
path = f'/vsicurl/{DEM_SOURCE}' if DEM_SOURCE.startswith('http') else DEM_SOURCE
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
    with rasterio.open(path) as src:
        # The DEM may be in degrees (geographic) or metres (projected): get the native cell size in metres either way
        if src.crs.is_geographic:
            native_x = abs(src.transform.a) * m_per_deg * np.cos(np.radians(CENTER_LAT))
            native_y = abs(src.transform.e) * m_per_deg
        else:
            native_x, native_y = abs(src.transform.a), abs(src.transform.e)
        print(f'DEM: {src.width} x {src.height} px, {native_x:.0f} m/px, crs={src.crs.to_string()[:60]}')
        xs, ys = warp_transform(f'+proj=longlat +R={MARS_RADIUS_M} +no_defs', src.crs, [west, east], [south, north])
        win = from_bounds(xs[0], ys[0], xs[1], ys[1], src.transform)
        col0, row0 = max(0, int(round(win.col_off))), max(0, int(round(win.row_off)))
        col1 = min(src.width, int(round(win.col_off + win.width)))
        row1 = min(src.height, int(round(win.row_off + win.height)))
        if col1 <= col0 or row1 <= row0:
            raise RuntimeError('Area of interest lies outside the DEM - change CENTER_LAT / CENTER_LON')
        win = Window(col0, row0, col1 - col0, row1 - row0)

        factor = max(1.0, TARGET_RES_M / native_x)
        out_w, out_h = max(3, int(round(win.width / factor))), max(3, int(round(win.height / factor)))
        z = src.read(1, window=win, out_shape=(out_h, out_w), resampling=Resampling.average, masked=True)
        z = z.astype('float32').filled(np.nan)
        cell_x = native_x * win.width / out_w
        cell_y = native_y * win.height / out_h

valid_share = float(np.isfinite(z).mean())
if valid_share < 0.5:
    raise RuntimeError(f'Only {valid_share:.0%} of the area has elevation data - move the area or use another DEM')

# Elevation
elev = z[np.isfinite(z)]
# Slope (degrees) from the elevation gradient
gy, gx = np.gradient(z, cell_y, cell_x)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
slope = slope[np.isfinite(slope)]
# Roughness: RMS height deviation from the local mean elevation in a window (includes the slope itself)
z_fill = np.nan_to_num(z, nan=float(np.nanmean(z)))
rough = {}
for L in ROUGHNESS_WINDOWS_M:
    k = max(3, int(round(L / cell_x)))
    rough[L] = float(np.mean(np.sqrt(np.maximum(uniform_filter(z_fill ** 2, k) - uniform_filter(z_fill, k) ** 2, 0))))
# Air gets thinner with height: density falls by exp(-dz/H) with scale height H = R*T/g
scale_height = R_MARS_AIR * COLD_AIR_K / MARS_GRAVITY
density_drop = 1 - np.exp(-(elev.max() - elev.min()) / scale_height)

rows = [
    ('ELEV_MIN', elev.min(), 'm'), ('ELEV_MEAN', elev.mean(), 'm'), ('ELEV_MAX', elev.max(), 'm'),
    ('ELEV_RANGE', elev.max() - elev.min(), 'm'),
    ('SLOPE_MEAN', slope.mean(), 'deg'), ('SLOPE_MEDIAN', np.median(slope), 'deg'),
    ('SLOPE_P95', np.percentile(slope, 95), 'deg'), ('SLOPE_MAX', slope.max(), 'deg'),
    ('SLOPE_SHARE_FLAT_LT5', (slope < 5).mean(), 'fraction'),
    (f'SLOPE_SHARE_GT{HAZARD_SLOPES_DEG[0]}', (slope > HAZARD_SLOPES_DEG[0]).mean(), 'fraction'),
    (f'SLOPE_SHARE_GT{HAZARD_SLOPES_DEG[1]}', (slope > HAZARD_SLOPES_DEG[1]).mean(), 'fraction'),
    (f'ROUGHNESS_{ROUGHNESS_WINDOWS_M[0]}M', rough[ROUGHNESS_WINDOWS_M[0]], 'm'),
    (f'ROUGHNESS_{ROUGHNESS_WINDOWS_M[1]}M', rough[ROUGHNESS_WINDOWS_M[1]], 'm'),
    ('AIR_DENSITY_DROP_ACROSS_RELIEF', density_drop, 'fraction'),
    ('VALID_DATA_SHARE', valid_share, 'fraction'), ('CELL_SIZE', cell_x, 'm'),
]
terrain_indicators_df = pd.DataFrame(rows, columns=['INDICATOR', 'VALUE', 'UNIT'])
terrain_indicators_df.insert(0, 'AOI', AOI_NAME)
terrain_indicators_df['VALUE'] = terrain_indicators_df['VALUE'].astype(float)

# Summarise + save
print(terrain_indicators_df.round(3).to_string(index=False))
terrain_indicators_df.to_csv(OUT_DIR / f'terrain_{AOI_NAME}.csv', index=False)
print('Mt sharp terrain data fetched and saved')