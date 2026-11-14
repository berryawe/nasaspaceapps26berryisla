# make_preview_png.py - shaded, coloured picture of the terrain tile
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib import cm

TIF = 'data/clean/dem_best_terrain_alaska_range_5km.tif'
OUT = 'data/clean/dem_best_terrain_alaska_range_5km_preview.png'
SUN_AZIMUTH = 315   # degrees, sun from the north-west
SUN_ALTITUDE = 40   # degrees above the horizon

with rasterio.open(TIF) as src:
    z = src.read(1).astype('float64')
    cell = src.res[0]
    nodata = src.nodata

z[z == nodata] = np.nan
z = np.where(np.isnan(z), np.nanmean(z), z)

# hillshade
dy, dx = np.gradient(z, cell)
slope = np.arctan(np.hypot(dx, dy))
aspect = np.arctan2(-dx, dy)
az = np.radians(360 - SUN_AZIMUTH + 90)
alt = np.radians(SUN_ALTITUDE)
shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
shade = np.clip(shade, 0, 1)

# colour by elevation, darkened by shade
colour = cm.terrain((z - z.min()) / (z.max() - z.min()))[..., :3]
picture = colour * (0.35 + 0.65 * shade[..., None])

plt.imsave(OUT, np.clip(picture, 0, 1))
print('saved', OUT, f'| elevation {z.min():.0f} to {z.max():.0f} m, cell {cell:.0f} m')