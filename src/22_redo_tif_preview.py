import numpy as np
from PIL import Image
import rasterio
from matplotlib import cm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Shaded, coloured pictures of the rebuilt terrain tiles (PNG opens anywhere: email, Preview, slides).
# Defining constants
CLEAN_DIR = BASE_DIR / 'data' / 'clean'
TILES = ['dem_best_terrain_alaska_range_5km_2m.tif', 'dem_best_terrain_alaska_range_25km_5m.tif']
SUN_AZIMUTH = 315        # degrees, sun from the north-west
SUN_ALTITUDE = 40        # degrees above the horizon
VERTICAL_EXAGGERATION = 1.0   # 1 = true shape; try 1.5 to make relief stand out
COLOUR_MAP = cm.terrain
MAX_SIDE_PX = None       # full-size
CLIP_PERCENT = (1, 99)   # colours stretched between these elevation percentiles so a few extreme cells do not wash out the picture

for name in TILES:
    tif = CLEAN_DIR / name
    if not tif.exists():
        print(f'{name}: not found in data/clean, skipping')
        continue
    with rasterio.open(tif) as src:
        z = src.read(1).astype('float64')
        cell = src.res[0]
        z[z == src.nodata] = np.nan
    empty = ~np.isfinite(z)
    z = np.where(empty, np.nanmean(z), z)

    # hillshade
    dy, dx = np.gradient(z * VERTICAL_EXAGGERATION, cell)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.radians(360 - SUN_AZIMUTH + 90)
    alt = np.radians(SUN_ALTITUDE)
    shade = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)

    # colour by elevation, darkened by shade
    low, high = np.nanpercentile(z[~empty], CLIP_PERCENT)
    colour = COLOUR_MAP(np.clip((z - low) / (high - low), 0, 1))[..., :3]
    picture = colour * (0.35 + 0.65 * shade[..., None])
    picture[empty] = 1.0                                   # empty cells show as white

    out = CLEAN_DIR / (tif.stem + '_preview.png')
    image = Image.fromarray((np.clip(picture, 0, 1) * 255).astype('uint8'))
    if MAX_SIDE_PX and max(image.size) > MAX_SIDE_PX:
        image = image.resize((MAX_SIDE_PX, MAX_SIDE_PX), Image.LANCZOS)
    image.save(out, optimize=True)
    print(f'saved {out.name} | {z.shape[1]} x {z.shape[0]} px, cell {cell:g} m, elevation {np.nanmin(z):.0f} to {np.nanmax(z):.0f} m, empty cells {empty.mean():.2%}')