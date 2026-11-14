import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# The best climate match and the best terrain match among the Earth candidates (analogues and controls excluded),
# shown next to Mars (Gale crater / Mount Sharp).
# Defining constants
CLEAN_DIR = BASE_DIR / 'data' / 'clean'
IN_FILE = CLEAN_DIR / 'earth_final_matches.csv'                       # every site from both methods, from the final combiner
OUT_FILE = CLEAN_DIR / 'final_results.csv'   # the two picks against Mars, one tidy table

# Which sites compete (STATUS column of the input file)
CLIMATE_POOL = ['Eligible', 'Terrain disproved (REMA)', 'Provisional (terrain unverified)']   # passes the climate gates; terrain does not matter
TERRAIN_POOL = ['Eligible']                                                                  # passes the climate gates AND terrain was verified

WINDOW_KM = 80.0                                                     # side of the square window the site was scored on (same as Mount Sharp)
M_PER_DEG = 111195.0                                                  # metres per degree of latitude

# Mars values (Gale crater, from the Mars database; terrain is Mount Sharp)
MARS = {'BEST_MONTH': 'Ls 135-178', 'T2M_MIN_C': -76.5, 'T2M_RANGE_K': 66.6, 'WS2M_MS': 11.1, 'WIND_NIGHT_DAY_RATIO': 2.0,
        'DUST_TAU': 0.49, 'DENSITY_VS_MARS': 1.0, 'ELEV_RANGE': 5443, 'SLOPE_MEAN': 7.2, 'ROUGHNESS_1000M': 35.8, 'ROUGHNESS_5000M': 161.6}

# Table rows: (group, indicator, unit, column, decimals)  - decimals None = text
ROWS = [('Match', 'Method', '', 'METHOD', None), ('Match', 'Status', '', 'STATUS', None),
        ('Match', 'Combined score (60% climate, 40% terrain)', '0-1', 'COMBINED_SCORE', 2),
        ('Match', 'Climate match', '0-1', 'CLIMATE_SCORE', 2), ('Match', 'Terrain match', '0-1', 'TERRAIN_SCORE', 2),
        ('Climate', 'Best month', '', 'BEST_MONTH', None),
        ('Climate', 'Night minimum air temperature', 'deg C', 'T2M_MIN_C', 1), ('Climate', 'Day-night temperature range', 'K', 'T2M_RANGE_K', 1),
        ('Climate', 'Mean wind speed, 2 m', 'm/s', 'WS2M_MS', 1), ('Climate', 'Wind night/day ratio', '', 'WIND_NIGHT_DAY_RATIO', 2),
        ('Climate', 'Dust optical depth', '', 'DUST_TAU', 3), ('Climate', 'Rain', 'mm/day', 'PRECIP_MM_DAY', 2),
        ('Climate', 'Air density compared with Mars', 'x Mars', 'DENSITY_VS_MARS', 0),
        ('Terrain', 'Elevation range', 'm', 'ELEV_RANGE', 0), ('Terrain', 'Mean slope', 'deg', 'SLOPE_MEAN', 1),
        ('Terrain', 'Roughness, 1 km window', 'm', 'ROUGHNESS_1000M', 1), ('Terrain', 'Roughness, 5 km window', 'm', 'ROUGHNESS_5000M', 1),
        ('Location', 'Best window latitude', 'deg', 'WIN_LAT', 3), ('Location', 'Best window longitude', 'deg', 'WIN_LON', 3),
        ('Check', 'Terrain data reliability', '', 'TERRAIN_DATA', None)]


def coordinates(lat, lon):
    return f'{abs(lat):.3f}\u00b0{"N" if lat >= 0 else "S"} {abs(lon):.3f}\u00b0{"E" if lon >= 0 else "W"}'


def extent(lat, lon):
    """South, north, west, east edges (degrees) of the WINDOW_KM square centred on the site."""
    half_lat = WINDOW_KM * 1000 / 2 / M_PER_DEG
    half_lon = half_lat / np.cos(np.radians(lat))
    return lat - half_lat, lat + half_lat, lon - half_lon, lon + half_lon

df = pd.read_csv(IN_FILE)
df = df[df['KIND'] == 'Candidate']                      # no analogues, no controls
df['PLACE'] = df['PLACE'].str.replace(r' \(A\)$', '', regex=True)


def best(pool, score):
    """Highest score among the sites in the pool; ties go to the thinner air (lower density compared with Mars)."""
    sites = df[df['STATUS'].isin(pool)].sort_values([score, 'DENSITY_VS_MARS'], ascending=[False, True])
    return sites.iloc[0]


climate_pick, terrain_pick = best(CLIMATE_POOL, 'CLIMATE_SCORE'), best(TERRAIN_POOL, 'TERRAIN_SCORE')
names = [f'Best climate match: {climate_pick["PLACE"]}', f'Best terrain match: {terrain_pick["PLACE"]}']

rows = [['Site', 'Place', '', '', climate_pick['PLACE'], terrain_pick['PLACE']],
        ['Site', 'Coordinates (centre of the best window)', 'lat, lon', '', coordinates(climate_pick['WIN_LAT'], climate_pick['WIN_LON']),
         coordinates(terrain_pick['WIN_LAT'], terrain_pick['WIN_LON'])]]
for i, edge in enumerate(('South edge', 'North edge', 'West edge', 'East edge')):
    rows.append(['Site', f'{WINDOW_KM:.0f} km window: {edge}', 'deg', '', f'{extent(climate_pick["WIN_LAT"], climate_pick["WIN_LON"])[i]:.3f}',
                 f'{extent(terrain_pick["WIN_LAT"], terrain_pick["WIN_LON"])[i]:.3f}'])
for group, indicator, unit, col, dec in ROWS:
    def fmt(v):
        if pd.isna(v):
            return ''
        return v if dec is None else (int(round(v)) if dec == 0 else round(float(v), dec))
    rows.append([group, indicator, unit, fmt(MARS.get(col)) if col in MARS else '', fmt(climate_pick[col]), fmt(terrain_pick[col])])
out = pd.DataFrame(rows, columns=['Group', 'Indicator', 'Unit', 'Mars (Gale / Mount Sharp)'] + names)

print('BEST CLIMATE MATCH AND BEST TERRAIN MATCH AGAINST MARS (analogues and controls excluded)')
print(f'Climate pick: highest climate score among sites that pass the climate gates ({len(df[df["STATUS"].isin(CLIMATE_POOL)])} sites).')
print(f'Terrain pick: highest terrain score among sites that pass the gates with verified terrain ({len(df[df["STATUS"].isin(TERRAIN_POOL)])} sites).\n')
print(out.drop(columns='Group').to_string(index=False))
out.to_csv(OUT_FILE, index=False)
print(f'\nSaved {OUT_FILE.name} in data/clean')