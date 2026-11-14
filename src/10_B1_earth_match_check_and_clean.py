import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Defining constants
RAW_DIR = BASE_DIR / 'data' / 'raw'
CLEAN_DIR = BASE_DIR / 'data' / 'clean'
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
FINAL_FILE = RAW_DIR / 'earth_b_final.csv'                  # best window per region with terrain and climate (Method B script)
ELEV_FILE = RAW_DIR / 'earth_b_elevation.csv'               # 1 km elevation table (Method B script)
MARS_TERRAIN_FILE = RAW_DIR / 'terrain_mount_sharp.csv'     # Mount Sharp indicators (Mars terrain script)
CHECK_FILE = RAW_DIR / 'earth_b_terrain_check.csv'
CLEAN_FILE = CLEAN_DIR / 'earth_matches_method_b_final.csv'
WINDOW_KM = 80.0                                            # same window as the Method B script
M_PER_DEG = 111195.0
# Gale / Mount Sharp climate reference values (from the Mars climate runs; wind ratio is the MCD average of the three sols)
MARS_REF = {'T2M_MIN_C': -76.5, 'T2M_RANGE_K': 66.6, 'WS2M_MS': 11.1, 'WIND_NIGHT_DAY_RATIO': 2.0, 'DENSITY_VS_MARS': 1.0}

best = pd.read_csv(FINAL_FILE)
elev = pd.read_csv(ELEV_FILE)
mars = pd.read_csv(MARS_TERRAIN_FILE).set_index('INDICATOR')['VALUE']

# ---------------------------------------------------------------- CHECK: 1 km elevation of each best window
rows = []
for r in best.itertuples():
    e = elev[(elev.SITE_LAT == r.LAT) & (elev.SITE_LON == r.LON)]
    km = WINDOW_KM / 2 / (M_PER_DEG / 1000)
    w = e[(abs(e.LAT - r.WIN_LAT) <= km) & (abs(e.LON - r.WIN_LON) <= km / np.cos(np.radians(r.LAT)))]
    rows.append({'REGION_NAME': r.REGION_NAME, 'CELLS': len(w), 'VALID': round(w.ELEV_M.notna().mean(), 2),
                 'ELEV_MIN_1KM': w.ELEV_M.min(), 'ELEV_MAX_1KM': w.ELEV_M.max(), 'RANGE_1KM': w.ELEV_M.max() - w.ELEV_M.min(),
                 'RANGE_200M': r.ELEV_RANGE, 'SLOPE_MEAN_200M': r.SLOPE_MEAN})
check_df = pd.DataFrame(rows)
print(check_df.round(2).to_string(index=False))
check_df.to_csv(CHECK_FILE, index=False)
print(f'Saved {CHECK_FILE}')

# ---------------------------------------------------------------- CLEAN: tidy table, indicators as rows and one column per region
final_df = best.copy()
final_df['GATES'] = np.where(final_df['GATES_PASSED'], 'Yes', 'No (best month shown is the least bad)')
# Regions whose terrain could be checked rank first; unverified ones follow, because their terrain score can't be trusted
final_df['UNVERIFIED'] = final_df['TERRAIN_DATA'].str.startswith('Unverified')
final_df = final_df.sort_values(['UNVERIFIED', 'COMBINED_SCORE'], ascending=[True, False]).reset_index(drop=True)

POWER, ASTER = 'NASA POWER (MERRA-2), modelled', 'ASTER GDEM v3 (NASA/METI), 200 m'
# (group, indicator, unit, source, Mars value, column in final_df, decimals; None = text)
spec = [
    ('Site', 'Search box centre latitude', 'deg', 'Chosen mountain region', '', 'LAT', 2),
    ('Site', 'Search box centre longitude', 'deg', 'Chosen mountain region', '', 'LON', 2),
    ('Site', 'Best month', '', POWER, 'Ls 135-178', 'BEST_MONTH', None),
    ('Site', 'Climate cell elevation', 'm', POWER, '', 'CLIMATE_ELEV_M', 0),
    ('Match scores', 'Terrain match', '0-1', ASTER, '', 'TERRAIN_SCORE', 2),
    ('Match scores', 'Climate match', '0-1', POWER, '', 'CLIMATE_SCORE', 2),
    ('Match scores', 'Combined match', '0-1', 'Mean of terrain and climate', '', 'COMBINED_SCORE', 2),
    ('Best terrain window', 'Window centre latitude', 'deg', ASTER, '', 'WIN_LAT', 2),
    ('Best terrain window', 'Window centre longitude', 'deg', ASTER, '', 'WIN_LON', 2),
    ('Best terrain window', 'Elevation range', 'm', ASTER, int(round(mars['ELEV_RANGE'])), 'ELEV_RANGE', 0),
    ('Best terrain window', 'Mean slope', 'deg', ASTER, round(mars['SLOPE_MEAN'], 1), 'SLOPE_MEAN', 1),
    ('Best terrain window', 'Roughness, 1 km window', 'm', ASTER, round(mars['ROUGHNESS_1000M'], 1), 'ROUGHNESS_1000M', 1),
    ('Best terrain window', 'Roughness, 5 km window', 'm', ASTER, round(mars['ROUGHNESS_5000M'], 1), 'ROUGHNESS_5000M', 1),
    ('Climate at that window', 'Climate gates passed', '', 'Day-night range and cold-night gates', '', 'GATES', None),
    ('Climate at that window', 'Night minimum air temperature', 'deg C', POWER, MARS_REF['T2M_MIN_C'], 'T2M_MIN_C', 1),
    ('Climate at that window', 'Day-night temperature range', 'K', POWER, MARS_REF['T2M_RANGE_K'], 'T2M_RANGE_K', 1),
    ('Climate at that window', 'Mean wind speed, 2 m', 'm/s', POWER, MARS_REF['WS2M_MS'], 'WS2M_MS', 1),
    ('Climate at that window', 'Wind night/day ratio', '', POWER + ', hourly', MARS_REF['WIND_NIGHT_DAY_RATIO'], 'WIND_NIGHT_DAY_RATIO', 2),
    ('Climate at that window', 'Air density compared with Mars', 'x Mars', POWER, MARS_REF['DENSITY_VS_MARS'], 'DENSITY_VS_MARS', 0),
    ('Elevation check', 'Lowest point, 1 km grid', 'm', 'ASTER GDEM v3 (NASA/METI), 1 km', '', 'ELEV_MIN_1KM', 0),
    ('Elevation check', 'Highest point, 1 km grid', 'm', 'ASTER GDEM v3 (NASA/METI), 1 km', '', 'ELEV_MAX_1KM', 0),
    ('Elevation check', 'Terrain data reliability', '', 'Comparison with known elevations', '', 'TERRAIN_DATA', None),
]
region_names = [f'#{i + 1} {name}' for i, name in enumerate(final_df['REGION_NAME'])]
clean_rows = []
for group, indicator, unit, source, mars_value, col, dec in spec:
    values = [v if dec is None else (int(round(v)) if dec == 0 else round(float(v), dec)) for v in final_df[col]]
    clean_rows.append([group, indicator, unit, source, mars_value] + values)
clean_df = pd.DataFrame(clean_rows, columns=['Group', 'Indicator', 'Unit', 'Source', 'Mount Sharp / Gale (Mars)'] + region_names)

# Summarise + save
print(clean_df.iloc[:, :8].to_string(index=False))
clean_df.to_csv(CLEAN_FILE, index=False)
print(f'Saved {CLEAN_FILE}')