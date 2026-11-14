import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Defining constants
RAW_DIR = BASE_DIR / 'data' / 'raw'
OUT_DIR = BASE_DIR / 'data' / 'clean'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DAILY_FILE = RAW_DIR / 'gale_late_winter_daily_part2.csv'
TERRAIN_FILE = RAW_DIR / 'terrain_mount_sharp.csv'

REMS = 'REMS (measured)'
MCD = 'MCD v6.2 (modelled, climatology)'
DEM = 'USGS MOLA-HRSC DEM v2, 200 m'

# (column in raw file, group, label, unit, source, decimals, multiply by)
DAILY_SPEC = [
    ('HOURS_COVERED',      'Data',        'Hours of sol covered',          'h',        REMS, 0, 1),
    ('P_MEAN',             'Pressure',    'Mean pressure',                 'Pa',       REMS, 1, 1),
    ('P_MIN',              'Pressure',    'Minimum pressure',              'Pa',       REMS, 1, 1),
    ('P_MAX',              'Pressure',    'Maximum pressure',              'Pa',       REMS, 1, 1),
    ('RHO_MEAN',           'Density',     'Mean air density',              'kg/m3',    REMS, 4, 1),
    ('RHO_MIN',            'Density',     'Minimum air density',           'kg/m3',    REMS, 4, 1),
    ('RHO_VS_EARTH',       'Density',     'Mean density vs Earth sea level', '%',      REMS, 2, 100),
    ('AIR_T_NIGHT_MIN_C',  'Temperature', 'Night minimum air temperature', 'deg C',    REMS, 1, 1),
    ('AIR_T_DAY_MAX_C',    'Temperature', 'Day maximum air temperature',   'deg C',    REMS, 1, 1),
    ('AIR_T_RANGE_K',      'Temperature', 'Day-night temperature range',   'K',        REMS, 1, 1),
    ('MCD_WIND_NIGHT_MEAN','Wind',        'Mean wind, night (18-06 h)',    'm/s',      MCD,  1, 1),
    ('MCD_WIND_DAY_MEAN',  'Wind',        'Mean wind, day (06-18 h)',      'm/s',      MCD,  1, 1),
    ('MCD_WIND_MAX',       'Wind',        'Peak wind',                     'm/s',      MCD,  1, 1),
    ('MCD_WIND_MAX_HOUR',  'Wind',        'Local time of peak wind',       'LMST h',   MCD,  0, 1),
    ('MCD_WIND_MIN',       'Wind',        'Calmest wind',                  'm/s',      MCD,  1, 1),
    ('MCD_WIND_MIN_HOUR',  'Wind',        'Local time of calmest wind',    'LMST h',   MCD,  0, 1),
    ('MCD_CONV_UP_MAX',    'Wind',        'Peak convective updraft',       'm/s',      MCD,  1, 1),
    ('MCD_CONV_HOURS',     'Wind',        'Hours per sol with updraft > 5 m/s', 'h',   MCD,  0, 1),
    ('MCD_DUST_TAU_MEAN',  'Dust',        'Dust column optical depth',     '',         MCD,  2, 1),
    ('MCD_RHO_MEAN',       'Density',     'Mean air density (model)',      'kg/m3',    MCD,  4, 1),
    ('MCD_RHO_MIN',        'Density',     'Minimum air density (model)',   'kg/m3',    MCD,  4, 1),
]

TERRAIN_SPEC = [
    ('ELEV_MIN',                       'Elevation', 'Lowest point',                      'm', 0, 1),
    ('ELEV_MEAN',                      'Elevation', 'Mean elevation',                    'm', 0, 1),
    ('ELEV_MAX',                       'Elevation', 'Highest point',                     'm', 0, 1),
    ('ELEV_RANGE',                     'Elevation', 'Elevation range',                   'm', 0, 1),
    ('SLOPE_MEAN',                     'Slope',     'Mean slope',                        'deg', 1, 1),
    ('SLOPE_MEDIAN',                   'Slope',     'Median slope',                      'deg', 1, 1),
    ('SLOPE_P95',                      'Slope',     '95th percentile slope',             'deg', 1, 1),
    ('SLOPE_MAX',                      'Slope',     'Maximum slope',                     'deg', 1, 1),
    ('SLOPE_SHARE_FLAT_LT5',           'Slope',     'Area flatter than 5 deg',           '%', 1, 100),
    ('SLOPE_SHARE_GT15',               'Slope',     'Area steeper than 15 deg',          '%', 1, 100),
    ('SLOPE_SHARE_GT25',               'Slope',     'Area steeper than 25 deg',          '%', 1, 100),
    ('ROUGHNESS_1000M',                'Roughness', 'Roughness, 1 km window',            'm', 1, 1),
    ('ROUGHNESS_5000M',                'Roughness', 'Roughness, 5 km window',            'm', 1, 1),
    ('AIR_DENSITY_DROP_ACROSS_RELIEF', 'Atmosphere', 'Air density drop, base to summit', '%', 0, 100),
    ('CELL_SIZE',                      'Data',      'DEM cell size',                     'm', 0, 1),
    ('VALID_DATA_SHARE',               'Data',      'Area with valid data',              '%', 0, 100),
]

# Daily table: indicators as rows, one column per sol
daily_df = pd.read_csv(DAILY_FILE)
rows = []
for col, group, label, unit, source, dec, mult in DAILY_SPEC:
    row = {'Group': group, 'Indicator': label, 'Unit': unit, 'Source': source}
    for _, r in daily_df.iterrows():
        row[f"Sol {int(r['SOL'])} (Ls {r['LS']:.1f})"] = round(r[col] * mult, dec)
    rows.append(row)
mars_clean_df = pd.DataFrame(rows)

# Terrain table: one value per indicator
terrain_df = pd.read_csv(TERRAIN_FILE).set_index('INDICATOR')
rows = []
for col, group, label, unit, dec, mult in TERRAIN_SPEC:
    rows.append({'Group': group, 'Indicator': label, 'Value': round(terrain_df.loc[col, 'VALUE'] * mult, dec),
                 'Unit': unit, 'Source': DEM})
terrain_clean_df = pd.DataFrame(rows)

# Summarise + save
print(mars_clean_df.to_string(index=False))
print()
print(terrain_clean_df.to_string(index=False))
mars_clean_df.to_csv(OUT_DIR / 'gale_late_winter_final.csv', index=False)
terrain_clean_df.to_csv(OUT_DIR / 'mount_sharp_terrain_final.csv', index=False)
print('Clean tables saved')