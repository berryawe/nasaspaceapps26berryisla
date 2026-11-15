import pandas as pd
import numpy as np
from pathlib import Path
import requests
import re
import math

BASE_DIR = Path(__file__).resolve().parent.parent

# Defining constants
BASE_URL = 'https://atmos.nmsu.edu/PDS/data/mslrem_1001/DATA/'

# Late southern winter (Ls 135-180) in Mars Year 38 (Earth 12/11/2024 to 30/9/2026) - start, middle and end of the window
TARGETS = {'begin': 4650, 'middle': 4693, 'end': 4730}
SOL_RETRIES = 4
EXPECTED_FORMAT = 'MODRDR6.FMT'
R_MARS_AIR = 191.0        # J/(kg K), specific gas constant of Mars air (approx.)
EARTH_SEA_LEVEL_DENSITY = 1.225   # kg/m3
NIGHT_HOURS = {18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5}   # Local mean solar time hours counted as night (sunset: 18, sunrise: 06)

RAW_DIR = BASE_DIR / 'data' / 'raw' / 'rems_modrdr'
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = BASE_DIR / 'data' / 'raw'

# All 40 columns of the MODRDR table, in file order (from the product's XML label)
# The full list of columns used line up with the right position in the file
COLUMNS = [
    'TIMESTAMP', 'LMST', 'LTST',
    'HORIZONTAL_WIND_SPEED', 'VERTICAL_WIND_SPEED', 'WIND_DIRECTION', 'WS_CONFIDENCE_LEVEL',
    'BRIGHTNESS_TEMP', 'BRIGHTNESS_TEMP_LONG_TERM_UNCERTAINTY', 'BRIGHTNESS_TEMP_SHORT_TERM_UNCERTAINTY', 'GTS_CONFIDENCE_LEVEL',
    'BOOM1_LOCAL_AIR_TEMP', 'ATS_BOOM1_CONFIDENCE_LEVEL', 'BOOM2_LOCAL_AIR_TEMP', 'ATS_BOOM2_CONFIDENCE_LEVEL',
    'AMBIENT_TEMP', 'AMBIENT_TEMP_CONFIDENCE_LEVEL',
    'UV_A', 'UV_B', 'UV_C', 'UV_ABC', 'UV_D', 'UV_E',
    'UV_A_UNCERTAINTY', 'UV_B_UNCERTAINTY', 'UV_C_UNCERTAINTY', 'UV_ABC_UNCERTAINTY', 'UV_D_UNCERTAINTY', 'UV_E_UNCERTAINTY',
    'UVS_CONFIDENCE_LEVEL',
    'LOCAL_RELATIVE_HUMIDITY', 'HS_TEMP', 'LOCAL_RELATIVE_HUMIDITY_UNCERTAINTY',
    'VOLUME_MIXING_RATIO', 'VOLUME_MIXING_RATIO_UNCERTAINTY', 'HS_CONFIDENCE_LEVEL',
    'PS_CONFIGURATION', 'PRESSURE', 'PRESSURE_UNCERTAINTY', 'PS_CONFIDENCE_LEVEL',
]

STRING_COLS = ['LMST', 'LTST', 'PS_CONFIGURATION'] + [c for c in COLUMNS if c.endswith('CONFIDENCE_LEVEL')]
UNKNOWN = -999.0          # "unknown" constant for numeric columns

# Only load needed columns 
USE_COLS = ['TIMESTAMP', 'LMST', 'AMBIENT_TEMP', 'AMBIENT_TEMP_CONFIDENCE_LEVEL', 'PRESSURE']
STRING_USED = [c for c in STRING_COLS if c in USE_COLS]

# Defining functions
def solar_longitude(j2000_seconds):
    """Mars areocentric solar longitude (Ls, degrees) - Allison & McEwen (2000) formula."""
    t = j2000_seconds / 86400.0
    M = (19.3871 + 0.52402073 * t) % 360
    alpha_fms = (270.3863 + 0.52403840 * t) % 360
    A = [0.0071, 0.0057, 0.0039, 0.0037, 0.0021, 0.0020, 0.0018]
    tau = [2.2353, 2.7543, 1.1177, 15.7866, 2.1354, 2.4694, 32.8493]
    phi = [49.409, 168.173, 191.837, 21.736, 15.704, 95.528, 49.095]
    pbs = sum(a * math.cos(math.radians(0.985626 * t / ta + p)) for a, ta, p in zip(A, tau, phi))
    r = math.radians(M)
    nu_m = ((10.691 + 3.0e-7 * t) * math.sin(r) + 0.623 * math.sin(2 * r) + 0.050 * math.sin(3 * r)
            + 0.005 * math.sin(4 * r) + 0.0005 * math.sin(5 * r) + pbs)
    return (alpha_fms + nu_m) % 360

def summarise_sol(sol, tab_path):
    """One sol's table correspods to one row of indicators."""
    df = pd.read_csv(tab_path, header=None, names=COLUMNS, usecols=USE_COLS, skipinitialspace=True,
                     dtype={c: str for c in STRING_USED})
    num_cols = [c for c in USE_COLS if c not in STRING_USED]
    df[num_cols] = df[num_cols].replace(UNKNOWN, np.nan)
    for c in STRING_USED:
        df[c] = df[c].str.strip()

    hour = df['LMST'].str.extract(r'M(\d{2}):')[0].astype(float)
    amb, p = df['AMBIENT_TEMP'], df['PRESSURE']
    rho = p / (R_MARS_AIR * amb)                                   # air density, kg/m3
    reliable = df['AMBIENT_TEMP_CONFIDENCE_LEVEL'].str[-1] == '1'  # is observation "reliable"?
    night = hour.isin(NIGHT_HOURS)

    night_min = amb[night].quantile(0.01)                          # 1st percentile ignores single-reading anomalies
    day_max = amb.quantile(0.99)
    return {
        'SOL': sol,
        'LS': solar_longitude(df['TIMESTAMP'].mean()),
        'ROWS': len(df),
        'HOURS_COVERED': hour.nunique(),
        'NIGHT_ROWS': int(night.sum()),
        'AIR_T_RELIABLE_SHARE': reliable.mean(),
        'P_MEAN': p.mean(), 'P_MIN': p.min(), 'P_MAX': p.max(),
        'RHO_MEAN': rho.mean(), 'RHO_MIN': rho.min(),
        'RHO_VS_EARTH': rho.mean() / EARTH_SEA_LEVEL_DENSITY,
        'AIR_T_NIGHT_MIN_C': night_min - 273.15,
        'AIR_T_DAY_MAX_C': day_max - 273.15,
        'AIR_T_RANGE_K': day_max - night_min,
    }

# Block out folders from the DATA index page
index_html = requests.get(BASE_URL, timeout=60).text
blocks = sorted(set((int(a), int(b), f'SOL_{a}_{b}') for a, b in re.findall(r'SOL_(\d{5})_(\d{5})/', index_html)))
if not blocks:
    raise RuntimeError('No SOL_xxxxx_xxxxx folders found at BASE_URL - page layout may have changed')

rows = []
for label, target_sol in TARGETS.items():
    for sol in range(target_sol, target_sol + SOL_RETRIES):
        block = [name for a, b, name in blocks if a <= sol <= b]
        if not block:
            print(f'{label}: sol {sol} has no block folder'); continue
        sol_url = f'{BASE_URL}{block[0]}/SOL{sol:05d}/'
        resp = requests.get(sol_url, timeout=60)
        if resp.status_code == 404:
            print(f'{label}: sol {sol} has no folder, trying next'); continue
        resp.raise_for_status()

        # File names contain a changing number, so find the MODRDR (RMD) table in the listing
        tabs = re.findall(r'href="(RME_[^"]*RMD[^"]*\.TAB)"', resp.text)
        if not tabs:
            listed = re.findall(r'href="(RME_[^"]*)"', resp.text)
            print(f'{label}: sol {sol} has no RMD .TAB. Files listed: {listed[:8]}'); continue
        tab_name = sorted(tabs)[-1]
        lbl_name = tab_name.replace('.TAB', '.LBL')

        # Double-check the column layout version
        lbl_text = requests.get(sol_url + lbl_name, timeout=60).text
        fmt = re.search(r'\^STRUCTURE\s*=\s*"([^"]+)"', lbl_text)
        if not fmt or fmt.group(1) != EXPECTED_FORMAT:
            print(f'{label}: sol {sol} format {fmt.group(1) if fmt else None}, not {EXPECTED_FORMAT}, trying next'); continue

        tab_path = RAW_DIR / tab_name
        if not tab_path.exists():
            with requests.get(sol_url + tab_name, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tab_path, 'wb') as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        row = summarise_sol(sol, tab_path)
        row['LABEL'], row['TARGET_SOL'] = label, target_sol
        rows.append(row)
        break
    else:
        print(f'{label}: no usable sol found in {target_sol} to {target_sol + SOL_RETRIES - 1}')

if not rows:
    raise RuntimeError('No sols were processed - see messages above')

gale_daily_df = pd.DataFrame(rows)
gale_daily_df = gale_daily_df[['LABEL', 'TARGET_SOL'] + [c for c in gale_daily_df.columns if c not in ('LABEL', 'TARGET_SOL')]]

# Indicator table: one row per sol per indicator
UNITS = {'P_MEAN': 'Pa', 'P_MIN': 'Pa', 'RHO_MEAN': 'kg/m3', 'RHO_MIN': 'kg/m3', 'RHO_VS_EARTH': 'fraction of Earth sea level',
         'AIR_T_NIGHT_MIN_C': 'C', 'AIR_T_DAY_MAX_C': 'C', 'AIR_T_RANGE_K': 'K'}
gale_indicators_df = gale_daily_df.melt(id_vars=['LABEL', 'SOL', 'LS'], value_vars=list(UNITS),
                                        var_name='INDICATOR', value_name='VALUE')
gale_indicators_df['UNIT'] = gale_indicators_df['INDICATOR'].map(UNITS)

# Summarise + save
print(gale_daily_df[['LABEL', 'SOL', 'LS', 'HOURS_COVERED', 'P_MEAN', 'RHO_MEAN', 'AIR_T_NIGHT_MIN_C', 'AIR_T_RANGE_K']].round(3).to_string(index=False))
gale_daily_df.to_csv(OUT_DIR / 'gale_late_winter_daily.csv', index=False, na_rep='NaN')
gale_indicators_df.to_csv(OUT_DIR / 'gale_late_winter_indicators.csv', index=False, na_rep='NaN')
print('Gale crater late-southern-winter data fetched and saved')