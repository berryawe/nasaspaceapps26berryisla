import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Final combiner: Method A (climate first) and Method B round 2 (terrain first, with analogues and controls) on the same rules,
# with the REMA verdicts for the Antarctic windows. Gives the final top 3 and compares them with the known Mars analogues.
# Defining constants
CLEAN_DIR = BASE_DIR / 'data' / 'clean'
RAW_DIR = BASE_DIR / 'data' / 'raw'
A_FILE = CLEAN_DIR / 'earth_matches_method_a_final.csv'     # Method A tidy table (indicators as rows, one column per site)
B2_FILE = CLEAN_DIR / 'earth_b2_regions.csv'                # Method B round 2: best window per region (candidates, analogues, controls)
REMA_FILE = RAW_DIR / 'rema_check.csv'                      # ASTER checked against REMA at the Antarctic windows
REMA_POINTS_FILE = RAW_DIR / 'rema_points_raw.csv'          # the REMA points (gives each checked window's centre)
OUT_FILE = CLEAN_DIR / 'earth_final_matches.csv'            # every site from both methods, scored on the same rules
TOP_FILE = CLEAN_DIR / 'earth_final_top3_vs_mars.csv'       # top 3 against Gale, with the analogues alongside
TOP_N = 3                                                   # how many best matches to report

# Our priorities, applied the same way to every site
CLIMATE_WEIGHT = 0.6             # 1. weather first: this is a weather-based analogue for drone testing (terrain gets the other 40%)
MAX_NIGHT_MIN_C = -35.0          # 2. a site must pass the climate gates: nights cold enough...
MIN_RANGE_K = 8.0                #    ...and a real day-night cycle (polar night has none)
                                 # 3. only terrain that passed the elevation check can win (ASTER is unreliable over ice; REMA settles Antarctica)
                                 # 4. air density is a gap and a tie-breaker, not part of the score (Earth is always far denser)
REMA_MATCH_DEG = 0.05            # a REMA window matches a site if its centre is within this many degrees
PLATEAU = ('Aksai Chin', 'Tibetan plateau', 'Changtang', 'Kunlun', 'Altun', 'Himalaya', 'Qilian', 'Zanskar', 'Ladakh')   # Tibetan Plateau and its rim
SAME_REGION_KM = 800.0           # a Method A site and a Method B2 site this close count as the same area
NEAR_TIE = 0.01                  # combined scores closer than this are reported as a tie
SWEEP = np.round(np.arange(0.0, 1.01, 0.1), 1)   # weather weights tried, to see whether the winner depends on our choice

# Names for the Method A sites (display only)
A_NAMES = {'30.0°N 85.0°E': 'Tibetan plateau (A)', '55.0°N 65.0°W': 'Labrador-Quebec (A)',
           '70.0°N 80.0°E': 'Northern Siberia (A)', '75.0°S 20.0°E': 'Queen Maud Land, Antarctica (A)',
           '70.0°S 50.0°E': 'East Antarctica, Enderby Land (A)', '70.0°S 130.0°E': 'East Antarctica, Wilkes Land (A)',
           '70.0°S 105.0°E': 'East Antarctica, Wilkes Land 2 (A)', '75.0°N 40.0°W': 'Greenland ice sheet (A)'}

# Indicator names in the Method A tidy table -> short column names
RENAME = {'Climate match': 'CLIMATE_SCORE', 'Terrain match': 'TERRAIN_SCORE', 'Best month': 'BEST_MONTH',
          'Night minimum air temperature': 'T2M_MIN_C', 'Day-night temperature range': 'T2M_RANGE_K',
          'Mean wind speed, 2 m': 'WS2M_MS', 'Wind night/day ratio': 'WIND_NIGHT_DAY_RATIO',
          'Air density compared with Mars': 'DENSITY_VS_MARS', 'Dust optical depth': 'DUST_TAU',
          'Window centre latitude': 'WIN_LAT', 'Window centre longitude': 'WIN_LON', 'Elevation range': 'ELEV_RANGE',
          'Mean slope': 'SLOPE_MEAN', 'Roughness, 1 km window': 'ROUGHNESS_1000M', 'Roughness, 5 km window': 'ROUGHNESS_5000M',
          'Terrain data reliability': 'TERRAIN_DATA'}
NUMBER_COLS = ['CLIMATE_SCORE', 'TERRAIN_SCORE', 'T2M_MIN_C', 'T2M_RANGE_K', 'WS2M_MS', 'WIND_NIGHT_DAY_RATIO', 'DENSITY_VS_MARS',
               'DUST_TAU', 'PRECIP_MM_DAY', 'WIN_LAT', 'WIN_LON', 'ELEV_RANGE', 'SLOPE_MEAN', 'ROUGHNESS_1000M', 'ROUGHNESS_5000M']
# Short headings and decimals for the printed tables
HEAD = {'RANK': 'Rank', 'KIND': 'Kind', 'METHOD': 'Method', 'PLACE': 'Place', 'COMBINED_SCORE': 'Combined', 'CLIMATE_SCORE': 'Climate',
        'TERRAIN_SCORE': 'Terrain', 'T2M_MIN_C': 'Night min (C)', 'T2M_RANGE_K': 'Range (K)', 'WS2M_MS': 'Wind (m/s)',
        'DENSITY_VS_MARS': 'Density (x Mars)', 'DUST_TAU': 'Dust', 'STATUS': 'Status', 'WHY': 'Reason'}
DECIMALS = {'COMBINED_SCORE': 2, 'CLIMATE_SCORE': 2, 'TERRAIN_SCORE': 2, 'T2M_MIN_C': 1, 'T2M_RANGE_K': 1, 'WS2M_MS': 1,
            'DENSITY_VS_MARS': 0, 'DUST_TAU': 2}


def heading(text):
    print(f'\n{text}\n{"-" * len(text)}')


def table(df, cols):
    """Print a table with short headings and sensible decimals."""
    out = df[cols].copy()
    for c in cols:
        if c in DECIMALS:
            out[c] = out[c].map(lambda v: f'{v:.{DECIMALS[c]}f}' if pd.notna(v) else '')
    out = out.rename(columns=HEAD).fillna('')
    print(out.to_string(index=False))


def km_between(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = np.radians([lat1, lon1, lat2, lon2])
    return 2 * 6371 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))


# ---------------------------------------------------------------- Load both methods into one table
a_raw = pd.read_csv(A_FILE)
a_df = a_raw.set_index('Indicator')[list(a_raw.columns[5:])].T.rename(columns=RENAME).reset_index().rename(columns={'index': 'SITE'})
a_df['SITE'] = a_df['SITE'].str.replace(r'^#\d+ ', '', regex=True)
a_df['METHOD'], a_df['KIND'] = 'A', 'Candidate'
a_df['PLACE'] = [A_NAMES.get(s, s + ' (A)') for s in a_df['SITE']]

b_df = pd.read_csv(B2_FILE).rename(columns={'REGION_NAME': 'SITE'})
b_df['METHOD'] = 'B2'
b_df['PLACE'] = b_df['SITE']

keep = ['METHOD', 'KIND', 'SITE', 'PLACE', 'BEST_MONTH', 'TERRAIN_DATA'] + NUMBER_COLS
all_df = pd.concat([a_df.reindex(columns=keep), b_df.reindex(columns=keep)], ignore_index=True)
for c in NUMBER_COLS:
    all_df[c] = pd.to_numeric(all_df[c], errors='coerce')

# ---------------------------------------------------------------- REMA verdicts replace the ASTER 'Unverified' flag
if REMA_FILE.exists() and REMA_POINTS_FILE.exists():
    rema = pd.read_csv(REMA_FILE)
    centres = pd.read_csv(REMA_POINTS_FILE)
    centres = centres[centres['POINT'] == 'window centre'].set_index('LABEL')
    for label, verdict in zip(rema['LABEL'], rema['VERDICT']):
        c_lat, c_lon = centres.loc[label, 'LAT'], centres.loc[label, 'LON']
        hit = (abs(all_df['WIN_LAT'] - c_lat) <= REMA_MATCH_DEG) & (abs(all_df['WIN_LON'] - c_lon) <= REMA_MATCH_DEG)
        if not hit.any():
            print(f'Warning: REMA window "{label}" matches no site')
        elif verdict.startswith('ASTER agrees'):
            all_df.loc[hit, 'TERRAIN_DATA'] = 'Verified (REMA agrees with ASTER)'
        elif verdict.startswith('ASTER disagrees'):
            all_df.loc[hit, 'TERRAIN_DATA'] = 'Disproved (REMA: flat ice sheet)'

# ---------------------------------------------------------------- Same rules for every site
all_df['COMBINED_SCORE'] = CLIMATE_WEIGHT * all_df['CLIMATE_SCORE'] + (1 - CLIMATE_WEIGHT) * all_df['TERRAIN_SCORE']
all_df['GATES_OK'] = (all_df['T2M_MIN_C'] <= MAX_NIGHT_MIN_C) & (all_df['T2M_RANGE_K'] >= MIN_RANGE_K)
all_df['TERRAIN_OK'] = ~all_df['TERRAIN_DATA'].str.startswith(('Unverified', 'Disproved'))
all_df['STATUS'] = np.where(~all_df['GATES_OK'], 'Fails climate gates',
                            np.where(all_df['TERRAIN_DATA'].str.startswith('Disproved'), 'Terrain disproved (REMA)',
                                     np.where(~all_df['TERRAIN_OK'], 'Provisional (terrain unverified)', 'Eligible')))
reasons = []
for r in all_df.itertuples():
    why = []
    if r.T2M_MIN_C > MAX_NIGHT_MIN_C:
        why.append(f'nights only {r.T2M_MIN_C:.0f} C')
    if r.T2M_RANGE_K < MIN_RANGE_K:
        why.append(f'day-night range {r.T2M_RANGE_K:.1f} K')
    if r.TERRAIN_DATA.startswith('Disproved'):
        why.append('REMA shows a flat ice sheet')
    elif not r.TERRAIN_OK:
        why.append('terrain unverified')
    reasons.append(', '.join(why))
all_df['WHY'] = reasons

# Rank the eligible candidates by combined score; ties go to the thinner air (lower density compared with Mars)
all_df = all_df.sort_values(['COMBINED_SCORE', 'DENSITY_VS_MARS'], ascending=[False, True]).reset_index(drop=True)
cand = (all_df['KIND'] == 'Candidate') & (all_df['STATUS'] == 'Eligible')
all_df['RANK'] = pd.Series(np.arange(1, cand.sum() + 1), index=all_df.index[cand]).reindex(all_df.index).astype('Int64')
top = all_df[cand].head(TOP_N).reset_index(drop=True)
analogues = all_df[all_df['KIND'] == 'Analogue']
controls = all_df[all_df['KIND'] == 'Control']

# ---------------------------------------------------------------- Results
print('EARTH LOCATIONS THAT BEST MATCH GALE CRATER (Mount Sharp)')
print('=' * 58)
print(f'Weather {CLIMATE_WEIGHT:.0%} + terrain {1 - CLIMATE_WEIGHT:.0%}  |  nights <= {MAX_NIGHT_MIN_C:.0f} C, day-night range >= {MIN_RANGE_K:.0f} K  |  '
      f'verified terrain only  |  Method A + B2 on the same rules ({int((all_df["KIND"] == "Candidate").sum())} candidates, '
      f'{len(analogues)} analogues, {len(controls)} controls)')

heading(f'TOP {TOP_N} MATCHES')
table(top, ['RANK', 'METHOD', 'PLACE', 'COMBINED_SCORE', 'CLIMATE_SCORE', 'TERRAIN_SCORE', 'T2M_MIN_C', 'T2M_RANGE_K', 'WS2M_MS',
            'DUST_TAU', 'DENSITY_VS_MARS'])
for i in range(len(top) - 1):
    if top.loc[i, 'COMBINED_SCORE'] - top.loc[i + 1, 'COMBINED_SCORE'] < NEAR_TIE:
        print(f'Note: ranks {i + 1} and {i + 2} are within {NEAR_TIE} of each other, effectively a tie.')

heading('OTHER ELIGIBLE CANDIDATES')
rest = all_df[cand].iloc[TOP_N:]
table(rest, ['RANK', 'METHOD', 'PLACE', 'COMBINED_SCORE', 'CLIMATE_SCORE', 'TERRAIN_SCORE']) if len(rest) else print('none')

heading('KNOWN MARS ANALOGUES ON THE SAME RULES (none are candidates; shown for comparison)')
table(analogues, ['PLACE', 'COMBINED_SCORE', 'CLIMATE_SCORE', 'TERRAIN_SCORE', 'T2M_MIN_C', 'T2M_RANGE_K', 'WS2M_MS', 'DUST_TAU',
                  'DENSITY_VS_MARS', 'STATUS'])
w = top.iloc[0]
print(f'\n{w["PLACE"]} against the analogues:')
print(f'  - passes the gates: yes; analogues that pass the gates: {int(analogues["GATES_OK"].sum())} of {len(analogues)}')
print(f'  - combined score {w["COMBINED_SCORE"]:.2f}; analogues range {analogues["COMBINED_SCORE"].min():.2f} to '
      f'{analogues["COMBINED_SCORE"].max():.2f} (median {analogues["COMBINED_SCORE"].median():.2f}); controls {controls["COMBINED_SCORE"].median():.2f}')
close = analogues[(analogues['COMBINED_SCORE'] - w['COMBINED_SCORE']).abs() < NEAR_TIE * 4]
print(f'  - analogues within {NEAR_TIE * 4:.2f} of it on score: {", ".join(close["PLACE"]) if len(close) else "none"}')
better = analogues[analogues['COMBINED_SCORE'] > w['COMBINED_SCORE'] + NEAR_TIE]
if len(better):
    print(f'  - analogues that score higher but fail the gates: ' + ', '.join(f'{p} ({why})' for p, why in zip(better['PLACE'], better['WHY'])))
print('  - the score alone does not separate the winner from the hot analogues; the climate gates do.')

heading('NOT RANKED: PASS THE CLIMATE GATES BUT TERRAIN IS DISPROVED OR UNVERIFIED')
table(all_df[(all_df['KIND'] == 'Candidate') & all_df['STATUS'].isin(['Terrain disproved (REMA)', 'Provisional (terrain unverified)'])],
      ['METHOD', 'PLACE', 'COMBINED_SCORE', 'CLIMATE_SCORE', 'TERRAIN_SCORE', 'WHY'])
heading('NOT RANKED: CANDIDATES THAT FAIL THE CLIMATE GATES (best 12 by score)')
table(all_df[(all_df['KIND'] == 'Candidate') & (all_df['STATUS'] == 'Fails climate gates')].head(12),
      ['METHOD', 'PLACE', 'COMBINED_SCORE', 'CLIMATE_SCORE', 'TERRAIN_SCORE', 'WHY'])

# Do the two methods find the same area?
heading(f'DO THE METHODS AGREE? (best eligible window of each, within {SAME_REGION_KM:.0f} km)')
best_a, best_b = [all_df[cand & (all_df['METHOD'] == m)].head(1) for m in ('A', 'B2')]
if len(best_a) and len(best_b):
    a, b = best_a.iloc[0], best_b.iloc[0]
    km = km_between(a['WIN_LAT'], a['WIN_LON'], b['WIN_LAT'], b['WIN_LON'])
    print(f'Method A best: {a["PLACE"]} ({a["COMBINED_SCORE"]:.2f}); Method B2 best: {b["PLACE"]} ({b["COMBINED_SCORE"]:.2f}); {km:.0f} km apart: '
          f'{"same area" if km <= SAME_REGION_KM else "different areas"}')

# Does the winner depend on how much we weight weather?
heading('DOES THE TOP 3 DEPEND ON THE WEATHER WEIGHT?')
el = all_df[cand]
firsts = []
for wt in SWEEP:
    s = el.assign(S=wt * el['CLIMATE_SCORE'] + (1 - wt) * el['TERRAIN_SCORE']).sort_values(['S', 'DENSITY_VS_MARS'], ascending=[False, True]).head(TOP_N)
    firsts.append(s['PLACE'].iloc[0])
    print(f'Weather {wt:>4.0%} / terrain {1 - wt:>4.0%}:  ' + '  >  '.join(s['PLACE']))
stable = [x for x, p in zip(SWEEP, firsts) if p == top['PLACE'].iloc[0]]
print(f'\n{top["PLACE"].iloc[0]} stays first for weather weights {min(stable):.0%} to {max(stable):.0%}.')
on_plateau = [x for x, p in zip(SWEEP, firsts) if p.startswith(PLATEAU)]
print(f'A Tibetan Plateau site (Method A or B2) is first for weather weights {min(on_plateau):.0%} to {max(on_plateau):.0%}.') if on_plateau else None

# ---------------------------------------------------------------- The top matches and the analogues against Gale, side by side (saved)
mars_ref = a_raw.set_index('Indicator')['Mount Sharp / Gale (Mars)']
side = pd.concat([top, analogues.sort_values('COMBINED_SCORE', ascending=False)], ignore_index=True)
names = [f'#{i + 1} {p}' for i, p in enumerate(top['PLACE'])] + [f'(analogue) {p}' for p in analogues.sort_values('COMBINED_SCORE', ascending=False)['PLACE']]
spec = [('Match', 'Combined score', '0-1', 'COMBINED_SCORE', 2), ('Match', 'Climate match', '0-1', 'CLIMATE_SCORE', 2),
        ('Match', 'Terrain match', '0-1', 'TERRAIN_SCORE', 2), ('Match', 'Status', '', 'STATUS', None),
        ('Climate', 'Best month', '', 'BEST_MONTH', None),
        ('Climate', 'Night minimum air temperature', 'deg C', 'T2M_MIN_C', 1), ('Climate', 'Day-night temperature range', 'K', 'T2M_RANGE_K', 1),
        ('Climate', 'Mean wind speed, 2 m', 'm/s', 'WS2M_MS', 1), ('Climate', 'Wind night/day ratio', '', 'WIND_NIGHT_DAY_RATIO', 2),
        ('Climate', 'Dust optical depth', '', 'DUST_TAU', 3), ('Climate', 'Rain', 'mm/day', 'PRECIP_MM_DAY', 2),
        ('Climate', 'Air density compared with Mars', 'x Mars', 'DENSITY_VS_MARS', 0),
        ('Terrain', 'Elevation range', 'm', 'ELEV_RANGE', 0), ('Terrain', 'Mean slope', 'deg', 'SLOPE_MEAN', 1),
        ('Terrain', 'Roughness, 1 km window', 'm', 'ROUGHNESS_1000M', 1), ('Terrain', 'Roughness, 5 km window', 'm', 'ROUGHNESS_5000M', 1),
        ('Location', 'Best window latitude', 'deg', 'WIN_LAT', 2), ('Location', 'Best window longitude', 'deg', 'WIN_LON', 2),
        ('Check', 'Terrain data reliability', '', 'TERRAIN_DATA', None)]
rows = []
for group, indicator, unit_text, col, dec in spec:
    mars_value = mars_ref.get(indicator, '')
    mars_value = '' if pd.isna(mars_value) else mars_value
    values = [('' if pd.isna(v) else v) if dec is None else ('' if pd.isna(v) else (int(round(v)) if dec == 0 else round(float(v), dec)))
              for v in side[col]]
    rows.append([group, indicator, unit_text, mars_value] + values)
top_df = pd.DataFrame(rows, columns=['Group', 'Indicator', 'Unit', 'Gale / Mount Sharp (Mars)'] + names)
heading(f'TOP {TOP_N} AND THE BEST ANALOGUES AGAINST GALE')
shown = ['Indicator', 'Unit', 'Gale / Mount Sharp (Mars)'] + names[:TOP_N] + names[TOP_N:TOP_N + 2]
print(top_df[shown].to_string(index=False))

# Save
all_df[['RANK', 'STATUS', 'WHY', 'KIND', 'METHOD', 'PLACE', 'COMBINED_SCORE', 'CLIMATE_SCORE', 'TERRAIN_SCORE', 'BEST_MONTH'] + NUMBER_COLS[2:] +
       ['TERRAIN_DATA']].round(3).to_csv(OUT_FILE, index=False)
top_df.to_csv(TOP_FILE, index=False)
print(f'\nSaved {OUT_FILE.name} (all {len(all_df)} sites) and {TOP_FILE.name} (top {TOP_N} and all {len(analogues)} analogues against Gale) in data/clean')