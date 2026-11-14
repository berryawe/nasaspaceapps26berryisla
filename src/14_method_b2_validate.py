import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Does the winner of Method B round 2 depend on where we put the climate gates (and on the weather weight)?
# Uses the windows already scored by the B2 search: no downloads, no new requests.
# Defining constants
CLEAN_DIR = BASE_DIR / 'data' / 'clean'
IN_FILE = CLEAN_DIR / 'earth_b2_top_windows.csv'            # the 162 chosen windows with climate and terrain scores
OUT_FILE = CLEAN_DIR / 'earth_b2_gate_sensitivity.csv'      # winner for every gate combination

DEFAULT_NIGHT_C, DEFAULT_RANGE_K = -35.0, 8.0               # the gates used in the B2 run
NIGHT_GATES_C = [-30, -32, -34, -35, -36, -38, -40]         # a window's night minimum must be at or below this
RANGE_GATES_K = [5, 6, 8, 10, 12]                           # a window's day-night range must be at or above this
CLIMATE_WEIGHT = 0.6                                        # weight used in the B2 run (terrain gets the rest)
WEIGHT_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]          # weather weights tried at the default gates
# Candidate regions on or at the edge of the Tibetan Plateau, to see whether "the plateau" wins even when the exact site changes
PLATEAU = ['Aksai Chin', 'Changtang plateau', 'Kunlun', 'Altun Shan', 'Himalaya', 'Qilian Shan', 'Zanskar-Spiti', 'Ladakh']

df = pd.read_csv(IN_FILE)
df = df[(df['KIND'] == 'Candidate') & ~df['TERRAIN_DATA'].str.startswith('Unverified')].copy()
df['REGION'] = df['REGION_NAME'].str.split(',').str[0].str.split('(').str[0].str.strip()   # short label for printing
print(f'{len(df)} candidate windows in {df["REGION"].nunique()} regions with verified terrain')


def ranking(night, rng, weight):
    """Candidate regions that pass the gates, best window of each, best combined score first (ties: thinner air)."""
    ok = df[(df['T2M_MIN_C'] <= night) & (df['T2M_RANGE_K'] >= rng)].copy()
    if ok.empty:
        return ok
    ok['SCORE'] = weight * ok['CLIMATE_SCORE'] + (1 - weight) * ok['TERRAIN_SCORE']
    best = ok.loc[ok.groupby('REGION')['SCORE'].idxmax()]
    return best.sort_values(['SCORE', 'DENSITY_VS_MARS'], ascending=[False, True]).reset_index(drop=True)


def is_plateau(region):
    return any(region.startswith(p) for p in PLATEAU)


# ---------------------------------------------------------------- Gate grid at the default weather weight
rows = []
for night in NIGHT_GATES_C:
    for rng in RANGE_GATES_K:
        r = ranking(night, rng, CLIMATE_WEIGHT)
        rows.append({'NIGHT_GATE_C': night, 'RANGE_GATE_K': rng, 'REGIONS_PASSING': len(r),
                     'WINNER': r['REGION'].iloc[0] if len(r) else '', 'SCORE': r['SCORE'].iloc[0] if len(r) else np.nan,
                     'RUNNER_UP': r['REGION'].iloc[1] if len(r) > 1 else '',
                     'MARGIN': r['SCORE'].iloc[0] - r['SCORE'].iloc[1] if len(r) > 1 else np.nan,
                     'PLATEAU_WINS': bool(len(r)) and is_plateau(r['REGION'].iloc[0])})
grid = pd.DataFrame(rows)

print(f'\nWINNER FOR EACH GATE COMBINATION (weather weight {CLIMATE_WEIGHT:.0%}; the B2 run used night <= {DEFAULT_NIGHT_C:.0f} C, range >= {DEFAULT_RANGE_K:.0f} K)')
winners = grid.pivot(index='NIGHT_GATE_C', columns='RANGE_GATE_K', values='WINNER')
winners.index = [f'night <= {n} C' for n in winners.index]
winners.columns = [f'range >= {k} K' for k in winners.columns]
print(winners.to_string())

print('\nNUMBER OF CANDIDATE REGIONS THAT PASS (how sharp the cliff is)')
passing = grid.pivot(index='NIGHT_GATE_C', columns='RANGE_GATE_K', values='REGIONS_PASSING')
passing.index = [f'night <= {n} C' for n in passing.index]
passing.columns = [f'range >= {k} K' for k in passing.columns]
print(passing.to_string())

print('\nHOW OFTEN EACH REGION WINS (out of all gate combinations that leave at least one region)')
valid = grid[grid['WINNER'] != '']
wins = valid['WINNER'].value_counts()
print(pd.DataFrame({'Wins': wins, 'Share': (wins / len(valid)).map('{:.0%}'.format)}).to_string())
print(f'\nA Tibetan Plateau region wins {valid["PLATEAU_WINS"].mean():.0%} of combinations '
      f'({", ".join(PLATEAU)}).')
default = grid[(grid['NIGHT_GATE_C'] == DEFAULT_NIGHT_C) & (grid['RANGE_GATE_K'] == DEFAULT_RANGE_K)].iloc[0]
print(f'At the default gates the winner is {default["WINNER"]} ({default["SCORE"]:.2f}), ahead of {default["RUNNER_UP"]} by {default["MARGIN"]:.2f}.')
print(f'Margin over the runner-up across combinations: median {valid["MARGIN"].median():.3f}, smallest {valid["MARGIN"].min():.3f}.')

# ---------------------------------------------------------------- Weather weight at the default gates
print(f'\nWEATHER WEIGHT AT THE DEFAULT GATES (night <= {DEFAULT_NIGHT_C:.0f} C, range >= {DEFAULT_RANGE_K:.0f} K): top 3')
for w in WEIGHT_SWEEP:
    r = ranking(DEFAULT_NIGHT_C, DEFAULT_RANGE_K, w)
    print(f'Weather {w:>4.0%} / terrain {1 - w:>4.0%}:  ' + '  >  '.join(f'{a} ({s:.2f})' for a, s in zip(r['REGION'].head(3), r['SCORE'].head(3))))

print('\nHow to read this: if the same region or family keeps winning as the gates move, the answer is robust.')
print('If the winner changes, name the region rather than one site. Months are the ones chosen in the B2 run, so very loose or tight')
print('gates are approximate (a window might pass in a month that was not the best-scoring one).')

grid.round(3).to_csv(OUT_FILE, index=False)
print(f'\nSaved {OUT_FILE.name} in data/clean')