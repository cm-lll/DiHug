"""Summarize paired readouts by connected start and fitting archive."""

import json

import pandas as pd

from run import OUT

EPS = 1e-12


def main():
    verification = json.loads((OUT / 'verification.json').read_text())
    x = pd.read_csv(OUT / 'per_state.csv.gz')
    assert verification['states'] == len(x) == 24 * 2048
    assert x.groupby(['start','archive_arm']).step.nunique().eq(2048).all()
    assert x[['start','archive_arm','step']].duplicated().sum() == 0
    rows = []
    for k in (2, 4):
        for arm in ('pilot', 'bank4', 'residual'):
            chosen = x[f'{arm}_{k}_slots'].str.split()
            assert chosen.str.len().eq(k).all() and chosen.map(lambda values: len(set(values))).eq(k).all()
            if k == 4:
                two = x[f'{arm}_2_slots'].str.split().map(tuple)
                assert chosen.map(lambda slots: tuple(slots[:2])).equals(two)
                assert (x[f'{arm}_4_gain'] >= x[f'{arm}_2_gain']).all()
        for fit in ('pilot', 'bank4'):
            panel = x[['start', 'archive_arm', 'step']].copy()
            panel['fit'], panel['k'] = fit, k
            panel['learned'], panel['residual'] = x[f'{fit}_{k}_gain'], x[f'residual_{k}_gain']
            panel['paired_gain'] = panel.learned - panel.residual
            panel['learned_hit'], panel['residual_hit'] = panel.learned > EPS, panel.residual > EPS
            panel['step_win'], panel['step_loss'] = panel.paired_gain > EPS, panel.paired_gain < -EPS
            rows.append(panel)
    frame = pd.concat(rows, ignore_index=True)
    fields = ['learned', 'residual', 'paired_gain', 'learned_hit', 'residual_hit', 'step_win', 'step_loss']
    trajectories = frame.groupby(['start', 'archive_arm', 'fit', 'k'], as_index=False)[fields].mean()
    trajectories.to_csv(OUT/'per_trajectory.csv', index=False)
    fits = trajectories.groupby(['start','fit','k'], as_index=False)[fields].mean()
    fits.to_csv(OUT/'per_fit_start.csv', index=False)
    starts = fits.groupby(['start','k'], as_index=False)[fields].mean()
    counts = x.groupby('start', as_index=False).agg(paths=('archive_arm','nunique'), states=('step','size'))
    starts = starts.merge(counts, on='start', validate='m:1')
    starts.to_csv(OUT/'per_start.csv', index=False)
    tests = []
    for k in (2, 4):
        c = starts[starts.start.str.endswith('_C') & (starts.k == k)]
        fi = fits[fits.start.str.endswith('_C') & (fits.k == k)].groupby('fit').paired_gain.mean()
        assert set(c.start) == {'13002_C','15001_C','15002_C'}
        tests.append(dict(k=k, mean_learned=c.learned.mean(), mean_residual=c.residual.mean(),
                          c_start_wins=int((c.paired_gain > EPS).sum()), c_start_total=len(c),
                          mean_paired_gain=c.paired_gain.mean(), fits_c_mean_differences=fi.to_dict(),
                          passed=bool((c.paired_gain > EPS).all() and (fi > EPS).all())))
    (OUT/'decision.json').write_text(json.dumps(tests, indent=2, allow_nan=False)+'\n')
    print(starts.round(7).to_string(index=False))
    print(json.dumps(tests, indent=2))


if __name__ == '__main__':
    main()
