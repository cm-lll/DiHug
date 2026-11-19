"""Summarize saved offline choices with each held-out trajectory weighted equally."""

from itertools import combinations
import numpy as np
import pandas as pd

from run import BUDGETS, EPS, OUT, random_outcome


PRIMARY = 4
METRICS = ['gain', 'gap', 'hit', 'oracle']


def main():
    frame = pd.read_csv(OUT / 'per_state.csv')
    keys = ['panel', 'path', 'step', 'budget']
    assert len(frame) == 2 * 1664 * len(BUDGETS) * 5
    assert frame.groupby(keys).size().eq(5).all()
    assert frame.groupby(keys).oracle.nunique().eq(1).all()
    assert np.allclose(frame.gap, frame.oracle - frame.gain, atol=EPS, rtol=0)
    count = frame[frame.method != 'random'].shortlisted_slots.str.split().str.len().to_numpy()
    assert np.array_equal(count, frame.loc[frame.method != 'random', 'budget'])
    for k in (1, 2, 4):
        test = np.array([0, 0, 0.1, 0.1, 0.2, 0.4, 0.7, 0.9])
        draws = np.array([max(x) for x in combinations(test, k)])
        assert np.allclose(random_outcome(test, k), [draws.mean(), (draws > 0).mean()])

    panels = []
    for label, rows in [('all', frame), ('connected', frame[frame.connected])]:
        paths = rows.groupby(['panel', 'path', 'budget', 'method'], as_index=False).agg(
            states=('step', 'size'), **{k: (k, 'mean') for k in METRICS})
        panels.append(paths.assign(subset=label))
    paths = pd.concat(panels, ignore_index=True)
    paths.to_csv(OUT / 'per_trajectory.csv', index=False)
    combined = pd.concat([paths, paths.assign(panel='pooled')], ignore_index=True)
    summary = combined.groupby(['subset', 'panel', 'budget', 'method'], as_index=False).agg(
        paths=('path', 'size'), states=('states', 'sum'),
        **{k: (k, 'mean') for k in METRICS}, trajectory_gain_sd=('gain', 'std'))
    summary.to_csv(OUT / 'summary.csv', index=False)
    pair_rows = []
    for (subset, panel, k), rows in combined.groupby(['subset', 'panel', 'budget']):
        # For pooled rows, path names can appear in both independent archives.
        if panel == 'pooled':
            local = paths[(paths.subset == subset) & (paths.budget == k)]
            paired = local.pivot(index=['panel', 'path'], columns='method', values='gain')
        else:
            paired = rows.pivot(index='path', columns='method', values='gain')
        for method in paired.columns.drop('learned'):
            diff = paired.learned - paired[method]
            pair_rows.append(dict(subset=subset, panel=panel, budget=k, baseline=method,
                                  paths=len(diff), gain_difference=diff.mean(), sd=diff.std(ddof=1),
                                  wins=(diff > EPS).sum(), ties=(diff.abs() <= EPS).sum(), losses=(diff < -EPS).sum()))
    paired = pd.DataFrame(pair_rows)
    paired.to_csv(OUT / 'paired.csv', index=False)
    print(summary[(summary.budget == PRIMARY) & ((summary.panel == 'pooled') | (summary.subset == 'all'))].to_string(index=False))
    print(paired[(paired.budget == PRIMARY) & (paired.panel == 'pooled')].to_string(index=False))


if __name__ == '__main__':
    main()
