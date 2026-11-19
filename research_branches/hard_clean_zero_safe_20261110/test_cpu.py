"""CPU-only algebra, independent primal solver, and historical-Gram checks."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, itertools
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from combine import project_mean, loo_gradient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research_branches/hard_clean_pilot_v2_20261110'))
from objective import solve_gram, loo_gradient as previous_loo


def run():
    out = ROOT/'results/hard_clean_zero_safe_20261110'
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026111001)
    # Zero group, true opposition, redundant constraints, and all-zero case.
    matrices = [np.array([[1., 0], [0, 1], [0, 0]]),
                np.array([[1., 0], [-1, 0], [0, 0]]),
                np.array([[1., 2], [1, 2], [-1, 1], [0, 0]]),
                np.zeros((5, 3))]
    for j in range(200):
        g = rng.normal(size=(5, 9))
        if j % 2 == 0: g[-1] = 0
        if j % 3 == 0: g[-2] = g[0]
        if j % 7 == 0: g[1] = -g[0]
        matrices.append(g)
    max_solver_error = 0.
    for g in matrices:
        coeff, meta = project_mean(g @ g.T)
        v = coeff @ g; b = g.mean(0)
        assert np.min(g @ v) >= -1e-7
        assert np.linalg.norm(v) <= np.linalg.norm(b) + 1e-7
        assert abs(b @ v - v @ v) <= 1e-7
        # Independent primal SLSQP, no shared dual/active-set implementation.
        independent = minimize(lambda x: .5 * np.sum((x-b)**2), b,
            jac=lambda x: x-b, constraints=[dict(type='ineq', fun=lambda x:g@x,
            jac=lambda x:g)], method='SLSQP', options=dict(ftol=1e-12, maxiter=1000))
        assert independent.success, independent.message
        err = float(np.linalg.norm(independent.x-v)); max_solver_error=max(err,max_solver_error)
        assert err < 1e-5
    g = matrices[0]; old, _ = solve_gram(g@g.T); new, _ = project_mean(g@g.T)
    assert np.linalg.norm(old@g) < 1e-10 and np.linalg.norm(new@g) > .4
    # Enumerated expectation: estimate remains the exact score-function gradient.
    p = np.array([.2, .6, .8]); graphs = np.array(list(itertools.product([0.,1.],repeat=3)))
    risk = (graphs.sum(1)-1.)**2
    mass = np.prod(np.where(graphs, p, 1-p), axis=1)
    exact = (mass*risk) @ (graphs-p)
    average = np.zeros(3)
    for i,j in itertools.product(range(8),repeat=2):
        average += mass[i]*mass[j]*loo_gradient(p,graphs[[i,j]],risk[[i,j],None])[0]
    assert np.max(abs(average-exact)) < 1e-14
    bits=rng.binomial(1,.5,(16,31)); costs=rng.normal(size=(16,5)); prob=np.full(31,.5)
    assert np.max(abs(loo_gradient(prob,bits,costs)-previous_loo(prob,bits,costs))) < 1e-14
    constant=np.full((16,5),4/191**2)
    assert np.count_nonzero(loo_gradient(prob,bits,constant)) == 0
    history=[]
    for path in sorted((ROOT/'results/hard_clean_pilot_v2_20261110').glob('*HARD_MGDA/updates.jsonl')):
        for line in path.read_text().splitlines():
            r=json.loads(line); h=np.array(r['Gram']); c, meta=project_mean(h)
            history.append(dict(task=path.parent.name,update=r['update'],coefficients=c.tolist(),**meta))
    result=dict(passed=True,synthetic_cases=len(matrices),independent_solver_max_error=max_solver_error,
        enumerated_LOO_max_error=float(np.max(abs(average-exact))),constant_cost_exact_zero=True,
        zero_group_witness=dict(old_norm=float(np.linalg.norm(old@g)),new_norm=float(np.linalg.norm(new@g))),
        historical_Gram_cases=len(history),historical=history,
        caveat='No model invocation; no actual-AdamW or quality guarantee.')
    (out/'cpu_tests.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='historical'}))

if __name__=='__main__':run()
