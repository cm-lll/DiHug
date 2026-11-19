"""Scalar algebra illustration only; no graph/model evaluation or optimization run."""
import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
start = time.perf_counter()
# L1(x)=(x+1)^2/2; L2(x)=(x-.2)^2/2. At x=0 the
# constraints g_i*d >= 0 force d=0. CAGrad's c=.5 interval
# is [.2,.6]; min(d,-.2*d) is maximized at d=.2.
g = [1.0, -0.2]
g0 = sum(g) / 2
c = 0.5
d = g0 * (1-c)
loss = lambda x: [(x+1)**2/2, (x-.2)**2/2]
records = []
for eta in (0.1, 5.0):
    before, after = loss(0.0), loss(-eta*d)
    records.append(dict(eta=eta, before=before, after=after,
                        mean_before=sum(before)/2, mean_after=sum(after)/2))
assert math.isclose(g0*d, (1-c)*g0*g0)
assert records[0]['mean_after'] < records[0]['mean_before']
assert records[0]['after'][1] > records[0]['before'][1]
assert records[1]['mean_after'] > records[1]['mean_before']
paths = ['results/coordinate_joint_pilot_20261110/REPORT_ZH.md',
         'research_branches/coordinate_joint_pilot_20261110/README.md']
out = dict(status='literature_and_algebra_review_complete',
           scope='No new graph evaluation, model calls, training, or GPU use',
           scalar_illustration=dict(gradients=g, average_gradient=g0, c=c,
                                    selected_descent_vector=d, steps=records),
           conclusions=['A strict common descent constraint can force zero.',
                        'An average anchor can retain progress with a task cost.',
                        'First-order descent alone does not ensure finite-step descent.'],
           graph_benefit_verified=False, formal_training_authorized=False,
           future_candidate='CAGrad-style average-progress anchor; AdamW mapping unresolved',
           sources=['https://proceedings.neurips.cc/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf',
                    'https://papers.neurips.cc/paper_files/paper/2023/file/b2fe1ee8d936ac08dd26f2ff58986c8f-Paper-Conference.pdf',
                    'https://raw.githubusercontent.com/Cranial-XIX/FAMO/main/experiments/nyuv2/trainer.py'],
           asset_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
           cpu_elapsed_seconds=time.perf_counter()-start, gpu_hours=0)
dest = ROOT / 'results/mean_anchored_review_20261110'
dest.mkdir(parents=True, exist_ok=True)
(dest/'review.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
