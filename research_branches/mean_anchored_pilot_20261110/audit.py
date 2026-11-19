"""Read-only independent reconstruction of candidate geometry and scored outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,csv,time,hashlib,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/mean_anchored_pilot_20261110_attempt2'
start=time.time();dec=list(csv.DictReader((OUT/'decisions.csv').open()));rows=list(csv.DictReader((OUT/'scores.csv').open()));checked=[];norm_gaps=[]
for r in dec:
    key=f"{r['dataset']}_{r['seed']}_{r['panel']}";x=np.load(OUT/f'{key}.npz');p=x['p_unchanged'];a=(p>0)&(p<1);z=np.log(p[a])-np.log1p(-p[a]);G=x['gradients'];m=G.mean(0);eta=float(r['eta'])
    directions={}
    for arm in ('cagrad','average','same_norm_average'):
        q=x['p_'+arm];assert np.array_equal(p[~a],q[~a]);v=np.zeros_like(p);v[a]=np.log(q[a])-np.log1p(-q[a])-z;directions[arm]=v
    d=-directions['cagrad']/eta
    assert np.linalg.norm(d-m)<=.5*np.linalg.norm(m)+1e-9
    assert m@d>=(.5-1e-8)*(m@m)
    np.testing.assert_allclose(directions['average'],-eta*m,atol=1e-9,rtol=1e-7)
    gap=abs(np.linalg.norm(directions['cagrad'])-np.linalg.norm(directions['same_norm_average']))
    norm_gaps.append((float(gap),float(gap/max(1,float(r['original_norm'])))))
    # Reuse the worker's pre-existing scale-relative serialization tolerance.
    assert gap<1e-7*max(1,float(r['original_norm']))
    cert=json.loads((OUT/f'{key}_solver.json').read_text());w=np.array(cert['multipliers']);assert min(w)>=0 and abs(w.sum()-1)<1e-10
    gw=w@G;upper=gw@m+.5*np.linalg.norm(m)*np.linalg.norm(gw);primal=min(G@d)
    assert abs(upper-primal)<=cert['tolerance']*(m@m)+1e-10
    values=x['values'];cost=x['evaluation_remaining_costs']
    for i,arm in enumerate(x['arms']):
        np.testing.assert_allclose(values[i,[1,4]],cost[i].mean(0),rtol=1e-12,atol=1e-12)
        rr=next(v for v in rows if v['dataset']==r['dataset'] and v['seed']==r['seed'] and v['panel']==r['panel'] and v['arm']==arm)
        assert abs(float(rr['mean_five'])-values[i].mean())<1e-12
    checked.append(key)
bindings=json.loads((OUT/'bindings.json').read_text())
for p,h in bindings.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
result=dict(passed=True,conditions=len(checked),checks=['support preservation','actual stored logit displacement','mean progress bound','same norm control','dual upper bound/primal gap','sample split means and CSV','bound asset hashes'],max_norm_gap_absolute=max(x[0] for x in norm_gaps),max_norm_gap_relative=max(x[1] for x in norm_gaps),asset_count=len(bindings),seconds=time.time()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0)
(OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
