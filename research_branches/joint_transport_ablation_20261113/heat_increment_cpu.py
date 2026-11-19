"""Bounded offline temporal increment metric versus an amplitude-only control.

All graph states pre-exist. No network import, backward, graph construction or
training. Not a production scorer. The temporal transform is invertible, but
Euclidean distances deliberately change; no quality guarantee is claimed.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import json
import resource
import signal
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import calibration_reuse_audit as assets

ROOT, SRC, CAL, ROB = assets.ROOT, assets.SRC, assets.CAL, assets.ROB
OUT = ROOT / 'results/heat_increment_cpu_20261113'

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))

def table(name, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)

def temporal(h):
    n = len(h)
    return np.concatenate((h[:, :9, None], h[:, 9:].reshape(n, 9, -1)), axis=2)

def flatten(h):
    return np.concatenate((h[:, :, 0], h[:, :, 1:].reshape(len(h), -1)), axis=1)

def increment(h):
    x = temporal(h)
    return flatten(np.concatenate((x[:, :1], np.diff(x, axis=1)), axis=1))

def matrices(sc, b):
    h, d = sc.features(b); dh = increment(h); rh = increment(sc.rh)
    out = []
    for typ, (scale, db, hb) in sc.scales.items():
        ix = np.flatnonzero(sc.types == typ)
        gd = cdist(d[ix]/scale/np.sqrt(db), sc.rd[ix]/scale/np.sqrt(db), 'sqeuclidean')
        gh = cdist(h[ix]/np.sqrt(hb), sc.rh[ix]/np.sqrt(hb), 'sqeuclidean')
        gi = cdist(dh[ix]/np.sqrt(hb), rh[ix]/np.sqrt(hb), 'sqeuclidean')
        assert np.isfinite(gi).all() and (gi <= 4*gh + 1e-8).all()
        out.append((int(typ), ix, gd, gh, gi))
    return out, h, d

def solved(ms, coef, mode):
    parts=[];orders=[]
    for typ, ix, gd, gh, gi in ms:
        heat = gh if mode == 'original' else gi if mode == 'increment' else coef*gh
        ii, jj = linear_sum_assignment(gd+heat)
        assert len(set(jj)) == len(ix)
        dc, hc = float(gd[ii,jj].mean()), float(heat[ii,jj].mean())
        parts.append(dict(type=typ, degree=dc, heat=hc, score=dc+hc))
        orders.append((ix, jj))
    return {k: float(np.mean([p[k] for p in parts])) for k in ('degree','heat','score')}, parts, orders

def main():
    start=time.monotonic(); OUT.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600-second CPU bound')))
    signal.alarm(600)
    for p in (Path(__file__),Path(assets.__file__),Path(__file__).with_name('transport.py'),Path(assets.heatmod.__file__)):
        assets.bind(p)
    dump('protocol.json',dict(version='first_heat_level_plus_adjacent_increments_v1',
        transform='For each return/type-arrival channel x1..x9: x1, x2-x1,...,x9-x8. Keep original reference scales and block factors.',
        objective='D+increment_heat, same type-preserving joint assignment; no refitting bandwidth or searching coefficient',
        control='D+cH. c=median across historical development graphs of increment heat/original heat at ORIGINAL joint matching. Frozen before remaining re-scoring.',
        calibration='64 already viewed historical development relocations; not new holdout, no generated outputs used',
        panel='128 old relocation states +120 prior add/delete/concentrate/swap states +156 prior stress states +64 old generation outputs',
        criteria=['identity/permutation and transform inversion', 'finite cost and fixed support',
                  'retain observed degree-preserving distinction and severe disconnection response',
                  'compare old output preference with amplitude-only control; no improvement claim from score reduction alone'],
        limitations=['correlated times do not by themselves prove erroneous counting',
                     'equal log-time Euclidean integration and increment norm are different design choices',
                     'suppressing persistent discrepancies can weaken legitimate large-scale structure penalties',
                     'no new information, no automatic statistical whitening or solver guarantee on quality'],
        CPU_threads=1,memory_GiB=2,max_wall_seconds=600,model_calls=0,GPU_hours=0,new_graphs=0))
    parents=assets.read(SRC/'protocol.json')['parents']
    cache={ds:assets.load(ds,next(v for v in parents.values() if v['dataset']==ds)) for ds in ('IMDB','PubMed')}
    rng=np.random.default_rng(202611131601); x=rng.normal(size=(7,45)); dx=increment(x)
    inversion=float(np.max(abs(flatten(np.cumsum(temporal(dx),axis=1))-x)));assert inversion<1e-12
    const=np.ones((1,45));constant_ratio=float(np.sum(increment(const)**2)/np.sum(const**2));assert abs(constant_ratio-1/9)<1e-12
    pulse=np.zeros((1,9,5));pulse[:,4,:]=1;pulse=flatten(pulse)
    pulse_ratio=float(np.sum(increment(pulse)**2)/np.sum(pulse**2));assert pulse_ratio==2
    manifest=assets.read(CAL/'panel_manifest.json'); z=np.load(assets.bind(CAL/'panel_bits.npz'))
    developer=[];precache={}
    for i,r in enumerate(manifest):
        if r['split']!='development':continue
        b=z[str(i)];assert assets.hashlib.sha256(b.tobytes()).hexdigest()==r['state_hash']
        ms,h,d=matrices(cache[r['dataset']],b);precache[i]=(ms,h,d)
        v,_,orders=solved(ms,1,'original'); inc=0
        for (_,ix,gd,gh,gi),(_,jj) in zip(ms,orders):inc+=float(gi[np.arange(len(ix)),jj].mean())/len(ms)
        assert v['heat']>1e-12
        developer.append(dict(dataset=r['dataset'],route=r['route'],index=r['index'],old_heat=v['heat'],increment_heat_at_old_match=inc,ratio=inc/v['heat']))
    coefficients={ds:float(np.median([r['ratio'] for r in developer if r['dataset']==ds])) for ds in cache}
    assert all(np.isfinite(c) and c>1e-12 for c in coefficients.values())
    dump('control_frozen.json',dict(coefficients=coefficients,development_states=len(developer),rule='median paired heat-energy ratio at original match',no_search=True))
    table('development_control.csv',developer)
    rows=[];types=[];times=[];nodegroups=[]
    def process(ident,b,expected=None,prepared=None):
        ds=ident['dataset'];sc=cache[ds];ms,h,d=prepared if prepared is not None else matrices(sc,b)
        vals={};order={}
        for mode in ('original','increment','amplitude_control'):
            v,pp,orders=solved(ms,coefficients[ds],mode);vals[mode]=v;order[mode]=orders
            types.extend({**ident,'mode':mode,**p} for p in pp)
        if expected is not None:assert abs(vals['original']['score']-expected)<1e-8
        row={**ident,'edges':int(b.sum()),'isolates':int((d.sum(1)==0).sum()),'max_degree':float(d.sum(1).max()),
             'degree_preserved':bool(np.array_equal(d,sc.rd))}
        for mode,v in vals.items():row.update({mode+'_'+k:q for k,q in v.items()})
        rows.append(row)
        if ident['panel']=='evaluation':
            # Paired temporal contributions at each metric's own matching.
            for mode in ('original','increment','amplitude_control'):
                for (typ,ix,_,_,_),(_,jj) in zip(ms,order[mode]):
                    hb=sc.scales[typ][2]
                    a=h[ix];ref=sc.rh[ix][jj]
                    delta=temporal(increment(a)-increment(ref)) if mode=='increment' else temporal(a-ref)
                    squared=delta**2/hb*(coefficients[ds] if mode=='amplitude_control' else 1)
                    contribution=squared.sum(2)/len(ix)/len(ms)
                    for k,t in enumerate(assets.heatmod.TIMES):
                        times.append({**ident,'mode':mode,'type':typ,'scale_index':k,'heat_time':float(t),
                                      'heat_contribution':float(contribution[:,k].sum()),
                                      'isolate_contribution':float(contribution[d[ix].sum(1)==0,k].sum())})
                    degree=d[ix].sum(1);cut=np.quantile(sc.rd[ix].sum(1),.9)
                    masks={'isolate':degree==0,'low_nonisolated':(degree>0)&(degree<=1),
                           'hub':(degree>1)&(degree>=cut),'middle':(degree>1)&(degree<cut)}
                    for group,mask in masks.items():
                        nodegroups.append({**ident,'mode':mode,'type':typ,'group':group,'support':int(mask.sum()),
                                           'heat_contribution':float(contribution[mask].sum())})
        return row
    oldrows=list(csv.DictReader(assets.bind(CAL/'original.csv').open()))
    lookup={(r['dataset'],r['route'],int(r['index'])):float(r['cost']) for r in oldrows}
    for i,r in enumerate(manifest):
        b=z[str(i)];assert assets.hashlib.sha256(b.tobytes()).hexdigest()==r['state_hash']
        process(dict(dataset=r['dataset'],panel='old_'+r['split'],route=r['route'],state=str(i),level=2),b,
                lookup[(r['dataset'],r['route'],r['index'])],precache.get(i))
    gates={r['state']:r['score'] for r in assets.read(ROOT/'results/joint_transport_score_gate_20261113/scores.json')}
    z=np.load(assets.bind(ROOT/'results/unified_organization_audit_20261113/cpu_perturbation_bits.npz'))
    for key in z.files:
        ds,route,seed,level=key.split('_');b=np.unpackbits(z[key])[:len(cache[ds].labels)]
        row=process(dict(dataset=ds,panel='legacy_perturbation',route=route,state=key,level=int(level)),b,gates[key])
        if route=='swap':assert row['degree_preserved']
    m=assets.read(ROB/'panel_manifest.json');z=np.load(assets.bind(ROB/'panel.npz'))
    for i,r in enumerate(m):
        b=z[str(i)]
        if 'state_hash' in r:assert assets.hashlib.sha256(b.tobytes()).hexdigest()==r['state_hash']
        process(dict(dataset=r['dataset'],panel='legacy_'+r['kind'],route=r['route'],state=str(i),
                     perturb_seed=r['index'],level=r.get('level',2)),b)
    for ep in sorted((SRC/'evaluation').glob('*.json')):
        ev=assets.read(ep);ds,seed,*rest=ep.stem.split('_');gp=assets.bind(Path(ev['graph_path']))
        assert assets.sha(gp)==ev['graph_sha256']
        st=ev['raw']['statistics'];cmp=ev['raw']['comparison']
        process(dict(dataset=ds,panel='evaluation',route='_'.join(rest[:-1]),state=ep.stem,train_seed=int(seed),sample_seed=int(rest[-1]),
            triangles=st['triangles'],triangle_error=abs(st['triangles']-ev['reference']['triangles']),
            degree_W1=cmp['main_incident_W1'],joint_TV=cmp['main_joint_degree_TV'],LCC=st['lcc'],
            MP3_error=abs(st['mp'][2]['total']-ev['reference']['mp'][2]['total'])),
            assets.bits(cache[ds],np.load(gp)),ev['unified_transport_score'][0])
    assert len(rows)==468
    checks=[]
    for ds,sc in cache.items():
        ms,_,_=matrices(sc,sc.labels)
        for mode in ('original','increment','amplitude_control'):
            v,_,_=solved(ms,coefficients[ds],mode);assert v['score']<1e-12
        perm=np.arange(sc.n);prng=np.random.default_rng(2026111309)
        for typ in sc.scales:
            ix=np.flatnonzero(sc.types==typ);perm[ix]=prng.permutation(ix)
        b=np.zeros_like(sc.labels)
        for u,v in sc.q[:,sc.labels.astype(bool)].T:b[sc.lookup[tuple(sorted((int(perm[u]),int(perm[v]))))]]=1
        ms,_,_=matrices(sc,b)
        v,_,_=solved(ms,coefficients[ds],'increment');assert v['score']<1e-10
        checks.append(dict(dataset=ds,permutation_cost=v['score']))
    for name,rr in [('states.csv',rows),('types.csv',types),('time_contributions.csv',times),('node_groups.csv',nodegroups)]:table(name,rr)
    assert 'torch' not in sys.modules
    assert all(assets.sha(p)==h for p,h in assets.hashes.items())
    dump('checks.json',dict(inversion_max_error=inversion,constant_offset_energy_ratio=constant_ratio,one_scale_pulse_energy_ratio=pulse_ratio,
                           reference_checks=checks,inequality='increment energy <=4*original energy checked for all node-pair matrices',source_unchanged=True))
    dump('sources.json',assets.hashes)
    dump('receipt.json',dict(status='complete',seconds=time.monotonic()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        old_states=468,new_scientific_graphs=0,invariance_permutations=2,models=0,GPU_hours=0,formal_modified=False))
    print(json.dumps(dict(control=coefficients,receipt=json.loads((OUT/'receipt.json').read_text())),indent=2))

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        if OUT.exists():dump('failure.json',dict(traceback=traceback.format_exc()))
        raise
