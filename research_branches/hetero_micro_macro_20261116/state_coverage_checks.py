"""Independent slot/label/source/terminal count checks; no model imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys, importlib.util, json, hashlib, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'results/hetero_micro_macro_effect_run_20261116T083122'
OUT=ROOT/'results/state_coverage_audit_20261116_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.time();p=json.loads((SRC/'protocol.json').read_text())
pkg=Path(p['parents'][0]['data']).parent.parent
spec=importlib.util.spec_from_file_location('coverage_original_slots',pkg/'hsgd/slots.py')
slots_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=slots_module;spec.loader.exec_module(slots_module)
for path,h in p['runtime_source_bindings'].items():assert sha(path)==h,path
for parent in p['parents']:
    ds=parent['dataset'];assert sha(parent['data'])==parent['config']['datasets'][ds]['sha256']
    z=np.load(parent['data']);ty=z['types'];e=z['edges'];e=e if e.shape[0]==2 else e.T
    S=slots_module.TypedSlots(ty,tuple(map(tuple,z['schema'])))
    clean={tuple(sorted(x)) for x in e.T}
    q=np.concatenate([S.decode(f,np.arange(int(c))) for f,c in enumerate(S.counts)],axis=1)
    # Audit's independent list construction must agree with production storage.
    pairs=[]
    import itertools
    for a,b in z['schema']:
        na=np.flatnonzero(ty==a);nb=np.flatnonzero(ty==b)
        pairs.extend(list(itertools.combinations(na,2)) if a==b else [tuple(sorted((u,v))) for u in na for v in nb])
    np.testing.assert_array_equal(q,np.array(pairs).T)
    a=np.load(SRC/parent['id']/'CE/inputs.npz')
    for split,count in [('train',128),('validation',12)]:
        for i in range(count):
            k=f'{split}_{i}__';qe=a[k+'query'];fam,loc=S.encode(qe)
            np.testing.assert_array_equal(fam,a[k+'f'])
            np.testing.assert_array_equal(S.offsets[fam]+loc,a[k+'ids'])
            np.testing.assert_array_equal(a[k+'labels'],[tuple(sorted(x)) in clean for x in qe.T])
old=json.loads((SRC/'per_graph.json').read_text());rows=json.loads((OUT/'states.json').read_text())
for r in rows:
    if r['kind']!='terminal':continue
    x=next(x for x in old if x['view']=='whole' and x['graph']==r['path'])
    for k in ('edges','triangles','maximum_degree','isolates'):assert x[k]==r[k],(r['path'],k)
receipt=dict(status='passed',runtime_source_files=len(p['runtime_source_bindings']),data_hashes=4,
    production_slot_order_verified=4,input_query_label_bindings=560,terminal_existing_metric_agreements=48,
    seconds=time.time()-start,model_calls=0,GPU_hours=0,
    sources={str(ROOT/'research_branches/hetero_micro_macro_20261116'/f):sha(ROOT/'research_branches/hetero_micro_macro_20261116'/f) for f in ('state_coverage_audit.py','state_coverage_checks.py')})
(OUT/'independent_checks.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
