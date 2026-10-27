"""Reexecute the frozen attempt02 exactly in a fresh output dir, verify canonical arrays and full proposal decisions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import tempfile,json,numpy as np,contextlib,io,hashlib
from pathlib import Path
import refine
HOME=refine.ROOT;dst=Path(tempfile.mkdtemp(prefix='refined192_replay_',dir='/tmp'));(dst/'samples').mkdir();refine.ROOT=dst
with (dst/'replay.log').open('w') as f,contextlib.redirect_stdout(f):refine.run()
checks={}
for ds in ['IMDB','PubMed']:
 a=np.load(dst/'samples'/ds/'graph.npz');b=np.load(HOME/'samples'/ds/'graph.npz');g=np.load(HOME/'inputs/reference'/f'{ds}.npz');ids=np.sort(a['original_ids']);inv=np.full(len(g['types']),-1,int);inv[ids]=np.arange(192);mask=(inv[g['edges']]>=0).all(1)
 vals={'types':g['types'][ids],'edges':inv[g['edges'][mask]],'relations':g['relations'][mask],'schema':g['schema'],'original_ids':ids}
 checks[ds]={'canonical_arrays_equal':all(np.array_equal(v,b[k]) for k,v in vals.items()),'full_proposal_trace_equal':json.loads((dst/'samples'/ds/'optimization_trace.json').read_text())==json.loads((HOME/'samples'/ds/'optimization_trace.json').read_text())}
result={'scope':'Fresh CPU optimization replay, not just chosen-node reconstruction','replay_directory':str(dst),'datasets':checks,'all_passed':all(all(v.values()) for v in checks.values())}
(HOME/'portable_replay_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['all_passed']
