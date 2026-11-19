"""CPU audit: distinct RNG draws need not yield distinct graph states."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np,json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parents[2]/'results/analytic_joint_short_20261113'
rows=[]
for folder in sorted(OUT.glob('*_A_transport')):
    z=np.load(folder/'inputs.npz');manifest=json.loads((folder/'inputs.json').read_text());keys=[]
    for r in manifest:
        edges=z[r['tag']+'__edges'];fam=z[r['tag']+'__edge_families']
        # Ignore query ordering when comparing the actual noisy graph.
        a=np.sort(edges,axis=0)
        records=sorted(set((int(u),int(v),int(f)) for (u,v),f in zip(a.T,fam)))
        digest=hashlib.sha256(json.dumps([r['t'],records]).encode()).hexdigest()
        keys.append(dict(**r,state_digest=digest))
    for t in (1,75,250,550,800,1000):
        train={r['state_digest'] for r in keys if r['split']=='train' and r['t']==t}
        validation=[r['state_digest'] for r in keys if r['split']=='validation' and r['t']==t]
        rows.append(dict(job=folder.name,t=t,training_draws=16,distinct_training_states=len(train),validation_draws=2,
            distinct_validation_states=len(set(validation)),validation_draws_matching_a_training_state=sum(k in train for k in validation)))
(OUT/'input_state_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
