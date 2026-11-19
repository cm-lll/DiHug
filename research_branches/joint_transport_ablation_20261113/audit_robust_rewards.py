"""Read existing training graphs; scalar rewards only, no model/gradient calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,csv,re,resource
from pathlib import Path
import numpy as np
import sensitivity_cpu as s
from robust_heat_v1 import evaluate
from check_robust_heat import bits
OUT=s.ROOT/'results/robust_heat_transport_20261113'
def main():
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));start=time.time();protocol=json.loads((s.SRC/'protocol.json').read_text());deadline=json.loads((OUT/'protocol.json').read_text())['deadline_epoch'];rows=[];batches=[];cache={};hashes={}
    for job in protocol['jobs']:
        ds=job['dataset'];folder=s.SRC/job['id']/(ds+'_T_eager')
        if ds not in cache:cache[ds]=s.load(ds,next(v for v in protocol['parents'].values() if v['dataset']==ds))
        sc=cache[ds]
        for u in range(1,5):
            group=[]
            for sp in sorted(folder.glob(f'round{u}_seed*_score.json')):
                if time.time()>deadline-30:raise TimeoutError('deadline')
                ss=int(re.search(r'_seed(\d+)',sp.name)[1]);gp=folder/f'train_round{u}_seed{ss}'/'graph.npz';hashes[str(sp)]=s.digest(sp);hashes[str(gp)]=s.digest(gp);v,_=evaluate(sc,bits(sc,np.load(gp)));old=json.loads(sp.read_text());assert abs(v['original']['total']-old['cost'])<1e-8
                group.append(dict(dataset=ds,train_seed=job['train_seed'],arm=job['architecture'],update=u,chain_seed=ss,original=v['original']['total'],robust=v['robust']['total'],degree=v['robust']['degree'],heat=v['robust']['heat']))
            assert len(group)==4
            for k in ('original','robust','degree','heat'):
                x=np.array([r[k] for r in group]);a=(4*x-x.sum())/3
                for i,r in enumerate(group):r[k+'_advantage']=float(a[i])
            assert np.max(abs(np.array([r['robust_advantage']-r['degree_advantage']-r['heat_advantage'] for r in group])))<1e-9
            batches.append(dict(dataset=ds,train_seed=job['train_seed'],arm=job['architecture'],update=u,heat_degree_advantage_ratio=float(np.linalg.norm([x['heat_advantage'] for x in group])/np.linalg.norm([x['degree_advantage'] for x in group])),advantage_sign_flips=sum(np.sign(x['original_advantage'])!=np.sign(x['robust_advantage']) for x in group),preferred_chain_changed=min(group,key=lambda x:x['original'])['chain_seed']!=min(group,key=lambda x:x['robust'])['chain_seed']))
            rows+=group
    for name,rr in [('training_rescore.csv',rows),('training_batch_rescore.csv',batches)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,list(rr[0]));w.writeheader();w.writerows(rr)
    result={ds:dict(batches=16,heat_degree_ratio_median=float(np.median([r['heat_degree_advantage_ratio'] for r in batches if r['dataset']==ds])),heat_larger=sum(r['heat_degree_advantage_ratio']>1 for r in batches if r['dataset']==ds),preferred_chain_changed=sum(r['preferred_chain_changed'] for r in batches if r['dataset']==ds),advantage_sign_flips=sum(r['advantage_sign_flips'] for r in batches if r['dataset']==ds)) for ds in cache}
    # Counts may be numpy integer scalars; canonicalise only serialization.
    result.update(CPU_seconds=time.time()-start,model_calls=0,GPU_hours=0,source_unchanged=all(s.digest(k)==v for k,v in hashes.items()));encoded=json.dumps(result,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x);(OUT/'training_rescore_receipt.json').write_text(encoded);(OUT/'training_source_hashes.json').write_text(json.dumps(hashes,indent=2));print(encoded)
if __name__=='__main__':main()
