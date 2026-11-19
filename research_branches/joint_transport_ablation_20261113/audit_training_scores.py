"""CPU read-only audit of saved terminal rewards and hard graphs (128 chains)."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,re,time,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'results/joint_transport_ablation_20261113T011928'
OUT=ROOT/'results/joint_transport_decomposition_20261113'
def read(p):return json.loads(p.read_text())
def table(name,rows):
    with (OUT/(name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    start=time.time();p=read(SRC/'protocol.json');refs=read(SRC/'reference_metrics.json');rows=[];batches=[];updates=[];hashes={}
    for job in p['jobs']:
        folder=SRC/job['id']/(job['dataset']+'_T_eager');ds=job['dataset'];seed=job['train_seed'];arm=job['architecture'];id=dict(dataset=ds,train_seed=seed,arm=arm)
        up=folder/'updates.jsonl';hashes[str(up)]=hashlib.sha256(up.read_bytes()).hexdigest()
        for line in up.read_text().splitlines():
            u=json.loads(line);updates.append(dict(**id,update=u['update'],gradient_norm_before_clip=u['gradient_norm_before_clip'],clip_scale=u['clip_scale']))
        for u in range(1,5):
            bp=folder/f'round{u}_batch.json';batch=read(bp);hashes[str(bp)]=hashlib.sha256(bp.read_bytes()).hexdigest();items=[]
            for sp in sorted(folder.glob(f'round{u}_seed*_score.json')):
                ss=int(re.search(r'_seed(\d+)',sp.name)[1]);sc=read(sp);gp=folder/f'train_round{u}_seed{ss}'/'graph.npz'
                for fp in (gp,sp):hashes[str(fp)]=hashlib.sha256(fp.read_bytes()).hexdigest()
                z=np.load(gp);a=np.zeros((len(z['types']),len(z['types'])));e=z['edges'];e=e.T if e.shape[0]==2 else e;a[e[:,0],e[:,1]]=1;a+=a.T
                assert np.max(a)<=1 and not np.diag(a).any();tri=float(np.trace(a@a@a)/6);deg=float(np.mean([x['degree_at_joint_match'] for x in sc['type_parts']]));heat=float(np.mean([x['heat_at_joint_match'] for x in sc['type_parts']]))
                items.append(dict(**id,update=u,chain_seed=ss,cost=sc['cost'],degree_cost=deg,heat_cost=heat,edges=len(e),isolates=int((a.sum(1)==0).sum()),triangles=tri,triangle_error=abs(tri-refs[ds]['triangles'])))
            assert len(items)==4 and np.allclose(sorted(x['cost'] for x in items),sorted(batch['costs']),atol=1e-10,rtol=0)
            costs=np.array([x['cost'] for x in items]);dc=np.array([x['degree_cost'] for x in items]);hc=np.array([x['heat_cost'] for x in items]);ac=(4*costs-costs.sum())/3;ad=(4*dc-dc.sum())/3;ah=(4*hc-hc.sum())/3
            assert np.max(abs(ac-ad-ah))<1e-10
            for i,x in enumerate(items):x.update(cost_advantage=float(ac[i]),degree_advantage=float(ad[i]),heat_advantage=float(ah[i]))
            best=min(items,key=lambda x:x['cost']);worst=max(items,key=lambda x:x['cost'])
            batches.append(dict(**id,update=u,heat_advantage_L2=float(np.linalg.norm(ah)),degree_advantage_L2=float(np.linalg.norm(ad)),heat_degree_ratio=float(np.linalg.norm(ah)/np.linalg.norm(ad)),cost_isolate_spearman=float(spearmanr(costs,[x['isolates'] for x in items]).statistic) if len(set(x['isolates'] for x in items))>1 else None,best_cost_triangle_error=best['triangle_error'],worst_cost_triangle_error=worst['triangle_error'],best_cost_isolates=best['isolates'],worst_cost_isolates=worst['isolates'],minimum_cost=best['cost'],maximum_cost=worst['cost']))
            rows+=items
    table('training_graphs',rows);table('training_batches',batches);table('training_updates',updates)
    assert len(rows)==128 and len(batches)==32 and len(updates)==32
    (OUT/'training_source_hashes.json').write_text(json.dumps(hashes,indent=2))
    (OUT/'training_receipt.json').write_text(json.dumps(dict(status='complete',graphs=len(rows),batches=len(batches),seconds=time.time()-start,GPU_hours=0,model_calls=0,source_hashes_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())),indent=2))
    print((OUT/'training_receipt.json').read_text())
if __name__=='__main__':main()
