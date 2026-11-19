"""Read-only CPU decomposition of the frozen 64-graph panel, no model imports.

All terms use the SAME joint Hungarian assignment as production. Node groups
use current graph total degree: low <=1, hub >= reference type p90 (and >1).
These are descriptive groups, not matched identities or new reward terms.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,time,hashlib,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from transport import TransportScore,heatmod

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/joint_transport_ablation_20261113T011928'
OUT=ROOT/'results/joint_transport_decomposition_20261113'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def table(name,rows):
    with (OUT/(name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def edges(z):
    e=z['edges'];return e.T if e.shape[0]==2 else e
def main():
    start=time.time();OUT.mkdir(exist_ok=False)
    protocol=read(SOURCE/'protocol.json');sources=[SOURCE/'protocol.json',Path(__file__),Path(__file__).with_name('transport.py'),Path(heatmod.__file__)]
    assets=sorted((SOURCE/'evaluation').glob('*.json'));assert len(assets)==64
    dump('protocol.json',dict(source=str(SOURCE),panel=[str(p) for p in assets],scope='CPU saved graphs only; no perturbations, model calls, fitting or GPU',
        groups='current total degree <=1 low; >1 and >=reference type p90 hub; remaining middle',
        assignment='joint degree+heat assignment unchanged; additive decomposition at same assignment',
        heat_groups='9 return times then 9 times x 4 destination types; direct unnormalised squared errors also retained',
        interpretation='descriptive posthoc decomposition, not causal training attribution',max_CPU_seconds=600))
    cache={};nodes=[];features=[];types=[];graphs=[];scales=[];maxerr=0.;hashes={}
    for ep in assets:
        assert time.time()-start<600
        meta=ep.stem.split('_');ds=meta[0];seed=int(meta[1]);ss=int(meta[-1]);arm='_'.join(meta[2:-1]);ident=dict(dataset=ds,train_seed=seed,arm=arm,sample_seed=ss)
        ev=read(ep);gp=Path(ev['graph_path']);assert sha(gp)==ev['graph_sha256'];sources.extend([ep,gp])
        if ds not in cache:
            parent=next(v for v in protocol['parents'].values() if v['dataset']==ds);rp=Path(parent['data']);sources.append(rp);z=np.load(rp);t=z['types'];schema=z['schema'];names=parent['type_names'];n=len(t)
            lookup={tuple(sorted(map(int,r))):i for i,r in enumerate(schema)}
            qs=[];fs=[]
            for u in range(n):
                for v in range(u+1,n):
                    key=tuple(sorted((int(t[u]),int(t[v]))))
                    if key in lookup:qs.append((u,v));fs.append(lookup[key])
            q=np.array(qs).T;fam=np.array(fs);slots={tuple(e):i for i,e in enumerate(q.T)}
            def bits(e):
                b=np.zeros(q.shape[1],np.uint8)
                for u,v in e:b[slots[tuple(sorted((int(u),int(v))))]]=1
                return b
            labels=bits(edges(z));prior=np.array([labels[fam==i].mean() for i in range(len(schema))]);case=dict(space=SimpleNamespace(types=t),n_families=len(schema),prior=prior)
            score=TransportScore(ds,case,q,fam,labels);cache[ds]=(score,slots,names,schema)
            for typ,(scale,db,hb) in score.scales.items():
                sel=t==typ;scales.append(dict(dataset=ds,type=names[typ],nodes=int(sel.sum()),degree_bandwidth=db,heat_bandwidth=hb,heat_multiplier=1/hb,reference_low=int((score.rd[sel].sum(1)<=1).sum()),hub_cutoff=float(np.quantile(score.rd[sel].sum(1),.9)),degree_scale=scale.tolist()))
        score,slots,names,schema=cache[ds];z=np.load(gp);assert np.array_equal(z['types'],score.types);b=np.zeros(score.q.shape[1],np.uint8)
        for u,v in edges(z):b[slots[tuple(sorted((int(u),int(v))))]]=1
        h,d=score.features(b);r=score.r;nt=len(names);tot=0.;dc_all=hc_all=0.
        a=np.zeros((score.n,score.n));sel=b.astype(bool);a[score.q[0,sel],score.q[1,sel]]=1;a+=a.T;tri=np.diag(a@a@a)/2
        assert abs(tri.sum()/3-ev['raw']['statistics']['triangles'])<1e-8
        for typ in score.scales:
            idx=np.flatnonzero(score.types==typ);scale,db,hb=score.scales[typ];x=np.concatenate([d[idx]/scale/np.sqrt(db),h[idx]/np.sqrt(hb)],1);y=score.reference[typ]
            ii,jj=linear_sum_assignment(cdist(x,y,'sqeuclidean'));assert np.array_equal(ii,np.arange(len(idx)))
            delta=(x-y[jj])**2;dc=delta[:,:r].sum(1);hc=delta[:,r:].sum(1);total=dc+hc;tot+=total.mean()/nt;dc_all+=dc.mean()/nt;hc_all+=hc.mean()/nt
            rd=score.rd[idx][jj];rh=score.rh[idx][jj];cut=float(np.quantile(score.rd[idx].sum(1),.9));degs=d[idx].sum(1)
            for j,node in enumerate(idx):
                group='low' if degs[j]<=1 else 'hub' if degs[j]>=cut else 'middle'
                nodes.append(dict(**ident,type=names[typ],node=int(node),matched_reference_node=int(idx[jj[j]]),group=group,degree=float(degs[j]),reference_degree=float(rd[j].sum()),triangles=float(tri[node]),degree_cost=float(dc[j]),heat_cost=float(hc[j]),total_cost=float(total[j]),contribution=float(total[j]/len(idx)/nt),isolated=bool(degs[j]==0)))
            types.append(dict(**ident,type=names[typ],nodes=len(idx),degree_cost=float(dc.mean()),heat_cost=float(hc.mean()),total_cost=float(total.mean()),return_cost=float(delta[:,r:r+9].sum(1).mean()),arrival_cost=float(delta[:,r+9:].sum(1).mean()),degree_unscaled_MSE=float(((d[idx]-rd)**2).sum(1).mean()),heat_unscaled_MSE=float(((h[idx]-rh)**2).sum(1).mean()),low_count=int((degs<=1).sum()),isolates=int((degs==0).sum()),max_degree=float(degs.max()),triangle_participations=float(tri[idx].sum()),contribution=float(total.mean()/nt)))
            for j in range(delta.shape[1]):
                if j<r:
                    block='degree';label='-'.join(names[int(t)] for t in schema[j]);tm=None;raw=float(((d[idx,j]-rd[:,j])**2).mean())
                else:
                    k=j-r;block='return' if k<9 else 'arrival';ti=k if k<9 else (k-9)//nt;label='self' if k<9 else names[(k-9)%nt];tm=float(heatmod.TIMES[ti]);raw=float(((h[idx,k]-rh[:,k])**2).mean())
                features.append(dict(**ident,type=names[typ],block=block,channel=label,heat_time=tm,cost=float(delta[:,j].mean()),contribution=float(delta[:,j].mean()/nt),unscaled_feature_MSE=raw))
        err=abs(tot-ev['unified_transport_score'][0]);maxerr=max(maxerr,err);assert err<1e-8,(ep,err)
        st=ev['raw']['statistics'];cmp=ev['raw']['comparison']
        graphs.append(dict(**ident,score=tot,degree_cost=dc_all,heat_cost=hc_all,score_error=err,edges=st['edges'],triangles=st['triangles'],triangle_error=abs(st['triangles']-ev['reference']['triangles']),MP3_error=abs(st['mp'][2]['total']-ev['reference']['mp'][2]['total']),degree_W1=cmp['main_incident_W1'],joint_TV=cmp['main_joint_degree_TV'],isolates=st['isolates'],LCC=st['lcc']))
    for p in set(sources):hashes[str(p)]=sha(p)
    table('graphs',graphs);table('types',types);table('features',features);table('nodes',nodes);table('reference_scaling',scales)
    # All per-sample paired deltas kept; no selecting favourable graphs.
    paired=[]
    for x in graphs:
        if not x['arm'].endswith('_transport'):continue
        y=next(z for z in graphs if z['dataset']==x['dataset'] and z['train_seed']==x['train_seed'] and z['sample_seed']==x['sample_seed'] and z['arm']==x['arm'].replace('_transport',''))
        paired.append({**{k:x[k] for k in ident},**{k:x[k]-y[k] for k in x if k not in ident}})
    table('paired_deltas',paired);dump('source_hashes.json',hashes)
    dump('receipt.json',dict(status='complete',graphs=len(graphs),node_records=len(nodes),feature_records=len(features),max_original_score_difference=maxerr,seconds=time.time()-start,GPU_hours=0,model_calls=0,new_graphs=0,source_hashes_unchanged=all(sha(Path(p))==h for p,h in hashes.items())))
    print(json.dumps(read(OUT/'receipt.json')))
if __name__=='__main__':main()
