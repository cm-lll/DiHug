"""Reuse saved draws: distinguish batching from changing the sampling distribution.
No Torch imports, model calls, new graph draws, or parameter updates.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,subprocess,time,resource
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'results/joint_feedback_audit_20261111'
TERM=ROOT/'results/terminal_structure_real_20261109T233600'
OUT=ROOT/'results/one_step_vs_terminal_plan_20261111'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False))
def csvsave(name,rows):
    save(name+'.json',rows)
    with (OUT/(name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def estimate(p,bits,cost,baseline):
    """baseline is external to these draws; loss gradient, no sample selection."""
    return np.mean((cost-baseline)[:,None]*(bits-p),axis=0)

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((SRC/'protocol.json').read_text())
    save('protocol.json',dict(scope='CPU reuse only; no fresh samples or models',
        source=str(SRC),panels=12,baseline_draws=16,evaluation_draws=112,K=[1,4,16],
        objective='unchanged C_joint+C_count, coefficients1 diagnostic only',
        pairing='first16 saved draws provide external baseline; same remaining112 draws grouped three ways',
        limitation='All p frozen; cannot compare online learning or broader noise-input coverage; batch means share data, not independent experiments',
        decision='Low split cosine is a warning, not a training-effect veto; no cherry picking or winner-only rule'))
    rows=[];selections=[];bindings={};maxerr=0.
    for i,item in enumerate(protocol['panel']):
        path=SRC/f'panel_{i:02d}.npz';bindings[str(path)]=sha(path)
        z=np.load(path);original=np.load(item['path']);p=z['probability'];f=original['families']
        target=np.bincount(f,weights=original['labels'],minlength=10)
        batches=[]
        for rep in range(8):
            rng=np.random.default_rng(np.random.SeedSequence([protocol['random_root'],item['train_seed'],item['t'],rep]))
            batches.append((rng.random((16,len(p)))<p).astype(float))
        b=np.concatenate(batches);nc=np.stack([b[:,f==r].sum(1) for r in range(10)],axis=1)
        cc=np.mean((nc-target)**2/np.maximum(1,target)**2,axis=1)
        cost=z['joint_costs']+cc;base=float(cost[:16].mean())
        x=b[16:];c=cost[16:];pooled=estimate(p,x,c,base)
        for k in [1,4,16]:
            g=np.stack([estimate(p,x[j:j+k],c[j:j+k],base) for j in range(0,112,k)])
            err=float(np.max(abs(g.mean(0)-pooled)));maxerr=max(maxerr,err)
            assert err<1e-12
            rows.append(dict(dataset=item['dataset'],seed=item['train_seed'],t=item['t'],K=k,
                batches=len(g),hard_samples=112,baseline=base,mean_cost=float(c.mean()),
                pooled_gradient_norm=float(np.linalg.norm(pooled)),pooled_max_error=err,
                empirical_batch_dispersion=float(np.sqrt(np.mean(np.sum((g-pooled)**2,axis=1)))),
                note='same112 draws, fixed logits; not online optimizer or significance comparison'))
        groups=c.reshape(7,16)
        selections.append(dict(dataset=item['dataset'],seed=item['train_seed'],t=item['t'],
            ordinary_mean=float(c.mean()),best_of16_mean=float(groups.min(1).mean()),
            best_of16_note='selection statistic only; cannot report as ordinary generated quality; no elite learning performed'))
    csvsave('batching',rows);csvsave('selection_statistics',selections)
    assets=[]
    for item in protocol['panel'][::3]:
        path=Path(item['checkpoint']);assets.append(dict(kind='A_u2000_raw_parent',dataset=item['dataset'],seed=item['train_seed'],path=str(path),exists=path.exists(),sha256=sha(path) if path.exists() else None))
    termp=json.loads((TERM/'protocol.json').read_text());chains=[]
    for ds,m in termp['models'].items():
        readout=Path(m['readout']);actual=sha(readout) if readout.exists() else None
        assets.append(dict(kind='A_u3000_EMA_terminal_parent',dataset=ds,seed=m['train_seed'],path=str(readout),exists=readout.exists(),sha256=actual))
        assert actual==termp['bindings'][str(readout)]
        for seed in termp['sample_seeds']:
            folder=TERM/ds/str(seed);s=json.loads((folder/'summary.json').read_text());g=folder/'graph.npz'
            assert sha(g)==s['graph_sha256']
            traces=sorted(folder.glob('step_*.npz'))
            assert len(traces)==8
            for tr in traces:
                q=np.load(tr);assert all(k in q for k in ['xt','xprev','p0','q','uniform','coverage'])
                bindings[str(tr)]=sha(tr)
            chains.append(dict(dataset=ds,seed=m['train_seed'],sample_seed=seed,graph=str(g),sha256=sha(g),seconds=s['seconds'],states=len(traces)))
    save('assets.json',dict(parents=assets,terminal_chains=chains,limitations=['u2000 raw caches and u3000 EMA chains not matched','saved old transition gradients are not clean-head eager gradients','old seven terminal costs differ from current joint+count objective']))
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    save('resources.json',dict(host=subprocess.check_output(['hostname'],text=True).strip(),gpu_query_returncode=gpu.returncode,gpu_query=gpu.stdout+gpu.stderr,disk=subprocess.check_output(['df','-h',str(ROOT)],text=True),GPU_reserved=False))
    # Historical complete-chain bounds, not fabricated current throughput.
    dsmean={ds:float(np.mean([c['seconds'] for c in chains if c['dataset']==ds])) for ds in ['IMDB','PubMed']}
    save('budget.json',dict(historical_chain_seconds=dsmean,
        old_eight_chain_GPU_hours=json.loads((TERM/'completion.json').read_text())['occupied_gpu_hours'],
        minimal_future_gpu_calibration=dict(wall_minutes=15,GPU_hours=.25,max_gpus=1,
            work='bounded same-state clean-head scoring/backward and timing, no training or new full chain',executed=False),
        full_chain_cost_warning='K4 per update historically about32min IMDB or42.5min PubMed; K4 x10 updates x4models approximately24.8GPUh just chain collection, before evaluation and optimizer overhead',
        next_comparison='Reuse8saved chains/64Xt at u3000EMA for same-state one-step versus eager diagnostic; requires new model calls, not yet performed',
        final_training_budget='not frozen; cannot certify before new-path timing and GPU availability',
        online_updates=0))
    for p,h in bindings.items():assert sha(p)==h
    save('bindings.json',bindings)
    save('receipt.json',dict(status='complete',seconds=time.perf_counter()-start,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        regrouped_existing_draws=1536,new_draws=0,model_calls=0,training_updates=0,reverse_chains=0,GPU_hours=0,
        equal_samples_batching_max_error=maxerr,source_unchanged=True))
if __name__=='__main__':main()
