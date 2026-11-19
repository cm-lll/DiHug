"""CPU-only asset binding and frozen 4/8/16-trajectory reliability plan.

No denoiser import, model calls, optimizer updates or graph generation.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import csv,hashlib,itertools,json,time,resource
from pathlib import Path
import numpy as np
import heat_increment_cpu as inc

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/fullchain_reliability_plan_20261113'
SRC=ROOT/'results/joint_transport_ablation_20261113T011928'
OLD=ROOT/'results/paper_ablation_20261112T190000'
bindings={}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def bind(p,expected=None):
    p=Path(p).resolve();h=sha(p)
    if expected is not None:assert h==expected,str(p)
    bindings[str(p)]=h;return p

def read(p):return json.loads(bind(p).read_text())

def write(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))

def table(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def digest(x):
    # Same persisted state-hash contract as the existing update check; CPU only.
    import torch
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.dtype,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(repr(k).encode());add(v[k])
        elif isinstance(v,(list,tuple)):
            for a in v:add(a)
        else:h.update(repr(v).encode())
    add(x);return h.hexdigest()

def main():
    start=time.time();OUT.mkdir(exist_ok=False)
    bind(__file__)
    p=read(SRC/'protocol.json');op=read(OLD/'protocol.json')
    jobs=[j for j in p['jobs'] if j['architecture']=='A'];assert len(jobs)==4
    # Generate all new seed/time plans and group assignments before rescoring assets.
    frozen=[]
    for j in jobs:
        seed=j['train_seed'];rg=np.random.default_rng(np.random.SeedSequence([202611132601,seed]))
        known={r['seed'] for u in j['terminal_plan'].values() for r in u}
        oldj=next(a for a in op['jobs'] if a['parent']==j['parent'] and a['reward']=='generic')
        known.update(r['seed'] for u in oldj['terminal_plan'].values() for r in u)
        new=[]
        for k in range(10):
            ss=int(rg.integers(1,2**31-1))
            assert ss not in known;known.add(ss)
            new.append(dict(index=6+k,seed=ss,times=sorted(map(int,rg.choice(np.arange(1,1001),8,replace=False)))))
        split=np.random.default_rng(np.random.SeedSequence([202611132602,seed]))
        oldidx=list(map(int,split.permutation(6)));newidx=list(map(int,split.permutation(np.arange(6,16))))
        quartets=[]
        for no in [2,1,2,1]:
            group=oldidx[:no]+newidx[:4-no];oldidx=oldidx[no:];newidx=newidx[4-no:]
            quartets.append(list(map(int,split.permutation(group))))
        assert sorted(sum(quartets,[]))==list(range(16))
        frozen.append(dict(id=j['id'],dataset=j['dataset'],train_seed=seed,new_trajectories=new,
            four_groups=quartets,eight_groups=[quartets[0]+quartets[1],quartets[2]+quartets[3]],sixteen_group=list(range(16))))
    write('frozen_sampling_groups.json',dict(new_seed_root=202611132601,partition_root=202611132602,
        rule='Indices0..3 previous transport first batch;4..5 prior generic first batch;6..15 new. Balance reused counts2/1/2/1 per quartet; no quality selection.',models=frozen))
    # CPU algebra: each subset LOO estimator equals sample covariance of cost and v.
    c=np.array([.2,.8,1.1,2.,.1]);v=np.arange(15,dtype=float).reshape(5,3)**1.1
    def estimate(ii):
        x=c[ii];y=v[ii];a=(len(x)*x-x.sum())/(len(x)-1)
        return (a[:,None]*y).mean(0)
    full=estimate(list(range(5)))
    for k in (2,3,4):
        avg=np.mean([estimate(list(ii)) for ii in itertools.combinations(range(5),k)],axis=0)
        np.testing.assert_allclose(avg,full,rtol=1e-12,atol=1e-12)
    records=[];models=[];rejected_duplicates=[];cache={};contracts=[]
    import torch
    torch.set_num_threads(1)
    for j,frozen_j in zip(jobs,frozen):
        ds=j['dataset'];row=p['parents'][j['parent']];folder=SRC/j['id']/(ds+'_T_eager')
        statefile=bind(folder/'initial_complete.pt');state=torch.load(statefile,map_location='cpu',weights_only=False)
        readout=bind(row['readout'],p['bindings'][row['readout']]);parent=torch.load(readout,map_location='cpu',weights_only=False)
        assert state['updates']==0 and not state['optimizer']['state']
        assert parent['updates']==3000 and parent['train_seed']==j['train_seed'] and parent['readout']=='ema_0.99'
        mh=digest(state['model']);assert mh==digest(parent['model'])
        cfg=parent['binding']['config'];assert cfg==row['config'] and cfg['inputs_per_update']==1 and cfg['rounds']==1 and cfg['T']==1000
        bind(row['data'],p['bindings'][row['data']]);binding=read(folder/'binding.json');assert binding['model_hash']==mh
        oldj=next(a for a in op['jobs'] if a['parent']==j['parent'] and a['reward']=='generic')
        oldfolder=OLD/oldj['id']/(ds+'_T_eager');ob=read(oldfolder/'binding.json')
        assert ob['model_hash']==mh and ob['config']==binding['config'] and ob['parent_readout_sha256']==sha(readout)
        oldproto=read(OLD/oldj['id']/'protocol.json')
        # Shared production implementations must retain hashes across both runs.
        shared=[q for q in p['bindings'] if q in oldproto['bindings'] and q.endswith('.py')]
        assert shared
        for q in shared:
            assert p['bindings'][q]==oldproto['bindings'][q];bind(q,p['bindings'][q])
        if ds not in cache:cache[ds]=inc.assets.load(ds,row)
        sc=cache[ds];plans=[(folder,x) for x in j['terminal_plan']['1']]+[(oldfolder,x) for x in oldj['terminal_plan']['1']]
        assert len(plans)==6 and len({x['seed'] for _,x in plans})==6
        paths=[]
        for idx,(base,plan) in enumerate(plans):
            d=base/f"train_round1_seed{plan['seed']}";ident=read(d/'identity.json');su=read(d/'summary.json')
            assert ident['model_hash']==mh and ident['updates']==0 and ident['R']==1 and ident['T']==1000 and ident['sample_seed']==plan['seed'] and ident['train_seed']==j['train_seed']
            assert su['status']=='complete' and su['steps']==1000 and su['model_unchanged']
            gp=bind(d/'graph.npz',su['graph_sha256']);lines=[json.loads(x) for x in bind(d/'steps.jsonl').read_text().splitlines()]
            assert len(lines)==1000
            for t,rec in zip(range(1000,0,-1),lines):
                assert rec['time']==t and rec['remaining_time']==t-1 and rec['exact_coverage'] and rec['blocks']==1 and rec['covered_slots']==binding['active_slots']
                assert len(rec['uniform_hash'])==64 and len(rec['query_hash'])==64
            for t in plan['times']:
                z=np.load(bind(d/f'Xt_{t:04d}.npz'));assert int(z['t'])==t and np.isin(z['xt'],[0,1]).all() and z['xt'].ndim==1
                assert len(z['xt'])==sc.q.shape[1]
            bits=inc.assets.bits(sc,np.load(gp));ms,_,_=inc.matrices(sc,bits);score,_,_=inc.solved(ms,1,'increment')
            records.append(dict(model=j['id'],dataset=ds,train_seed=j['train_seed'],index=idx,sample_seed=plan['seed'],model_hash=mh,path=str(d),seconds=su['seconds'],score=score['score'],degree_part=score['degree'],heat_part=score['heat'],snapshot_count=8,steps_verified=1000))
            paths.append(dict(index=idx,path=str(d),sample_seed=plan['seed'],times=plan['times']))
        # Other arms repeating the same seeds cannot supply extra independent samples.
        dups=[]
        for sibling in OLD.glob(f"{ds}_{j['train_seed']}_A_*/{ds}_T_eager/train_round1_seed*/identity.json"):
            z=read(sibling)
            if z['model_hash']==mh and z['sample_seed'] in {x['seed'] for _,x in plans} and str(sibling.parent) not in {x['path'] for x in paths}:dups.append(str(sibling.parent))
        rejected_duplicates.extend(dict(model=j['id'],path=q,reason='same policy/sample seed; not an independent trajectory') for q in dups)
        models.append(dict(**frozen_j,readout=str(readout),initial_complete=str(statefile),model_hash=mh,data=row['data'],config=cfg,reused=paths,duplicates_not_counted=len(dups)))
        contracts.append(dict(model=j['id'],shared_code_files_verified=len(shared),parent_parameters_exact=True,initial_feedback_optimizer_empty=True,initial_EMA_RNG_boundary_previously_verified=True,coverage_steps=6000,snapshots=48,old_semantics='generic reward never affects collection before first update',random_tensor_replay='Recorded hashes retained; no new GPU recomputation/bitwise certification'))
        del state,parent
    # Budget includes new chains only; per-device reservation starts on acquisition.
    chain_cost={ds:dict(mean_seconds=float(np.mean([r['seconds'] for r in records if r['dataset']==ds])),
                        max_seconds=float(max(r['seconds'] for r in records if r['dataset']==ds)))
                for ds in ('IMDB','PubMed')}
    chain_hours=sum(20*chain_cost[ds]['mean_seconds'] for ds in chain_cost)/3600
    upper_chain_hours=sum(20*chain_cost[ds]['max_seconds']*1.2 for ds in chain_cost)/3600
    pcount=6647441
    budget=dict(historical_chain_cost=chain_cost,new_chains=40,new_chain_nominal_GPU_hours=chain_hours,new_chain_max_times_1_2_GPU_hours=upper_chain_hours,
        diagnostic_forwards=512,diagnostic_backwards=512,optimizer_updates=0,new_trajectory_sampling_forwards=40000,
        gradient_load_save_allowance_GPU_hours=.7,engineering_allocation_allowance_GPU_hours=.3,
        proposed_GPU_hours_cap=9,proposed_wall_minutes=120,science_stop_offset_minutes=100,delivery_offset_minutes=120,
        concurrent_GPUs_max=8,collection_queue='two workers per parent, five new chains each; release collection allocation after task; no idle reserved cards',
        nominal_longest_collection_minutes=5*chain_cost['PubMed']['mean_seconds']/60,
        conservative_collection_minutes=5*chain_cost['PubMed']['max_seconds']*1.2/60,
        gradients='four jobs: per trajectory average 8 gradients, retain 16 vectors/model; CPU assemble 4/8/16 estimators and parameter-group Gram matrices; no optimizer.step',
        vector_storage_GiB=4*16*pcount*4/2**30,disk_reservation_GiB=15,
        memory='historical GPU allocated3.2GiB / process~3.5GiB; reserve5GiB; serialized gradient CPU jobs<=8GiB each, two maximum',
        uncertain='Eight-way CPU/disk contention, new worker integration and source loading. Stop before sampling if engineering actual prediction does not fit caps; never reduce panel silently.')
    assert upper_chain_hours+1 < 9
    protocol=dict(status='prepared_not_launched',scope='fixed-policy full-chain eager feedback reliability',score='frozen heat increment D+H_delta; no new coefficients/bandwidth/reward normalization',models=models,
        counts=dict(parents=4,trajectories_per_parent=16,reused_trajectories=24,new_trajectories=40,total_trajectories=64,feedback_snapshots=512,formal_training_updates=0),
        estimator='v_k=(1/8)sum_t grad log p_theta(G0_k|Xt_k,t); a_k=(K*C_k-sum C)/(K-1); g_K=.01/K sum a_k*v_k; report gradient-descent direction -g_K',
        interpretation='LOO estimator is a sample covariance of cost and eager vector; not an unbiased gradient of final expected reward. Sampling includes trajectory and selected-time variability.',
        checks=['unchanged four parent parameters/config/data/schema/prior; full-state restore; no update/EMA drift',
          'complete R1/T1000 logs, fixed support, model and RNG unchanged; full restart boundary; no tolerance relaxation',
          'scalar cost/gradient finite; zero direction marked undefined cosine; gradients not clipped or Adam-transformed for estimator comparison',
          'timed engineering segment counts toward envelope; complete full panel fit required before new chains'],
        reporting=['disjoint quartet pair cosines/ranges and norms; two disjoint8 comparisons;16 full direction descriptive only',
          'leave-one-trajectory-out direction and contribution concentration; no sample removal',
          'backbone/CN/degree/joint head nonoverlapping groups; degree and heat cost feedback disagreements',
          'raw graph metrics all64: joint relation degree, marginal degree, low/hub, triangles/MP3/composition, connectivity, count; scoring association not quality improvement',
          'old6/new10 per model separately; viewed development seeds, no independent training confirmation'],
        decision='No automatic training gate from one cosine. Consistent disjoint evidence may justify one later multi-update/full-generation test; persistent wrong graph preference means score issue, inconsistent feedback means sample insufficiency, ambiguous means unresolved. Neither more K nor stable direction implies good quality.',
        budget=budget,execution_authorized=False,no_actual_deadlines_bound=True)
    write('protocol.json',protocol);table('reused_trajectories.csv',records);write('contracts.json',contracts);write('duplicates_not_counted.json',rejected_duplicates)
    for q in (Path(inc.__file__),Path(inc.assets.__file__),Path(__file__).with_name('transport.py'),Path(inc.assets.heatmod.__file__)):bind(q)
    for q,h in bindings.items():assert sha(q)==h
    write('bindings.json',bindings)
    write('preparation_receipt.json',dict(status='prepared',seconds=time.time()-start,CPU_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        model_calls=0,CUDA_initialized=torch.cuda.is_initialized(),GPU_hours=0,new_graphs=0,original_hashes_unchanged=True,binding_count=len(bindings),
        algebra='finite population all-subsets mean of LOO covariance estimates verified for K2/3/4',next='budget confirmation and isolated GPU engineering/collection; no launch this preparation'))
    print(json.dumps(dict(counts=protocol['counts'],budget=budget,seconds=time.time()-start),indent=2))

if __name__=='__main__':main()
