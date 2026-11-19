"""Bounded CPU diagnosis: two read-only forwards, no optimizer/GPU/chains.

Reconstructed CPU draws are NOT asserted bitwise equal to unretained CUDA
draws. All original assets remain immutable; alternative eigensolvers only
diagnose the same mathematical heat operator, not change official scores.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json,signal,traceback,importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import eigh,expm
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'results/three_arm_learning_pilot_20261111T211641'
OUT=P/'failure_audit';OUT.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('audit_support',Path(__file__).with_name('update_preflight.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
torch,w,base=s.torch,s.w,s.base
start=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU audit 240 seconds')));signal.alarm(240)
results=[]
def save(name,x):base.write(OUT/name,x)
def numerical(a,types):
    n=len(a);d=a.sum(1);inv=np.divide(1,np.sqrt(d),out=np.zeros(n),where=d>0)
    L=np.diag((d>0).astype(float))-inv[:,None]*a*inv[None,:]
    ind=np.column_stack([types==t for t in np.unique(types)]).astype(float);root=np.sqrt(d)[:,None]*ind
    rows=[];reaches={}
    for driver in ('evr','evd','evx'):
        ev,v=eigh(L,driver=driver);hs=np.exp(-base.heat.TIMES[:,None]*ev)
        reach=np.stack([inv[:,None]*(v@(h[:,None]*(v.T@root))) for h in hs],axis=1);reach[d==0]=ind[d==0,None,:]
        reaches[driver]=reach
        rows.append(dict(driver=driver,mass_error=float(abs(reach.sum(2)-1).max()),minimum=float(reach.min()),
            orthogonality_max=float(abs(v.T@v-np.eye(n)).max()),eigen_residual_max=float(abs(L@v-v*ev).max()),eig_min=float(ev.min())))
    # Independent matrix exponential for the most discrepant original time.
    j=int(np.unravel_index(np.argmax(abs(reaches['evr'].sum(2)-1)),(n,len(base.heat.TIMES)))[1])
    H=expm(-float(base.heat.TIMES[j])*L);r=inv[:,None]*(H@root);r[d==0]=ind[d==0]
    return dict(drivers=rows,check_time=float(base.heat.TIMES[j]),expm_mass_error=float(abs(r.sum(1)-1).max()),
        evr_vs_expm=float(abs(reaches['evr'][:,j]-r).max()),evd_vs_expm=float(abs(reaches['evd'][:,j]-r).max()),
        evr_evd_reach_max=float(abs(reaches['evr']-reaches['evd']).max()),nodes=n,edges=int(a.sum()/2),isolates=int((d==0).sum()))
try:
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads((P/'protocol.json').read_text())
    for ds in ('IMDB','PubMed'):
        folder=P/(ds+'_S16');state=torch.load(folder/'stopped_complete.pt',map_location='cpu',weights_only=False)
        assert state['recovery']['exact_resume_allowed'] and state['updates']==state['ema']['updates']
        row=protocol['parents'][ds];z=torch.load(row['readout'],map_location='cpu',weights_only=False);cfg=z['binding']['config'];del z
        data=np.load(row['data']);edges=data['edges'];edges=edges.T if edges.shape[0]!=2 else edges
        case=w.common.make_case(data['types'],edges,tuple(map(tuple,data['schema'])))
        m=w.StructuralDenoiser(case,cfg,'A').cpu();m.load_state_dict(state['model'],strict=True);m.eval()
        before=s.state_digest(m.state_dict());rng=np.random.default_rng();rng.bit_generator.state=state['rng']['numpy']
        u=state['updates']+1;inp=w.common.train_input(case,1000,1,row['train_seed'],u,rng)
        tick=time.time()
        with torch.no_grad():logits,prob=w.runtime.predict(m,case['space'].types,inp['edges'],inp['edge_families'],inp['query'],inp['f'],inp['t']/1000,details=True)
        seconds=time.time()-tick
        assert before==s.state_digest(m.state_dict())
        ids,fam,q=w.common.plan_for(case,2026111402,1).block(0)
        ix=np.argsort(inp['ids']);bi=np.argsort(ids);assert np.array_equal(inp['ids'][ix],ids[bi])
        pp=torch.log_softmax(logits.double(),dim=-1)[torch.as_tensor(ix[np.argsort(bi)])][:,1].exp().numpy()
        aux=np.random.default_rng(np.random.SeedSequence([2026111404,row['train_seed'],u]));uniform=aux.random((16,len(ids)));bits=(uniform<pp).astype(np.uint8)
        np.savez_compressed(OUT/(ds+'_CPU_reconstructed.npz'),bits=bits,prob=pp,uniform=uniform,ids=ids,query=q,families=fam,t=inp['t'])
        rr=dict(dataset=ds,completed_updates=state['updates'],failed_position=u,t=inp['t'],input_hash=s.input_hash([inp]),
            CPU_forward_seconds=seconds,minimum_draw_margin=float(abs(uniform-pp).min()),CPU_reconstruction_not_CUDA_bitwise_proof=True,
            snapshot_sha256=base.sha(folder/'stopped_complete.pt'),model_unchanged=True,graphs=[])
        for k,b in enumerate(bits):
            a=base.heat.adj(len(case['space'].types),q[:,b.astype(bool)].T)
            try:base.heat.heat(a,case['space'].types);passed=True
            except AssertionError:passed=False
            item=dict(sample_index=k,original_guard_passed=passed,edges=int(b.sum()))
            if not passed:item['numerical']=numerical(a,case['space'].types)
            rr['graphs'].append(item)
        results.append(rr);save('reconstruction.json',results);del state,m
    # Saved records only: stratified trends and endpoint CE versus same-input baseline.
    trends=[];validation=[];objective=[]
    for ds in ('IMDB','PubMed'):
        for arm in ('S1','S16'):
            records=[json.loads(x) for x in (P/(ds+'_'+arm)/'updates.jsonl').read_text().splitlines()]
            for lo in range(1,len(records)+1,100):
                block=[r for r in records if lo<=r['update']<lo+100]
                for name,tlo,thi in [('all',1,1000),('low',1,100),('middle',101,700),('high',701,1000)]:
                    rr=[r for r in block if tlo<=r['records'][0]['t']<=thi]
                    if rr:trends.append(dict(dataset=ds,arm=arm,from_update=lo,to_update=min(lo+99,len(records)),band=name,n=len(rr),
                        mean_cost=float(np.mean([np.mean(r['records'][0]['costs']) for r in rr])),
                        mean_joint=float(np.mean([np.mean(np.array(r['records'][0]['components'])[:,0]) for r in rr])),
                        mean_count=float(np.mean([np.mean(np.array(r['records'][0]['components'])[:,1]) for r in rr])),
                        clipped=sum(r['clip_scale']<1 for r in rr),gradient_max=max(r['gradient_norm_before_clip'] for r in rr)))
        for readout in ('raw','EMA'):
            basev=json.loads((P/(ds+'_baseline')/f'validation_{readout}.json').read_text())
            for arm in ('S1','T_eager'):
                after=json.loads((P/(ds+'_'+arm)/f'validation_{readout}.json').read_text())
                for a,b in zip(basev,after):
                    assert a['input_hash']==b['input_hash']
                    validation.append(dict(dataset=ds,arm=arm,readout=readout,t=a['t'],baseline_CE=a['CE'],after_CE=b['CE'],
                        baseline_expected_edges=sum(r['expected_edges'] for r in a['relations']),after_expected_edges=sum(r['expected_edges'] for r in b['relations'])))
    save('training_trends.json',trends);save('validation_comparison.json',validation)
    # Exhaustive two-Bernoulli check of minimization sign and exact baseline invariance.
    p=np.array([.2,.35]);target=1.;expected=np.zeros(2);centered=np.zeros(2);mean_cost=0.
    import itertools
    for zz in itertools.product([0,1],repeat=2):
        z=np.array(zz);pr=np.prod(np.where(z,p,1-p));C=(z.sum()-target)**2;expected+=pr*C*(z-p);centered+=pr*(C-.7)*(z-p);mean_cost+=pr*C
    exact=p*(1-p)*(2*(p.sum()-target)+1-2*p)
    np.testing.assert_allclose(expected,exact,atol=1e-14);np.testing.assert_allclose(centered,expected,atol=1e-14)
    save('sign_check.json',dict(exact_gradient=exact.tolist(),enumerated=expected.tolist(),centered=centered.tolist(),
        note='Positive cost times log probability is the correct gradient for cost minimization; baseline changes variance, not expectation. Does not certify finite-sample AdamW trajectory.'))
    save('receipt.json',dict(status='complete',CPU_model_forwards=2,GPU=0,optimizer_updates=0,new_reverse_chains=0,seconds=time.time()-start,
        original_assets_unchanged=True,source_sha256=base.sha(__file__)))
except BaseException:
    save('failure.json',dict(traceback=traceback.format_exc(),seconds=time.time()-start));raise
