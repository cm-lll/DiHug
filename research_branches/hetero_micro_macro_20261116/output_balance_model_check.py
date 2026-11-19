"""Four raw128 states, two temporary AdamW updates each; no saved model adoption."""
import os,sys,json,time,signal,subprocess,traceback,hashlib,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREVIOUS=ROOT/'results/output_balance_model_check_20261116'
OUT=ROOT/'results/output_balance_model_check_20261116_v2'
SRC=ROOT/'results/hetero_micro_macro_effect_run_20261116T083122'
PANEL=ROOT/'results/paired_response_check_20261116'
def write(name,x):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker():
    import effect_worker as ew
    torch,np,s,w=ew.torch,ew.np,ew.s,ew.w
    from gradient_scale import compose
    manifest=json.loads((PANEL/'input_manifest.json').read_text());coeff=json.loads((SRC/'gradient_coefficients.json').read_text())
    rows=[];updates=[];bindings={};calls=0;vjp=0
    for index in range(4):
        c=ew.Context(OUT,index);seed=c.seed;cp=SRC/c.row['id']/'calibrated_micro_macro/step128_complete.pt'
        completion=json.loads((cp.parent/'completion.json').read_text());assert sha(cp)==completion['checkpoint_sha256']
        bindings[str(cp)]=sha(cp);st=torch.load(cp,map_location='cpu',weights_only=False);assert st['updates']==128
        # Use raw128 with its actual Adam/EMA state, not EMA parameters with raw Adam.
        source_digest=s.state_digest(st)
        c.state=copy.deepcopy(st);c.boundary=c.restore(copy.deepcopy(st));before=s.state_digest(c.checkpoint());params=list(c.m.parameters())
        sample=min(e['sample_seed'] for e in manifest if e['train_seed']==seed);entries=[next(e for e in manifest if e['train_seed']==seed and e['sample_seed']==sample and e['kind']==kind) for kind in ['normal','free']]
        inputs=[]
        for ent in entries:
            assert sha(ent['input'])==ent['input_sha256'];bindings[ent['input']]=ent['input_sha256'];z=np.load(ent['input'])
            query=z['query'];fam=z['f'];fullq=z['fullquery'];fullf=z['fullfamily'];mask=z['xt'].astype(bool)
            ee=fullq[:,mask];ef=fullf[mask];labels=np.array([c.ref[c.keymap[tuple(sorted(e))]] for e in query.T])
            inputs.append(dict(edges=ee,edge_families=ef,query=query,f=fam,t=int(z['t']),labels=labels))
        def evaluate(arm):
            original=s.state_digest(c.checkpoint())
            with torch.no_grad():
                for ent,inp in zip(entries,inputs):
                    lo,pr,terms=c.forward(inp);tp,ix=c.expand(pr,inp);blocks,_=c.blocks(tp)
                    P=tp.new_zeros((len(c.schema),len(c.ty),len(c.ty)));P[c.f,c.q[0],c.q[1]]=tp;P[c.f,c.q[1],c.q[0]]=tp
                    ds=ew.descriptors(P,c.ty,c.plan);closed=ds['closure'];rels=[];lab=inp['labels'].astype(bool);pp=pr.cpu().double().numpy();ts=terms.cpu().double().numpy()
                    for r in range(len(c.schema)):
                        pos=(inp['f']==r)&lab;neg=(inp['f']==r)&~lab
                        rels.append(dict(relation=r,positive_n=int(pos.sum()),negative_n=int(neg.sum()),missed_mass=float((1-pp[pos]).sum()),false_mass=float(pp[neg].sum()),positive_CE=float(ts[pos].mean()) if pos.any() else None,negative_CE=float(ts[neg].mean()) if neg.any() else None))
                    structural=ew.probability_metrics(tp.numpy(),c.ty,c.q,c.f,c.ref,len(c.schema))
                    rows.append(dict(dataset=c.ds,train_seed=seed,arm=arm,kind=ent['kind'],sample_seed=sample,t=inp['t'],CE=float(terms.sum()/c.case['space'].total),
                        positive_CE=float(ts[lab].mean()),negative_CE=float(ts[~lab].mean()),blocks={k:float(v) for k,v in blocks.items()},macro=float(compose(0.,blocks,coeff)),
                        expected_triangles=float(closed.sum()/3),relations=rels,structure=structural,
                        typed_closure_sums={str(int(a)):closed[torch.tensor(c.ty==a)].sum(0).tolist() for a in np.unique(c.ty)}))
            assert s.state_digest(c.checkpoint())==original,'evaluation contaminated state'
        evaluate('before')
        for arm in ['old_combined','output_balanced']:
            c.boundary=c.restore(copy.deepcopy(st));assert s.state_digest(c.checkpoint())==before
            inp=inputs[0];c.o.zero_grad(set_to_none=True)
            with c.boundary.step(129):
                rng_before=s.state_digest(s.rng_capture(c.rng,'cuda'))
                lo,pr,terms=c.forward(inp);ce=terms.sum()/c.case['space'].total;tp,ix=c.expand(pr,inp,True);blocks,_=c.blocks(tp);macro=compose(tp.new_zeros(()),blocks,coeff)
                dp=torch.autograd.grad(macro,tp)[0];pp=pr.detach().cpu().double();lab=torch.tensor(inp['labels'],dtype=torch.float64)
                gCE=(pp-lab)/c.case['space'].total;gM=dp[ix]*pp*(1-pp)
                assert torch.isfinite(gM).all() and gM.norm()>0 and gCE.norm()>0
                factor=float(gCE.norm()/gM.norm()) if arm=='output_balanced' else 1.
                gc=torch.autograd.grad(ce,params,retain_graph=True,allow_unused=True)
                gm=torch.autograd.grad(pr,params,grad_outputs=dp[ix].to(pr),allow_unused=True);vjp+=2
                normc=sum(float(g.double().square().sum()) for g in gc if g is not None)**.5;normm=sum(float(g.double().square().sum()) for g in gm if g is not None)**.5
                dot=sum(float((a.double()*b.double()).sum()) for a,b in zip(gc,gm) if a is not None and b is not None)
                for p,a,b in zip(params,gc,gm):
                    p.grad=None if a is None and b is None else (torch.zeros_like(p) if a is None else a)+(0 if b is None else factor*b)
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
                assert rng_before==s.state_digest(s.rng_capture(c.rng,'cuda'))
                old=[p.detach().clone() for p in params]
                c.boundary.mark_optimizer_started();c.o.step();c.ema.update(c.m,129);c.ema.at_update_boundary=True
                assert all(torch.isfinite(p).all() for p in params)
                delta=[p.detach()-o for p,o in zip(params,old)]
                record=dict(dataset=c.ds,train_seed=seed,arm=arm,multiplier=factor,output_macro_CE_norm_ratio=float(gM.norm()/gCE.norm()),
                    parameter_macro_CE_norm_ratio=normm/normc,parameter_CE_macro_cosine=dot/(normc*normm),gradient_norm_before_original_clip1=norm,
                    actual_displacement_L2=sum(float(d.double().square().sum()) for d in delta)**.5,
                    CE_gradient_dot_actual_update=sum(float((a.double()*d.double()).sum()) for a,d in zip(gc,delta) if a is not None),
                    macro_gradient_dot_actual_update=sum(float((b.double()*d.double()).sum()) for b,d in zip(gm,delta) if b is not None))
            assert c.checkpoint()['updates']==129
            assert s.state_digest(st)==source_digest,'sealed source mutated'
            record['end_state_hash']=s.state_digest(c.checkpoint());updates.append(record);evaluate(arm)
            write('rows.json',rows);write('updates.json',updates)
        c.boundary=c.restore(copy.deepcopy(st));assert s.state_digest(c.checkpoint())==before
        assert sha(cp)==bindings[str(cp)]
        calls+=c.calls;write('progress.json',dict(models_complete=index+1,forwards=calls,parameter_VJPs=vjp))
        del c,params,st,gc,gm,delta,old,lo,pr,terms,tp,blocks,macro,ce;torch.cuda.empty_cache()
    assert calls==32 and vjp==16
    write('bindings.json',bindings);write('completion.json',dict(status='complete',forwards=calls,parameter_VJPs=vjp,temporary_updates=8,new_graphs=0,
        peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20,raw_state_used=True,EMA_updated_not_evaluated=True,original_checkpoints_unchanged=True,script_sha256=sha(__file__)))

def launch():
    start=time.time();old_launch=json.loads((PREVIOUS/'launch.json').read_text());origin=old_launch['start_epoch']
    if start>origin+170:raise RuntimeError('Less than60sec original scientific window; no restart')
    OUT.mkdir(exist_ok=False)
    proto=json.loads((SRC/'protocol.json').read_text());proto['science_stop_epoch']=origin+230
    proto['temporary_diagnostic']=dict(start_epoch=origin,restart_epoch=start,compute_seconds_max=240,allocation_seconds_max=300,models=4,updates=8,new_graphs=0,
        first_attempt=str(PREVIOUS),fix='deep-copy sealed complete state before every load; optimizer step tensor can alias caller state',
        raw128_actual_optimizer=True,selection='minimum pre-existing sample seed per training seed; normal input for update, its paired free state for additional readout',
        rule='replace aggregate macro multiplier1 with detached current-output-logit gradient norm ratio CE/macro; no per-type fitting or search',
        tests='same complete starting state, same inputs; restore between arms; original LR/AdamW/clip1/EMA retained',
        limitation='Output-space balancing need not balance parameter or AdamW updates; normal/free inputs are already viewed development material; no independent quality conclusion',
        source_sha256=sha(__file__))
    write('protocol.json',proto)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
    for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
    with (OUT/'worker.log').open('w') as log:
        child=subprocess.Popen([sys.executable,__file__,'worker'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write('launch.json',dict(start_epoch=start,original_start_epoch=origin,scientific_stop_epoch=origin+230,watchdog_epoch=origin+240,pid=child.pid,GPU=0,GPU_uuid='GPU-12b85b68-69e6-12fa-76ce-986c0b46e8a7'))
        try:child.wait(timeout=max(.1,origin+240-time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        finally:write('release.json',dict(start_epoch=start,end_epoch=time.time(),GPU_hours=(time.time()-start)/3600,exit_code=child.returncode,owned_process_exited=child.poll() is not None))
    if child.returncode:raise SystemExit(child.returncode)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        try:worker()
        except BaseException:write('failure.json',dict(traceback=traceback.format_exc(),temporary_only=True));raise
    else:launch()
