"""One bounded engineering ticket. No sampling. Gates precede ALL updates."""
import os,sys,pathlib,json,time,hashlib,copy,gc,resource,signal,traceback
BASE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'package'))
import numpy as np
import torch
from prepare import load_case,save,sha,ROOT
from probe import equal,metric,flatten,compare_diagnostics,rng_get,rng_set,input_hash,traces,first_trace
from model_variants import StructuralDenoiser,copy_common
from normal2 import normal2,pair_inputs,sync
from hsgd.bw_preflight.common import cpu_tree,environment
from hsgd.computational import FrozenTypedContext
from hsgd.vendor.tie_extrema import equal_tie_max_pool
from torch_geometric.nn.pool import global_max_pool
import hsgd.vendor.layers as layers

REPO=pathlib.Path('/data2/lyh/gnn_project/DiHug')
OUT=REPO/'results/minmax_tie_mean_v1'
OLD=REPO/'results/minmax_rebuilt_diag_v1'
V2=REPO/'results/typed_refex_rolx_192_v2/probe'
ledger=json.loads((OUT/'activation.json').read_text())
status=dict(status='running',operator=None,cells={},actual_updates={},formal_training=0,generated_graphs=0)
def check():
    if time.time()>=ledger['compute_deadline_epoch']:raise TimeoutError('delivery reserve reached')
    if (OUT/'STOP').exists():raise InterruptedError('STOP')
def flush():save(OUT/'summary.json',status)
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def tensor_manifest(s):return {k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=digest(v)) for k,v in s.items()}
def exact_tree_structure(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact_tree_structure(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact_tree_structure(x,y) for x,y in zip(a,b))
    return a==b
def formula(x,b,g,size):
    # Independent CPU grouped reference, empty groups have no participating input.
    y=torch.zeros((size,x.shape[1]),dtype=x.dtype);grad=torch.zeros_like(x);counts=torch.zeros_like(y,dtype=torch.long)
    for k in range(size):
        idx=b==k
        if bool(idx.any()):
            v=x[idx].max(0).values;y[k]=v;mask=x[idx]==v;counts[k]=mask.sum(0)
            grad[idx]=mask*g[k]/counts[k]
    return y,grad,counts
def run_op(x,b,g,size,device,fn=equal_tie_max_pool):
    a=x.to(device).clone().requires_grad_(True);bb=b.to(device);gg=g.to(device)
    sync(device);start=time.perf_counter();y=fn(a,bb,size=size);y.backward(gg);sync(device)
    return y.detach().cpu(),a.grad.detach().cpu(),time.perf_counter()-start

def operator_gate():
    start=time.perf_counter();cap=torch.load(OLD/'captured.pt',map_location='cpu',weights_only=False)
    oldhash=json.loads((OLD/'captured_tensor_hashes.json').read_text())
    for n,c in cap.items():
        for k,v in c.items():
            if torch.is_tensor(v):assert digest(v)==oldhash[n][k]['content_sha256']
    save(OUT/'capture_binding.json',dict(file=str(OLD/'captured.pt'),sha256=sha(OLD/'captured.pt'),tensor_hashes=oldhash,source_sha256=sha(OLD/'binding.json')))
    reports={};allpass=True
    for n,c in cap.items():
        check();x,b,g=c['src'],c['batch'],c['upstream'];size=c['dim_size'];yref,gr,counts=formula(x,b,g,size)
        records=[];arrays={};first=None
        for i in range(20):
            y,grad,sec=run_op(x,b,g,size,'cuda');first=first or (y,grad)
            sums=torch.stack([grad[b==k].double().sum(0) for k in range(size)])
            rec=dict(repeat=i,forward_exact=torch.equal(y,c['output']),repeat_forward_exact=torch.equal(y,first[0]),repeat_gradient=metric(grad,first[1]),formula=metric(grad,gr),conservation=metric(sums,g),nonextreme_nonzero=int(((grad!=0)&(x!=y[b])).sum()),seconds=sec)
            records.append(rec);arrays[str(i)+'_output']=y.numpy();arrays[str(i)+'_gradient']=grad.numpy()
        np.savez_compressed(OUT/(n+'_gpu20.npz'),**arrays)
        cy,cg,cs=run_op(x,b,g,size,'cpu');oy,og,os_=run_op(x,b,g,size,'cuda',global_max_pool)
        np.savez_compressed(OUT/(n+'_cpu.npz'),output=cy.numpy(),gradient=cg.numpy())
        # Internal max U=-X, outer minimum -max(U): dL/dX=-dL/dU.
        minsign=None
        if n=='min_internal_max':
            a=(-x).cuda().requires_grad_();v=-equal_tie_max_pool(-a,b.cuda(),size);v.backward(-g.cuda());minsign=metric(a.grad.cpu(),-first[1]);assert minsign['passes']
        passed=all(r['forward_exact'] and r['repeat_forward_exact'] and r['repeat_gradient']['exact'] and r['formula']['passes'] and r['conservation']['passes'] and r['nonextreme_nonzero']==0 for r in records)
        passed=passed and torch.equal(oy,yref) and torch.equal(cy,yref) and metric(cg,gr)['passes']
        reports[n]=dict(passes=passed,unique_channels=int((counts==1).sum()),tied_channels=int((counts>1).sum()),records=records,cpu_formula=metric(cg,gr),cpu_gpu=metric(cg,first[1]),old_forward_exact=torch.equal(oy,first[0]),old_forward_backward_seconds=os_,new_forward_backward_seconds=[r['seconds'] for r in records],cpu_seconds=cs,min_outer_sign=minsign)
        allpass &= passed
    # Fixed synthetic panel: all-negative, zeros, positive extrema; groups 1/3 empty.
    x=torch.tensor([[-3.,0.,2.],[-3.,0.,1.],[-8.,-2.,2.],[4.,-9.,0.],[1.,-9.,0.]])
    b=torch.tensor([0,0,0,2,2]);g=torch.tensor([[.3,-.2,.8],[.7,.2,.1],[-.4,.6,.5],[.9,.1,.2]])
    synthetic={}
    for device in ['cpu','cuda']:
        for sign in [1,-1]:
            yref,gr,cnt=formula(sign*x,b,g,4);y,grad,_=run_op(sign*x,b,g,4,device)
            assert torch.equal(y,yref) and metric(grad,gr)['passes']
            perm=torch.tensor([4,1,3,0,2]);yp,gp,_=run_op(sign*x[perm],b[perm],g,4,device)
            back=torch.empty_like(gp);back[perm]=gp
            assert torch.equal(yp,y) and torch.equal(back,grad)
            # Original edge variables feed TWO duplicate representations. Chain
            # rule must sum their gradients back, not compare just one replica.
            a=(sign*x).to(device).requires_grad_();out=equal_tie_max_pool(a.repeat((2,1)),b.to(device).repeat(2),4);out.backward(g.to(device))
            assert torch.equal(out.cpu(),y) and metric(a.grad.cpu(),grad)['passes']
            synthetic[device+str(sign)]=dict(forward_exact=True,formula=True,permutation_forward_gradient_exact=True,duplicate_pullback=metric(a.grad.cpu(),grad),empty_groups=[1,3],counts=cnt.tolist())
        for invalid in [float('nan'),float('inf'),-float('inf')]:
            bad=x.clone();bad[0,0]=invalid
            try:run_op(bad,b,g,4,device)
            except FloatingPointError:pass
            else:raise AssertionError('nonfinite accepted')
        ey,eg,_=run_op(torch.empty((0,3)),torch.empty(0,dtype=torch.long),g,4,device)
        assert torch.equal(ey,torch.zeros_like(g)) and eg.shape==(0,3)
    a=torch.tensor([[1.1,3.7],[2.8,-.4],[-4.1,5.6]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda z:equal_tie_max_pool(z,torch.tensor([0,0,0]),1),(a,),atol=1e-5,rtol=1e-3)
    reports.update(synthetic=synthetic,smooth_gradcheck=True,passes=allpass,elapsed_seconds=time.perf_counter()-start)
    save(OUT/'operator.json',reports);return reports

def saved_pair(dataset,case):
    path=V2/(dataset+'_unified_inputs.npz');z=np.load(path)
    if dataset=='IMDB':assert sha(path)=='9d249b717066ec78981f3e01e33405766fd0f85cbcd686c326ca3bf7b9ff4635'
    inputs=[{k:z[str(i)+'_'+k] for k in ['ids','f','query','labels','edges','edge_families']} for i in range(2)]
    for inp in inputs:inp.update(t=500,ce_weight=1/case['space'].total,round_id=0)
    # Deterministic reconstruction verifies both saved datasets against ORIGINAL rule.
    _,cfg=load_case(dataset);regen=pair_inputs(case,cfg,50,np.random.default_rng(2026102801),500)
    assert input_hash(inputs)==input_hash(regen)
    save(OUT/(dataset+'_input_binding.json'),dict(file=str(path),sha256=sha(path),t=500,t_source='original probe force_t=500, NPZ does not contain t',input_content=input_hash(inputs),regeneration_exact=True,slots=[len(x['ids']) for x in inputs],Xt_edges=[len(x['edge_families']) for x in inputs]))
    return inputs

def setup_model(dataset,arm):
    case,cfg=load_case(dataset);e=cfg['datasets'][dataset];assert sha(ROOT/e['basis_file'])==e['basis_sha256'];basis=np.load(ROOT/e['basis_file'])['basis']
    torch.manual_seed(cfg['seed']);anchor=StructuralDenoiser(case,cfg,'A')
    # This private NumPy construction between anchor/model is part of original flow.
    pair_inputs(case,cfg,50,np.random.default_rng(2026102801),500)
    torch.manual_seed(cfg['seed']);m=StructuralDenoiser(case,cfg,arm,basis);copied=copy_common(anchor,m)
    shared={k:torch.equal(v,m.state_dict()[k]) for k,v in anchor.state_dict().items() if k in m.state_dict() and v.shape==m.state_dict()[k].shape}
    assert all(shared.values())
    return case,cfg,basis,m,dict(copied=copied,shared_exact=shared,rule='seed, anchor, private NumPy pair, reset seed, model, copy_common')

def full_forward(m,c,inp,T,old=False):
    previous=layers.equal_tie_max_pool
    if old:layers.equal_tie_max_pool=global_max_pool
    ctx=None
    try:
        m.eval();sync('cuda');start=time.perf_counter();ctx=FrozenTypedContext(m,c['space'].types,inp['edges'],inp['edge_families'],'timing')
        with torch.no_grad():_,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/T,'timing',details=True)
        sync('cuda');seconds=time.perf_counter()-start
        return dict(seconds=seconds,includes_backward=False,includes_cpu_hooks=False,synchronized=True,slots=len(inp['ids']),forward_count=1),logits.cpu(),p.cpu()
    finally:
        if ctx:ctx.close()
        layers.equal_tie_max_pool=previous

def six_gates():
    for dataset in ['IMDB','PubMed']:
        for arm in 'ABC':
            check();label=dataset+'_'+arm;folder=OUT/label;folder.mkdir();row=dict(status='running',updates=0);status['cells'][label]=row;flush();print('GATE',label,flush=True)
            case,cfg,basis,m,receipt=setup_model(dataset,arm);inputs=saved_pair(dataset,case);initial=cpu_tree(m.state_dict());rng=np.random.default_rng(2026102802)
            snap=dict(model=initial,rng=rng_get(rng,'cuda'),config=cfg,construction=receipt,purpose='reconstructed initial, before any forward',source_commit=ledger['source_commit'])
            torch.save(snap,folder/'initial.pt');save(folder/'initial_tensor_hashes.json',tensor_manifest(initial));row['initial_sha256']=sha(folder/'initial.pt')
            if label=='IMDB_A':
                old=torch.load(OLD/'reconstructed_initial.pt',map_location='cpu',weights_only=False)
                key='state_dict' if 'state_dict' in old else 'model';assert equal(initial,old[key]);row['matches_previous_reconstruction']=True;del old
            # Every initial saved before first forward; comparison uses fixed restored RNG.
            optargs={k:v for k,v in cfg['optimizer'].items() if k!='type'};opt=torch.optim.AdamW(m.parameters(),**optargs)
            cpu={};rng_set(snap['rng'],rng,'cuda');row['cpu_normal2']=normal2(m,opt,case,inputs,cfg['T'],'cpu',False,cpu)
            m=m.cuda();opt=torch.optim.AdamW(m.parameters(),**optargs);torch.cuda.reset_peak_memory_stats();dev=[];tr=[]
            for repeat in range(2):
                check();rng_set(snap['rng'],rng,'cuda');d={};trace=[];handles=traces(m,trace)
                try:rec=normal2(m,opt,case,inputs,cfg['T'],'cuda',False,d)
                finally:
                    for h in handles:h.remove()
                row.setdefault('gpu_normal2',[]).append(rec);dev.append(d);tr.append(trace)
            same=compare_diagnostics(dev[0],dev[1]);same['first_forward_hash_difference']=first_trace(tr[0],tr[1]);save(folder/'same_gpu.json',same)
            cross=compare_diagnostics(cpu,dev[0]);save(folder/'cpu_gpu.json',cross);save(folder/'traces.json',tr)
            row['same_gpu_passed']=same['passes'];row['cpu_gpu_passed']=cross['passes'];row['all_gpu_gradients_bitwise']=all(x['exact'] for x in same['gradients'].values())
            row['first_exceeding_gradient']=next((dict(name=k,**v) for k,v in same['gradients'].items() if not v['passes']),None)
            if not same['passes']:torch.save(dict(first=dev[0],second=dev[1]),folder/'failed_repeat_tensors.pt')
            # Forward cost pair on same initial weights and same full query; no hooks.
            timings=[];baseline=None
            for old in [True,False,True,False]:
                check();rec,l,p=full_forward(m,case,inputs[0],cfg['T'],old);rec['path']='old' if old else 'equal_tie';baseline=baseline or (l,p)
                rec['logits_vs_first']=metric(l,baseline[0]);rec['p0_vs_first']=metric(p,baseline[1]);timings.append(rec)
            row['full_query_forward_cost']=timings;row['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20;row['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20;row['rss_peak_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;row['status']='gradient_checked';flush()
            print('END_GATE',label,'sameGPU',row['same_gpu_passed'],'CPU_GPU',row['cpu_gpu_passed'],flush=True)
            del m,opt,cpu,dev,tr,snap,initial,same,cross,baseline;gc.collect();torch.cuda.empty_cache()

def update_phase():
    assert status['operator'] and all(v['same_gpu_passed'] for v in status['cells'].values()) and len(status['cells'])==6
    for label,row in status['cells'].items():
        check();dataset,arm=label.split('_');folder=OUT/label;case,cfg,basis,m,_=setup_model(dataset,arm)
        initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False);m.load_state_dict(initial['model']);m=m.cuda();args={k:v for k,v in cfg['optimizer'].items() if k!='type'};opt=torch.optim.AdamW(m.parameters(),**args);rng=np.random.default_rng(2026102802);rng_set(initial['rng'],rng,'cuda')
        binding=dict(source=source,config=cfg,arm=arm,data=cfg['datasets'][dataset],environment=env,initial_sha256=row['initial_sha256']);hist=[];continuous=None;previous=None
        for seq,pos in enumerate([1,2,2]):
            check()
            if seq==2:
                ck=torch.load(folder/'after1.pt',map_location='cpu',weights_only=False);assert sha(folder/'after1.pt')==row['checkpoint_sha256'] and ck['binding']==binding and ck['updates']==1
                del m,opt;gc.collect();m=StructuralDenoiser(case,cfg,arm,basis).cuda();m.load_state_dict(ck['model']);opt=torch.optim.AdamW(m.parameters(),**args);opt.load_state_dict(copy.deepcopy(ck['optimizer']));rng_set(ck['rng'],rng,'cuda')
                row['load_exact']=dict(model=equal(cpu_tree(m.state_dict()),ck['model']),optimizer=equal(cpu_tree(opt.state_dict()),ck['optimizer']),rng=equal(rng_get(rng,'cuda'),ck['rng']),updates=ck['updates']==1,binding=ck['binding']==binding)
                assert all(row['load_exact'].values())
            inp=pair_inputs(case,cfg,pos,rng,500);ih=input_hash(inp)
            if seq==2:assert ih==hist[1]['input_sha256']
            assert row['updates']<3;row['updates']+=1;status['actual_updates'][label]=row['updates'];flush()
            d={};rec=normal2(m,opt,case,inp,cfg['T'],'cuda',True,d);rec.update(input_sha256=ih,restored=seq==2,position=pos);hist.append(rec);save(folder/'updates.json',hist)
            if seq==0:
                start=time.perf_counter();snap=dict(model=cpu_tree(m.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=rng_get(rng,'cuda'),updates=1,binding=binding,purpose='temporary engineering only')
                tmp=folder/'after1.tmp';torch.save(snap,tmp);tmp.replace(folder/'after1.pt');row['save_seconds']=time.perf_counter()-start;row['checkpoint_sha256']=sha(folder/'after1.pt');del snap
            if seq==1:continuous=dict(model=cpu_tree(m.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng=rng_get(rng,'cuda'));previous=d
            if seq==2:
                cmp=compare_diagnostics(previous,d);cmp['parameters']={k:metric(v,m.state_dict()[k]) for k,v in continuous['model'].items()};now=cpu_tree(opt.state_dict());cmp['optimizer_structure_discrete_exact']=exact_tree_structure(continuous['optimizer'],now)
                cmp['optimizer_tensors']={k:metric(v,flatten(now)[k]) for k,v in flatten(continuous['optimizer']).items()};cmp['optimizer_step_exact']=all(equal(v,flatten(now)[k]) for k,v in flatten(continuous['optimizer']).items() if k.endswith('/step'));cmp['optimizer_bitwise']=equal(continuous['optimizer'],now);cmp['rng_exact']=equal(continuous['rng'],rng_get(rng,'cuda'));cmp['input_exact']=ih==hist[1]['input_sha256']
                cmp['passes']=cmp['passes'] and all(x['passes'] for x in cmp['parameters'].values()) and all(x['passes'] for x in cmp['optimizer_tensors'].values()) and cmp['optimizer_structure_discrete_exact'] and cmp['optimizer_step_exact'] and cmp['rng_exact'] and cmp['input_exact']
                save(folder/'restore.json',cmp);row['restore_passed']=cmp['passes'];row['restore_all_float_bitwise']=all(x['exact'] for x in cmp['parameters'].values()) and cmp['optimizer_bitwise'] and all(x['exact'] for x in cmp['gradients'].values())
                if not cmp['passes']:raise RuntimeError(label+' restore exceeds original tolerance; no further updates')
        row['status']='temporary_updates_complete';flush();print('RESTORE',label,row['restore_passed'],row['restore_all_float_bitwise'],flush=True)
        del m,opt,initial,continuous,previous,ck,d;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda s,f:(_ for _ in ()).throw(TimeoutError('compute deadline')))
    signal.alarm(max(1,int(ledger['compute_deadline_epoch']-time.time())))
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,lambda s,f:(_ for _ in ()).throw(InterruptedError('signal '+str(s))))
    try:
        check();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        env=environment('cuda');save(OUT/'environment.json',env)
        source={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.npz'] and '__pycache__' not in str(p) and 'review_repo' not in str(p)}
        save(OUT/'frozen_source.json',source);save(OUT/'contract.json',dict(atol=2e-6,rtol=1e-5,operator_repeat=20,six_cells_repeat=2,update_cap_per_cell=3,optimizer_load='exact',optimizer_continuation='original tolerance; discrete state exact',tie='FP exact equality, equal-share gradient',empty='original PyG zero, no fill counted',input='saved v2 full pair t500; normal2 denominator2*18336',cpu_gpu='separate gate, not sameGPU',scientific_training_authorized=False))
        status['operator']=operator_gate()['passes'];flush()
        if not status['operator']:raise RuntimeError('operator gate failed')
        six_gates()
        if all(v['same_gpu_passed'] for v in status['cells'].values()) and len(status['cells'])==6:update_phase()
        else:status['updates_blocked_reason']='one or more sameGPU full normal2 gates failed'
        status['status']='completed_engineering_checks'
    except BaseException as e:
        status.update(status='incomplete_or_failed',error=repr(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        status['finished_epoch']=time.time();status['elapsed_from_ticket_seconds']=time.time()-ledger['start_epoch'];status['optimizer_step_calls']=sum(status['actual_updates'].values());flush()
        print(json.dumps({k:v for k,v in status.items() if k!='cells'},indent=2),flush=True)
