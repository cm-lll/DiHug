"""Fixed original typed-edit panel; engineering outputs never enter scientific tables."""
from common import *
import argparse,traceback
from torch.utils.data import DataLoader
from dihug.core.mixed_edit_diffusion import build_mixed_edit_diffusion_dataset,mixed_edit_candidate_matrix,sample_edit_candidates,load_mixed_edit_checkpoint
from dihug.core.gpu_role_guided_edit import gpu_features_for_proposals,sample_gpu_role_guided_edit_diffusion
from observe import sample as observed_sample,source as observed_source
PROFILE='typed_closure_compact_v2';DATASETS=['imdb','pubmed','freebase','acm'];SEEDS=list(range(810,818));STARTS=list(range(100,108))

def verify():
    for p,h in read(RUN/'manifest.json')['files'].items():assert sha(p)==h,p

def freeze():
    files={str(p.resolve()):sha(p) for p in FROZEN.rglob('*.py')}
    for base in [Path(__file__).parent,RUN/'data']:
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in str(p):files[str(p.resolve())]=sha(p)
    put(RUN/'manifest.json',{'files':files,'fits':96,'updates':451200,'full_outputs':800,'maximum_suffix_outputs':800,
        'seeds':SEEDS,'starts':STARTS,'python':sys.version,'torch':torch.__version__})

def cache_pair(ds,j,engineering=False):
    tic=time.time();g=graph(ds);prior=MarginalGraphPrior.from_training_graph(g);role=ExchangeableRoleDegreePrior.from_training_graph(g)
    terminal=FamilyCountCondition(prior,role.family_counts(prior.family_endpoints)).sample_graph(2026111700+2*j,mode='binomial')
    d=build_mixed_edit_diffusion_dataset(g,trajectories=2,forward_steps=1000,record_every=10,candidates_per_example=16,
        seed=2026111600+10000019*j,actions=('birth','death','relocation','switch'),feature_profile=PROFILE,role_corruption_targets={0:terminal})
    width=read(RUN/'data'/ds/'audit.json')['feature_width']
    assert d.candidate_features.shape==(200,16,width) and np.isfinite(d.candidate_features).all()
    assert np.isfinite(d.candidate_improvements).all() and np.array_equal(np.bincount(d.trajectory_ids),[100,100])
    p=RUN/('gate' if engineering else 'cache')/ds/f'pair{j}';p.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(p/'data.npz',x=d.candidate_features,y=d.targets,ids=d.trajectory_ids+2*j,u=d.candidate_improvements)
    put(p/'bridge_terminal.json',graph_payload(terminal));put(p/'report.json',{'pair':j,'rows':200,'seconds':time.time()-tic,
        'sha256':sha(p/'data.npz'),'terminal_seed':2026111700+2*j,'builder_seed':2026111600+10000019*j,'engineering':engineering})
    return d

class Forbidden(torch.nn.Module):
    def __init__(self,width):super().__init__();self.input_dim=width;self.calls=0
    def forward(self,*args,**kwargs):self.calls+=1;raise AssertionError('N called scorer')

def config(ds,arm,seed,steps=None,offset=0,suffix=False):
    a=read(RUN/'data'/ds/'audit.json')
    return dict(steps=steps if steps is not None else a['steps'],candidates_per_step=16,seed=seed,device='cuda:0',
        metrics_every=1000,role_weight=.05,gain_scale=10.,schedule_power=1.,schedule_floor=1.,
        role_error_per_node_threshold=5.5,model_time_horizon_steps=a['horizon'],model_time_start_step=offset,
        closure_candidate_fraction=1.,targeted_role_candidate_fraction=.5,
        auto_role_balance=.5 if arm!='N' and not suffix else 0.,metric_edge_chunk_size=128,
        fixed_role_steps=0 if suffix else None,use_model_score=arm!='N')

def invariant(final,initial,suffix=False):
    final.validate();assert np.array_equal(final.node_types,initial.node_types)
    assert final.family_specs.keys()==initial.family_specs.keys()
    for f in initial.family_specs:
        assert len(final.family_edges[f])==len(initial.family_edges[f])
        if suffix:assert np.array_equal(final.family_degrees(f),initial.family_degrees(f))

def gate(ds):
    tic=time.time();d=cache_pair(ds,0,True);out=RUN/'gate'/ds;g=graph(ds);width=d.candidate_features.shape[-1]
    cs,_=sample_edit_candidates(g,np.random.default_rng(20261116),16,actions=('relocation','switch'))
    target=ExchangeableRoleDegreePrior.from_training_graph(g).sample_assignment(g.node_types,9000)
    cpu=mixed_edit_candidate_matrix(g,cs,.5,PROFILE);gpu=gpu_features_for_proposals(g,cs,target,.5,'cuda:0',width).cpu().numpy()
    error=float(np.max(abs(cpu-gpu)));assert error<=1e-5,error
    models={};times={};updates={}
    for arm in ['M','F','W']:
        m=model(width,arm,777,'cuda:0');o=c.optimizer(m);x=torch.from_numpy(d.candidate_features[:64]).cuda();y=torch.from_numpy(d.targets[:64]).cuda()
        begin=time.time()
        for _ in range(12):c.update(m,o,x,y)
        torch.cuda.synchronize();updates[arm]=(time.time()-begin)/12
        c.save_torch(out/f'{arm}_resume.pt',{'m':copy.deepcopy(m.state_dict()),'o':copy.deepcopy(o.state_dict()),'rng':c.rng_state()})
        loss,_=c.update(m,o,x,y);expected=copy.deepcopy(m.state_dict());state=torch.load(out/f'{arm}_resume.pt',weights_only=False)
        m.load_state_dict(state['m']);o.load_state_dict(state['o']);c.set_rng(state['rng']);loss2,_=c.update(m,o,x,y)
        assert loss==loss2 and all(torch.equal(expected[k],v) for k,v in m.state_dict().items());models[arm]=m
    models['N']=Forbidden(width)
    initial=restore_graph(read(RUN/'data'/ds/'starts/seed100/initial.json'));target=np.array(read(RUN/'data'/ds/'starts/seed100/target.json'))
    for arm,m in models.items():
        cfg=config(ds,arm,9000,128);begin=time.time();z=observed_sample(initial,m,target,**cfg);torch.cuda.synchronize();times[arm]=time.time()-begin
        invariant(z.graph,initial);put(out/f'{arm}_sample.json',{'trace':z.trace,'graph':graph_payload(z.graph),'scientific':False,'seconds':times[arm]})
    # Original versus instrumented, including an actual callback branch at step32.
    cfg=config(ds,'M',9001,64);cfg['role_error_per_node_threshold']=float(np.sum((np.column_stack([initial.family_degrees(f) for f in sorted(initial.family_specs)])-target)**2)/initial.num_nodes)*.999
    captures=[];z=observed_sample(initial,models['M'],target,observer=lambda g,t,k:captures.append((g,t,k)),**cfg)
    zz=sample_gpu_role_guided_edit_diffusion(initial,models['M'],target,**cfg)
    assert graph_payload(z.graph)==graph_payload(zz.graph) and z.trace==zz.trace and z.action_counts==zz.action_counts
    # Guarantee callback coverage without a scientific threshold change: engineering initial-phase switch.
    if not captures:
        cfg['role_error_per_node_threshold']=1e12
        z=observed_sample(initial,models['M'],target,observer=lambda g,t,k:captures.append((g,t,k)),**cfg)
        zz=sample_gpu_role_guided_edit_diffusion(initial,models['M'],target,**cfg)
        assert graph_payload(z.graph)==graph_payload(zz.graph) and z.trace==zz.trace and z.action_counts==zz.action_counts
    assert len(captures)==1
    suffix=observed_sample(captures[0][0],models['M'],target,**config(ds,'M',9002,32,captures[0][2],True));invariant(suffix.graph,captures[0][0],True)
    assert models['N'].calls==0
    allocated=torch.cuda.max_memory_allocated();assert allocated<8*1024**3
    put(out/'acceptance.json',{'passed':True,'feature_max_error':error,'resume_exact':True,'observer_original_identical':True,
        'observer_capture_step':captures[0][2],'stage2_degree_preserved':True,'N_calls':0,'sampling128_seconds':times,
        'update_seconds':updates,'peak_gpu_bytes':allocated,'cache_pair_seconds':read(out/'pair0/report.json')['seconds'],'wall_seconds':time.time()-tic})

def merge(ds):
    arrays={k:[] for k in ['x','y','ids','u']}
    for j in range(16):
        p=RUN/'cache'/ds/f'pair{j}';assert sha(p/'data.npz')==read(p/'report.json')['sha256'];z=np.load(p/'data.npz')
        for k in arrays:arrays[k].append(z[k])
    arrays={k:np.concatenate(v) for k,v in arrays.items()};assert np.array_equal(np.bincount(arrays['ids']),np.full(32,100))
    p=RUN/'cache'/ds;np.savez_compressed(p/'data.npz',**arrays);put(p/'acceptance.json',{'passed':True,'sha256':sha(p/'data.npz'),'train_rows':3000,'validation_rows':200})

def fit(ds,arm,seed):
    verify();tic=time.time();out=RUN/'fits'/ds/f'seed{seed}'/arm;out.mkdir(parents=True,exist_ok=False)
    p=RUN/'cache'/ds;assert sha(p/'data.npz')==read(p/'acceptance.json')['sha256'];z=np.load(p/'data.npz');x,y,ids=z['x'],z['y'],z['ids'];val=ids>=30
    m=model(x.shape[-1],arm,seed,'cuda:0');o=c.optimizer(m);rng=torch.Generator().manual_seed(seed)
    loader=DataLoader(torch.from_numpy(np.flatnonzero(~val)),batch_size=64,shuffle=True,generator=rng);order=hashlib.sha256();history=[];updates=0
    for epoch in range(100):
        losses=[]
        for ix in loader:
            ix=ix.numpy();order.update(ix.tobytes());loss,_=c.update(m,o,torch.from_numpy(x[ix]).cuda(),torch.from_numpy(y[ix]).cuda());updates+=1;losses.append(loss)
        history.append(float(np.mean(losses)))
        if (epoch+1)%5==0:c.save_torch(out/'recovery.pt',{'state_dict':m.state_dict(),'optimizer':o.state_dict(),'rng':c.rng_state(),'loader_rng':rng.get_state(),'epoch':epoch+1,'updates':updates})
        put(out/'status.json',{'epoch':epoch+1,'updates':updates,'seconds':time.time()-tic})
    assert updates==4700
    metrics,logits,prob=c.eval_model(m,x[val],y[val],'cuda:0');regimes={str(i):c.eval_model(m,x[ids==i],y[ids==i],'cuda:0')[0] for i in [30,31]}
    np.savez_compressed(out/'validation.npz',logits=logits,probabilities=prob,targets=y[val],ids=ids[val])
    h,l=architecture(x.shape[-1],arm)
    c.save_torch(out/'model.pt',{'format':'dihug-mixed-edit-diffusion-v1','input_dim':x.shape[-1],'hidden_dim':h,'layers':l,'dropout':.1,
        'state_dict':m.cpu().state_dict(),'prior':MarginalGraphPrior.from_training_graph(graph(ds)).to_json_dict(),'feature_profile':PROFILE,
        'train_report':{'epochs':100,'updates':updates,'validation':metrics},'training_noise':{'recipe':'half_micro_half_independent_binomial_bridge'}})
    put(out/'report.json',{'completed':True,'dataset':ds,'arm':arm,'seed':seed,'updates':updates,'validation':metrics,'validation_regimes':regimes,
        'history':history,'batch_order_sha256':order.hexdigest(),'checkpoint_sha256':sha(out/'model.pt'),'cache_sha256':sha(p/'data.npz'),
        'wall_seconds':time.time()-tic,'peak_gpu_bytes':torch.cuda.max_memory_allocated()});verify()

def generate(ds,arm,seed,prefix,suffix=False):
    verify();tic=time.time();kind='suffix' if suffix else 'full';out=RUN/'outputs'/ds/kind/arm/f'train{seed}'/f'prefix{prefix}';out.mkdir(parents=True,exist_ok=False)
    a=read(RUN/'data'/ds/'audit.json');cp=RUN/'fits'/ds/f'seed{seed}'/arm/'model.pt'
    m=Forbidden(a['feature_width']) if arm=='N' else load_mixed_edit_checkpoint(cp,'cuda:0')[0]
    start=RUN/'data'/ds/'starts'/f'seed{prefix}';offset=0
    if suffix:
        start=RUN/'prefixes'/ds/f'seed{prefix}';offset=read(start/'binding.json')['transition_step']
    initial=restore_graph(read(start/'initial.json'));target=np.array(read(start/'target.json'))
    cfg=config(ds,arm,prefix+700000 if suffix else prefix,a['steps']-offset,offset,suffix)
    def observe(g,t,k):
        p=RUN/'prefixes'/ds/f'seed{prefix}';p.mkdir(parents=True,exist_ok=False)
        put(p/'initial.json',graph_payload(g));put(p/'target.json',t.tolist());put(p/'binding.json',{'producer_arm':arm,'producer_seed':seed,
            'transition_step':k,'remaining_steps':a['steps']-k,'initial_sha256':sha(p/'initial.json'),'target_sha256':sha(p/'target.json'),
            'checkpoint_sha256':sha(cp),'full_output_path':str(out)})
    z=observed_sample(initial,m,target,observer=observe if not suffix and arm=='M' and seed==810 else None,**cfg)
    invariant(z.graph,initial,suffix)
    if arm=='N':assert m.calls==0
    put(out/'binding.json',{'dataset':ds,'kind':kind,'arm':arm,'training_seed':None if arm=='N' else seed,'prefix_seed':prefix,
        'checkpoint':None if arm=='N' else str(cp),'checkpoint_sha256':None if arm=='N' else sha(cp),'start_path':str(start),
        'initial_sha256':sha(start/'initial.json'),'target_sha256':sha(start/'target.json'),'config':cfg,'N_calls':m.calls if arm=='N' else None})
    put(out/'graph.json',graph_payload(z.graph));put(out/'report.json',{'completed':True,'trace':z.trace,'action_counts':z.action_counts,
        'proposed':z.proposed,'applied':z.applied,'wall_seconds':time.time()-tic,'graph_sha256':sha(out/'graph.json'),'peak_gpu_bytes':torch.cuda.max_memory_allocated()});verify()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','gate','cache','merge','fit','generate']);p.add_argument('--dataset',choices=DATASETS)
    p.add_argument('--pair',type=int);p.add_argument('--arm',choices=['M','F','W','N']);p.add_argument('--seed',type=int);p.add_argument('--prefix',type=int);p.add_argument('--suffix',action='store_true');a=p.parse_args();setup()
    try:
        if a.stage=='freeze':freeze()
        elif a.stage=='gate':gate(a.dataset)
        elif a.stage=='cache':verify();cache_pair(a.dataset,a.pair);verify()
        elif a.stage=='merge':merge(a.dataset)
        elif a.stage=='fit':fit(a.dataset,a.arm,a.seed)
        else:generate(a.dataset,a.arm,a.seed,a.prefix,a.suffix)
    except BaseException as e:
        put(RUN/f'failure_{a.stage}_{a.dataset}_{a.arm}_{a.seed}_{a.prefix}_{a.pair}_{a.suffix}.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
