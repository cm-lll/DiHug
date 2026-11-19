"""Bounded feature, learning, recovery and real-GPU cost gates, not quality screening."""
from r2_common import *
from dihug.core.mixed_edit_diffusion import mixed_edit_candidate_matrix,EditProposal,load_mixed_edit_checkpoint
from dihug.core.graph import TypedGraph,FamilySpec
from dihug.core.switch_diffusion import train_switch_denoiser
from torch.utils.data import DataLoader,TensorDataset
import argparse,traceback

def fixture():
 return TypedGraph(np.array([0,0,0,1,1,1]),('A','B'),
 {'AB':FamilySpec('AB',0,1,(0,1,2),(3,4,5))}, {'AB':[(0,3),(0,4),(1,4),(2,5)]})
def transformed(g,p,permutation):
 specs={f:FamilySpec(f,s.src_type,s.dst_type,tuple(permutation[x] for x in s.src_nodes),tuple(permutation[x] for x in s.dst_nodes)) for f,s in g.family_specs.items()}
 q=TypedGraph(g.node_types.copy(),g.type_names,specs,{f:[(permutation[u],permutation[v]) for u,v in es] for f,es in g.family_edges.items()})
 ps=[EditProposal(a.action,a.family,tuple(q.orient_edge(a.family,(permutation[u],permutation[v])) for u,v in a.old_edges),tuple(q.orient_edge(a.family,(permutation[u],permutation[v])) for u,v in a.new_edges)) for a in p]
 return q,ps
def run(out,device):
 torch.set_num_threads(1);tic=time.time();records={};g=fixture()
 ps=[EditProposal('noop','AB',(),()),EditProposal('relocation','AB',((0,3),),((1,3),)),EditProposal('switch','AB',((0,3),(2,5)),((0,5),(2,3)))]
 x=mixed_edit_candidate_matrix(g,ps,.3,'typed_closure_compact_v2');q,qs=transformed(g,ps,{0:2,1:0,2:1,3:4,4:5,5:3});xx=mixed_edit_candidate_matrix(q,qs,.3,'typed_closure_compact_v2')
 assert np.array_equal(x,xx);records['cross_type_relabel_features_exact']=True
 # Same-type canonical endpoint slots are inherited and NOT generally anonymous.
 h=TypedGraph(np.zeros(5,dtype=int),('A',),{'AA':FamilySpec('AA',0,0,tuple(range(5)),tuple(range(5)))},{'AA':[(0,1),(0,2),(0,3),(3,4)]})
 a=[EditProposal('death','AA',((0,1),),())];hh,aa=transformed(h,a,{0:1,1:0,2:2,3:3,4:4})
 v=mixed_edit_candidate_matrix(h,a,.3,'typed_closure_compact_v2');vv=mixed_edit_candidate_matrix(hh,aa,.3,'typed_closure_compact_v2')
 records['inherited_same_type_endpoint_ordering']={'equal':bool(np.array_equal(v,vv)),'max_feature_difference':float(np.abs(v-vv).max()),'scope':'Original fixed-ID scorer convention retained in every arm; NOT a fully equivariant generator. No new structural fix introduced.'}
 assert not np.array_equal(v,vv)
 for ds in ASSETS:
  w=torch.load(OLD/ASSETS[ds]['directory']/'mixed_edit_diffusion.pt',map_location='cpu',weights_only=False)
  assert (w['hidden_dim'],w['layers'],w['dropout'],w['input_dim'])==(256,3,.1,schema(ds)['width'])
  original,_,_=load_mixed_edit_checkpoint(OLD/ASSETS[ds]['directory']/'mixed_edit_diffusion.pt');manual=SwitchDenoiserMLP(w['input_dim'],256,3,.1);manual.load_state_dict(w['state_dict']);original.eval();manual.eval()
  raw,y,ids,u=load_data(ds,'M');xf,_,_,_=load_data(ds,'D');val=ids==ids.max();train=~val
  assert np.array_equal(raw[...,schema(ds)['keep_columns']],xf[...,schema(ds)['keep_columns']]) and not np.any(xf[...,schema(ds)['zero_columns']])
  with torch.no_grad():assert torch.equal(original(torch.from_numpy(raw[:8])),manual(torch.from_numpy(raw[:8])))
  # Independent degree-only features on the real reference, one stored legal candidate list.
  from dihug.core.sampling_session import read_snapshot,restore_graph
  from dihug.core.role_guided_edit_diffusion import sample_role_alignment_candidates
  snap=read_snapshot(HISTORY/ds/'seed100/prefix/transition.json');graph=restore_graph(snap['graph']);families=sorted(graph.family_specs)
  cs=sample_role_alignment_candidates(graph,np.array(snap['target_rows']),families,np.random.default_rng(9001),16,targeted_fraction=.5)
  actual=mixed_edit_candidate_matrix(graph,cs,.4,'typed_closure_compact_v2');expected=np.zeros_like(actual)
  degree=np.zeros(graph.num_nodes,dtype=int);fd={f:np.zeros(graph.num_nodes,dtype=int) for f in families}
  for f,es in graph.family_edges.items():
   for i,j in es:degree[i]+=1;degree[j]+=1;fd[f][i]+=1;fd[f][j]+=1
  base=len(SCORER_ACTIONS)+len(families);scale=max(1.,np.log1p(graph.num_nodes))
  for i,c in enumerate(cs):
   expected[i,list(SCORER_ACTIONS).index(c.action)]=1
   if c.action!='noop':expected[i,len(SCORER_ACTIONS)+families.index(c.family)]=1
   slots=list(c.old_edges)+[None]*(2-len(c.old_edges))+list(c.new_edges)+[None]*(2-len(c.new_edges))
   for j,e in enumerate(slots):
    if e is not None:
     n,k=e;expected[i,base+4*j:base+4*j+4]=np.log1p([degree[n],degree[k],fd[c.family][n],fd[c.family][k]])/scale
   sp=graph.family_specs[c.family];domain=len(sp.src_nodes)*len(sp.dst_nodes) if sp.bipartite else len(sp.src_nodes)*(len(sp.src_nodes)-1)//2
   expected[i,base+20:base+24]=[.4,0. if c.action=='noop' else len(graph.family_edges[c.family])/max(1,domain),len(c.old_edges),len(c.new_edges)]
  actual[:,schema(ds)['zero_columns']]=0;assert np.max(np.abs(expected-actual))<1e-7
  # Candidate row equivariance and excluded-feature immunity for both degree arms.
  checks={}
  for arm in ARMS:
   m=model(ds,arm,777);o=optimizer(m);data=raw if arm=='M' else xf
   b=torch.from_numpy(data[train][:64]);t=torch.from_numpy(y[train][:64]);m.eval()
   with torch.no_grad():
    z=m(b);assert torch.max(torch.abs(m(b.flip(1))-z.flip(1)))<1e-6
    if arm!='M':
     corrupt=b.clone();corrupt[...,schema(ds)['zero_columns']]=1000
     assert torch.equal(z,m(corrupt))
   before=eval_model(m,data[train][:64],y[train][:64],'cpu')[0]['ce'];step_losses=[]
   for _ in range(16):step_losses.append(update(m,o,b,t)[0])
   after=eval_model(m,data[train][:64],y[train][:64],'cpu')[0]['ce'];assert after<before
   state=copy.deepcopy(m.state_dict());opt=copy.deepcopy(o.state_dict());rng=rng_state()
   loss1,_=update(m,o,b,t);want=copy.deepcopy(m.state_dict());m.load_state_dict(state);o.load_state_dict(opt);set_rng(rng);loss2,_=update(m,o,b,t)
   assert loss1==loss2 and all(torch.equal(want[k],v) for k,v in m.state_dict().items())
   if arm!='M':assert torch.count_nonzero(m.network[0].weight[:,schema(ds)['zero_columns']])==0
   checks[arm]={'before_ce':before,'after_16_ce':after,'update_replay_exact':True,'parameters':sum(p.numel() for p in m.parameters())}
  # Retained features must provide actual listwise information; collisions reported, not discarded.
  def support(mask):
   rows=xf[mask];return {'rows':len(rows),'nonconstant_lists':int(np.any(rows!=rows[:,0:1],axis=(1,2)).sum()),'unique_target_rows':int(np.sum(np.sum(np.all(rows==rows[np.arange(len(rows)),y[mask]][:,None,:],axis=2),axis=1)==1))}
  # One epoch of our loop must match original training implementation on a fixed slice.
  take=np.r_[np.flatnonzero(train)[:128],np.flatnonzero(val)[:64]];toy=SwitchDiffusionDataset(raw[take],u[take],y[take],ids[take])
  ref,rep=train_switch_denoiser(toy,hidden_dim=64,layers=1,epochs=1,batch_size=64,seed=998,device='cpu')
  m=SwitchDenoiserMLP(schema(ds)['width'],64,1,.1);torch.manual_seed(998);m=SwitchDenoiserMLP(schema(ds)['width'],64,1,.1);o=optimizer(m)
  loader=DataLoader(TensorDataset(torch.from_numpy(raw[take][:128]),torch.from_numpy(y[take][:128])),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(998))
  for bx,by in loader:update(m,o,bx,by)
  assert all(torch.equal(v,ref.state_dict()[k]) for k,v in m.state_dict().items())
  # Real-batch GPU smoke, fixed updates and recovery. No formal weights retained.
  gpu={}
  for arm in ARMS:
   m=model(ds,arm,999,device);o=optimizer(m);data=raw if arm=='M' else xf;b=torch.from_numpy(data[train][:64]).to(device);t=torch.from_numpy(y[train][:64]).to(device)
   for _ in range(4):update(m,o,b,t)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
   for _ in range(16):
    start=time.perf_counter();update(m,o,b,t);torch.cuda.synchronize();times.append(time.perf_counter()-start)
   gpu[arm]={'median_update_seconds':float(np.median(times)),'max_allocated_bytes':torch.cuda.max_memory_allocated(),'updates':20}
   del m,o,b,t;torch.cuda.empty_cache()
  records[ds]={'schema':schema(ds),'train_support':support(train),'validation_support':support(val),'checks':checks,'gpu':gpu,'reference_loop_exact':True,'independent_degree_feature_error':float(np.abs(actual-expected).max())}
 records.update(passed=True,wall_seconds=time.time()-tic,scope='wiring, finite learning and timing only; same-type canonical slot noninvariance inherited and disclosed',formal_training_started=False)
 atomic(out/'preflight.json',records);print(json.dumps(records),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 try:run(a.out,a.device)
 except BaseException as e:atomic(a.out/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
