"""Run literal frozen author GAN once; final generated walks feed two assemblies."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,time,traceback,contextlib,types,resource
import numpy as np,scipy.sparse as sp,torch
from torch.utils.data import Dataset,DataLoader
from common import *
torch.set_num_threads(1)
class Bank(Dataset):
 def __init__(self,z):self.nodes=z['nodes'];self.types=z['types']
 def __len__(self):return len(self.nodes)
 def __getitem__(self,i):return (torch.nn.functional.one_hot(torch.from_numpy(self.types[i]).long(),5).float(),torch.nn.functional.one_hot(torch.from_numpy(self.nodes[i]).long(),193).float())
def compatible_assembly(score,target,patterns,groups,s):
 tree=ast.parse((ROOT/'vendor/old_safe_assembly.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sample')
 mod=ast.Module(body=[node])
 if hasattr(ast,'TypeIgnore'):mod.type_ignores=[]
 env=dict(np=np,sp=sp,time=time,c=types.SimpleNamespace(seed=seed,load=load_core))
 exec(compile(mod,str(ROOT/'vendor/old_safe_assembly.py'),'exec'),env)
 return env['sample'](score,target,patterns,groups,s)
def scores(env,tt,nn):
 # Exactly original defined-domain tail handling, totalized only for all-EOS.
 def tail(rows,eos):
  out=[]
  for row in rows:
   end=len(row)
   while end and row[end-1]==eos:end-=1
   out.append(row[:end].copy())
  return out
 ts=tail(tt,4);ns=tail(nn,192)
 valid_t=[[x for x in ts if len(x)==k and 4 not in x] for k in (2,3,4)]
 valid_n=[[x for x in ns if len(x)==k and 192 not in x] for k in (2,3,4)]
 score=sp.csr_matrix((192,192))
 for rows in valid_n:
  if rows:score=score+env['score_matrix_from_random_walks'](rows,192)
 # Native top-half per-length frequency rule, no schema or quality filtering.
 p=env['meta_path_frequency'](*valid_t)
 return score.tocsr(),p,dict(total=len(nn),all_EOS=sum(len(x)==0 for x in ns),usable_lengths=[len(x) for x in valid_n])
def main(d,s):
 start=time.time();folder=OUT/'run'/cell(d,s);folder.mkdir(parents=True,exist_ok=False);check_time(600)
 expected=read(OUT/'source_manifest.json')
 if source_manifest()!=expected:raise ValueError('source changed')
 z=np.load(str(OUT/'data'/d/'reference.npz'));typ=z['types'];schema=z['schema'];ref_edges=z['edges'];embedding=np.load(str(OUT/'data'/d/'embedding.npy'));asset=read(OUT/'data'/d/'embedding_receipt.json')
 assert sha(OUT/'data'/d/'embedding.npy')==asset['sha256'];bankpath=OUT/'data'/d/('bank'+str(s)+'.npz');bank=np.load(str(bankpath));assert len(bank['nodes'])==24000
 seed(s);env=load_core(True);g,dis,go,do,groups=setup(env,typ,embedding,'cuda',5);env['dataloader']=DataLoader(Bank(bank),batch_size=32,shuffle=True,num_workers=0,drop_last=True)
 def snapshot(step):return dict(g=g.state_dict(),d=dis.state_dict(),go=go.state_dict(),do=do.state_dict(),rng=rng(),dataset=d,train_seed=s,D_updates=step,G_updates=step//5,protocol_sha256=sha(ROOT/'protocol.json'),embedding_sha256=asset['sha256'],bank_sha256=sha(bankpath))
 torch.save(snapshot(0),str(folder/'initial.pt'));parameter_counts=dict(G=sum(x.numel() for x in g.parameters()),D=sum(x.numel() for x in dis.parameters()))
 last=[0];log=(folder/'training.jsonl').open('w')
 def record(step,loss):
  last[0]=step;check_time()
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite GAN loss')
  if step%25==0:
   log.write(json.dumps(dict(D_updates=step,G_updates=(step+4)//5,loss_D=float(loss),seconds=time.time()-start))+'\n');log.flush();save(folder/'progress.json',dict(stage='training',D_updates=step,G_updates=(step+4)//5,seconds=time.time()-start))
 env.update(record_update=record,print=lambda *a,**kw:None)
 tic=time.time()
 try:env['train']()
 except BaseException:
  torch.save(snapshot(last[0]),str(folder/'interrupted_state.pt'));log.close();raise
 torch.cuda.synchronize();train_seconds=time.time()-tic;log.close();assert last[0]==3750
 assert all(torch.isfinite(p).all() for model in (g,dis) for p in model.parameters())
 final=folder/'final.pt';torch.save(snapshot(3750),str(final));save(folder/'training_complete.json',dict(D_updates=3750,G_updates=750,seconds=train_seconds,weights_sha256=sha(final),parameter_counts=parameter_counts,known_native_node_head_no_gradient={n:p.grad is None for n,p in g.named_parameters() if n.startswith('lin_node_') and not n.startswith('lin_node_type')}))
 # Independent fixed generation stream; does not alter finished training state.
 seed((s+1000003)%(2**32));tt=[];nn=[];tic=time.time();g.eval()
 with torch.no_grad():
  for i in range(3000):
   check_time();t,n=g(env['make_noise']((32,64),'Uniform').cuda())
   if not torch.isfinite(t).all() or not torch.isfinite(n).all():raise FloatingPointError('nonfinite native model sampling')
   tt.append(t.argmax(2).cpu().numpy());nn.append(n.argmax(2).cpu().numpy())
   if i%100==0:save(folder/'progress.json',dict(stage='model_walks',D_updates=3750,batches=i+1,planned=3000))
 tt=np.concatenate(tt);nn=np.concatenate(nn);np.savez_compressed(str(folder/'generated_walks.npz'),types=tt,nodes=nn);sample_seconds=time.time()-tic
 score,patterns,info=scores(env,tt,nn);sp.save_npz(str(folder/'scores.npz'),score);save(folder/'patterns.json',[dict(types=list(map(int,k)),count=int(v)) for k,v in patterns.items()]);save(folder/'walk_receipt.json',dict(**info,seconds=sample_seconds,sha256=sha(folder/'generated_walks.npz')))
 peak=torch.cuda.max_memory_allocated()/2**20;del g,dis,go,do;torch.cuda.empty_cache()
 requests=[]
 for sample_seed in PROTOCOL['sample_seeds']:
  check_time();dest=folder/('sample_'+str(sample_seed));dest.mkdir();tic=time.time()
  adj,ledger,complete,reason=compatible_assembly(score,len(ref_edges)*2,patterns,groups,sample_seed)
  assert adj.shape==(192,192) and np.isfinite(adj).all() and np.array_equal(adj,adj.T)
  np.savez_compressed(str(dest/'raw_adjacency.npz'),adjacency=adj);loops=np.flatnonzero(np.diag(adj)).tolist();es=np.stack(np.triu(adj,1).nonzero(),axis=0).astype(np.int64)
  lookup={tuple(p):i for i,p in enumerate(schema)};relations=np.array([lookup[tuple(sorted((int(typ[u]),int(typ[v]))))]+1 for u,v in es.T],np.int64)
  # This export is exact when simple. Raw loops are retained above and disqualify
  # primary simple-graph metric evaluation; never silently call it a clean graph.
  np.savez_compressed(str(dest/'graph.npz'),types=typ,edges=es,schema=schema,relations=relations)
  with (dest/'assembly_actions.jsonl').open('w') as f:
   for row in ledger:f.write(json.dumps(row)+'\n')
  r=dict(dataset=d,train_seed=s,sample_seed=sample_seed,readout='final_G_after3750D_750G',complete=bool(complete),reason=reason,self_loops=loops,simple=len(loops)==0,edges=int(len(es.T)),raw_directed_edges=int(adj.sum()),target_undirected_edges=len(ref_edges),graph_sha256=sha(dest/'graph.npz'),raw_sha256=sha(dest/'raw_adjacency.npz'),weight_sha256=sha(final),reference_sha256=sha(OUT/'data'/d/'reference.npz'),embedding_sha256=asset['sha256'],seconds=time.time()-tic,variant=PROTOCOL['name'],assembly_proposals=len(ledger),zero_support_rejections=sum(x['status']!='accepted' for x in ledger),no_backbone=True)
  save(dest/'binding.json',r);requests.append(r)
 save(folder/'summary.json',dict(status='complete' if all(x['complete'] and x['simple'] for x in requests) else 'completed_with_failures',dataset=d,train_seed=s,D_updates=3750,G_updates=750,requests=requests,seconds=time.time()-start,peak_allocated_mib=peak,peak_RSS_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,finished_epoch=time.time()))
 print(cell(d,s),'finished',time.time()-start)
if __name__=='__main__':
 try:main(sys.argv[1],int(sys.argv[2]))
 except BaseException:save(OUT/('failure_'+cell(sys.argv[1],int(sys.argv[2]))+'.json'),dict(traceback=traceback.format_exc(),epoch=time.time()));raise
