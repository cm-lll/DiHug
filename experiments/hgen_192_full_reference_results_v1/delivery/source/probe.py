"""Two bounded engineering jobs. No engineering state enters scientific fits."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,time,traceback,resource,copy,io
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from common import *
torch.set_num_threads(1)
def batch_bank(d):
 s=PROTOCOL['train_seeds'][d][0];z=np.load(str(OUT/'data'/d/('bank'+str(s)+'.npz')));n=192
 t=torch.nn.functional.one_hot(torch.from_numpy(z['types'][:640]).long(),5).float();x=torch.nn.functional.one_hot(torch.from_numpy(z['nodes'][:640]).long(),n+1).float()
 return [(t[i:i+32],x[i:i+32]) for i in range(0,640,32)]
def main(d):
 begin=time.time();check_time(900);torch.cuda.set_device(0);folder=OUT/'engineering'/d;folder.mkdir(parents=True,exist_ok=False);seed(2026103009)
 z=np.load(str(OUT/'data'/d/'reference.npz'));types=z['types'];mp=read(OUT/'data'/d/'hin2vec_mapping.json');examples=np.load(str(OUT/'data'/d/'hin2vec_examples.npz'))
 m=module(ROOT/'vendor/hin2vec/model.py','native_Hin2Vec');h=m.HIN2vec(192,mp['path_size'],32,True).cuda();opt=torch.optim.AdamW(h.parameters());ds=TensorDataset(torch.from_numpy(examples['x']),torch.from_numpy(examples['y']));dl=DataLoader(ds,batch_size=32,shuffle=True,num_workers=0)
 elapsed=[];tic=time.time()
 for i,(x,y) in enumerate(dl):
  if i==100:break
  start=time.time();x,y=x.cuda(),y.cuda();opt.zero_grad();p=h(x[:,0],x[:,1],x[:,2]);loss=torch.nn.functional.binary_cross_entropy(p,y);assert torch.isfinite(loss);loss.backward();opt.step();torch.cuda.synchronize();elapsed.append(time.time()-start)
 embedding=h.start_embeds.weight.detach().cpu().numpy()[[mp['node2id'][str(i)] for i in range(192)]]
 hin_seconds=time.time()-tic;assert np.isfinite(embedding).all();del h,opt;torch.cuda.empty_cache()
 env=load_core(True);g,dis,go,do,groups=setup(env,types,embedding,'cuda',1);env['dataloader']=batch_bank(d);calls=[];last=[time.time()];dur=[]
 def record(step,loss):
  torch.cuda.synchronize();dur.append(time.time()-last[0]);last[0]=time.time();assert torch.isfinite(loss);calls.append(step)
 env.update(record_update=record,print=lambda *a,**k:None);tic=time.time();env['train']();torch.cuda.synchronize();hgen_seconds=time.time()-tic
 assert calls==list(range(1,21)) and all(torch.isfinite(p).all() for x in [g,dis] for p in x.parameters())
 node_grad={name:p.grad is None for name,p in g.named_parameters() if name.startswith('lin_node_') and not name.startswith('lin_node_type')}
 # Preserve known original integer-sampling gradient limitation; never repair it silently.
 snap=copy.deepcopy(dict(g=g.state_dict(),d=dis.state_dict(),go=go.state_dict(),do=do.state_dict(),rng=rng()))
 saved=folder/'temporary_checkpoint.pt';torch.save(snap,str(saved));loaded=torch.load(str(saved),map_location='cpu')
 g2,d2,go2,do2,_=setup(load_core(),types,embedding,'cuda',1);g2.load_state_dict(loaded['g']);d2.load_state_dict(loaded['d']);go2.load_state_dict(loaded['go']);do2.load_state_dict(loaded['do']);restore_rng(loaded['rng'])
 def exact(a,b):
  if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
  if isinstance(a,np.ndarray):return np.array_equal(a,b)
  if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
  if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
  return a==b
 assert exact(g2.state_dict(),snap['g']) and exact(d2.state_dict(),snap['d']) and exact(go2.state_dict(),snap['go']) and exact(do2.state_dict(),snap['do']) and exact(rng(),snap['rng'])
 tic=time.time()
 with torch.no_grad():
  for _ in range(8):
   t,n=g(env['make_noise']((32,64),'Uniform').cuda());assert torch.isfinite(t).all() and torch.isfinite(n).all()
 torch.cuda.synchronize();sample_seconds=time.time()-tic
 allocated=torch.cuda.max_memory_allocated()/2**20;reserved=torch.cuda.max_memory_reserved()/2**20;assert allocated<9000
 save(folder/'receipt.json',dict(status='passed',dataset=d,device=torch.cuda.get_device_name(0),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),temporary_Hin2Vec_updates=100,temporary_D_updates=20,temporary_G_updates=4,temporary_walks=256,temporary_assembled_graphs=0,Hin2Vec100_updates_seconds=hin_seconds,Hin2Vec_p95_update_seconds=float(np.quantile(elapsed[5:],.95)),native20_D4_G_seconds=hgen_seconds,native_D_period_mean_seconds=sum(dur)/20,native_D_period_p95_seconds=float(np.quantile(dur[5:],.95)),eight_walk_batches_seconds=sample_seconds,exact_model_optimizer_rng_load=True,known_native_node_gradient_absence=node_grad,peak_allocated_mib=allocated,peak_reserved_mib=reserved,peak_RSS_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,seconds=time.time()-begin,engineering_states_excluded_from_science=True))
 print(read(folder/'receipt.json'))
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException:save(OUT/('probe_failure_'+sys.argv[1]+'.json'),dict(traceback=traceback.format_exc(),epoch=time.time()));raise
