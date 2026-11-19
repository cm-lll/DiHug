"""Paired fixed-cache original-MLP fits. No checkpoint/seed selection by performance."""
from r2_common import *
from torch.utils.data import DataLoader,TensorDataset
import argparse,traceback
def train(run,ds,arm,seed,device):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 verify_manifest(run);out=run/'fits'/ds/f'seed{seed}'/arm;out.mkdir(parents=True,exist_ok=False)
 start=time.time();cpu=time.process_time();x,y,ids,u=load_data(ds,arm);val=ids==ids.max();ii=np.flatnonzero(~val)
 m=model(ds,arm,seed,device);o=optimizer(m);gen=torch.Generator().manual_seed(seed)
 loader=DataLoader(TensorDataset(torch.from_numpy(x[ii]),torch.from_numpy(y[ii])),batch_size=64,shuffle=True,generator=gen)
 initial,_l,_p=eval_model(m,x[val],y[val],device)
 # Evaluation consumes no random numbers; keep paired initial states explicit.
 initial_path=out/'initial.pt';save_torch(initial_path,{'state_dict':m.state_dict(),'seed':seed,'schema':schema(ds)})
 histories=[];updates=0;order_hash=hashlib.sha256();epochs=ASSETS[ds]['epochs']
 # Store batch order explicitly via an index dataset; same generator and order as original.
 index_loader=DataLoader(torch.from_numpy(ii),batch_size=64,shuffle=True,generator=gen)
 for epoch in range(epochs):
  losses=[]
  for batch in index_loader:
   index=batch.numpy();order_hash.update(index.tobytes());bx=torch.from_numpy(x[index]).to(device);by=torch.from_numpy(y[index]).to(device)
   loss,norm=update(m,o,bx,by);losses.append(loss);updates+=1
  if arm!='M':assert torch.count_nonzero(m.network[0].weight[:,schema(ds)['zero_columns']])==0
  histories.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'updates':updates})
  if (epoch+1)%5==0 or epoch+1==epochs:
   save_torch(out/'recovery.pt',{'state_dict':m.state_dict(),'optimizer':o.state_dict(),'rng':rng_state(),'loader_rng':gen.get_state(),'epoch':epoch+1,'updates':updates,'data_sha256':sha(OLD/ASSETS[ds]['directory']/'mixed_edit_dataset.npz'),'arm':arm,'seed':seed})
  atomic(out/'status.json',{'stage':'training','epoch':epoch+1,'epochs':epochs,'updates':updates,'wall_seconds':time.time()-start})
 metrics,logits,probs=eval_model(m,x[val],y[val],device);np.savez_compressed(out/'validation.npz',indices=np.flatnonzero(val),targets=y[val],logits=logits,probabilities=probs)
 payload=torch.load(OLD/ASSETS[ds]['directory']/'mixed_edit_diffusion.pt',map_location='cpu',weights_only=False)
 payload.update(state_dict={k:v.cpu() for k,v in m.state_dict().items()},hidden_dim=ARMS[arm][0],layers=ARMS[arm][1],
  train_report={'epochs':epochs,'updates':updates,'training_objective':'inverse_ce','validation':metrics},
  revision_control={'arm':arm,'seed':seed,'zero_columns':schema(ds)['zero_columns'] if arm!='M' else [],'training_data_sha256':sha(OLD/ASSETS[ds]['directory']/'mixed_edit_dataset.npz'),'from_scratch':True})
 save_torch(out/'mixed_edit_diffusion.pt',payload);verify_manifest(run)
 report={'completed':True,'dataset':ds,'arm':arm,'seed':seed,'epochs':epochs,'updates':updates,'train_examples':int((~val).sum()),'validation_examples':int(val.sum()),'initial_validation':initial,'validation':metrics,'batch_order_sha256':order_hash.hexdigest(),'initial_sha256':sha(initial_path),'checkpoint_sha256':sha(out/'mixed_edit_diffusion.pt'),'history':histories,'wall_seconds':time.time()-start,'cpu_seconds':time.process_time()-cpu,'gpu_device':str(device),'max_allocated_bytes':torch.cuda.max_memory_allocated()}
 atomic(out/'report.json',report);atomic(out/'status.json',{'stage':'completed','updates':updates});print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',choices=ASSETS,required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 try:train(a.run,a.dataset,a.arm,a.seed,a.device)
 except BaseException as e:atomic(a.run/f'failure_train_{a.dataset}_{a.seed}_{a.arm}.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
