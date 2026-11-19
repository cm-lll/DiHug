"""Exercise the actual fit/export entry with one epoch per arm, then verify saved outputs."""
from r2_train import *
from scipy.special import logsumexp
def gate(out):
 out.mkdir(parents=True,exist_ok=False);paths=list((ROOT/'scripts').glob('r2_*.py'))+list((ROOT/'src/dihug').rglob('*.py'))
 atomic(out/'manifest.json',{'files':{str(p):sha(p) for p in paths}});records=[]
 for ds in ASSETS:
  ASSETS[ds]['epochs']=1
  for arm in ARMS:
   train(out,ds,arm,997,'cuda:0');p=out/'fits'/ds/'seed997'/arm;z=np.load(p/'validation.npz');r=json.loads((p/'report.json').read_text())
   logits=z['logits'].astype(float);ce=float((logsumexp(logits,axis=1)-logits[np.arange(len(logits)),z['targets']]).mean());assert abs(ce-r['validation']['ce'])<1e-6
   from dihug.core.mixed_edit_diffusion import load_mixed_edit_checkpoint
   m,_,_=load_mixed_edit_checkpoint(p/'mixed_edit_diffusion.pt','cuda:0');x,y,ids,_=load_data(ds,arm);val=ids==ids.max();_,_,prob=eval_model(m,x[val],y[val],'cuda:0');assert np.array_equal(prob,z['probabilities'])
   recovery=torch.load(p/'recovery.pt',map_location='cuda:0',weights_only=False);m.load_state_dict(recovery['state_dict']);o=optimizer(m);o.load_state_dict(recovery['optimizer'])
   # Exercise GPU saved model/Adam and RNG state, not just a weights-only load.
   b=torch.from_numpy(x[~val][:64]).to('cuda:0');t=torch.from_numpy(y[~val][:64]).to('cuda:0');state=copy.deepcopy(m.state_dict());opt=copy.deepcopy(o.state_dict());rng=rng_state()
   a=update(m,o,b,t);want=copy.deepcopy(m.state_dict());m.load_state_dict(state);o.load_state_dict(opt);set_rng(rng);bb=update(m,o,b,t)
   assert a==bb and all(torch.equal(want[k],v) for k,v in m.state_dict().items())
   records.append({'dataset':ds,'arm':arm,'updates':r['updates'],'export_probability_exact':True,'gpu_optimizer_rng_replay_exact':True,'wall_seconds':r['wall_seconds']})
 atomic(out/'gate.json',{'passed':True,'records':records,'scientific_outputs':0,'scope':'one epoch entry/export/optimizer replay smoke, no quality selection'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 try:gate(a.out)
 except BaseException as e:atomic(a.out/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
