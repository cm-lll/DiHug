"""Conditional stage-2 only; common historical start and target, fresh learned weights."""
from r2_common import *
import argparse,traceback,gzip
from dihug.core.sampling_session import read_snapshot,restore_graph,graph_payload,counter_rng,digest,save_snapshot
from dihug.core.role_guided_edit_diffusion import ExchangeableRoleDegreePrior,sample_role_guided_edit_diffusion
from dihug.core.mixed_edit_diffusion import load_mixed_edit_checkpoint

class Journal:
 def __init__(self,directory,snapshot,ds,seed,limit=None):
  self.directory=directory;self.restore=copy.deepcopy(snapshot);self.target_rows=np.array(snapshot['target_rows']);self.ds=ds;self.seed=seed
  self.arm='M';self.diagnostics={};self.limit=limit;self.count=0;self.chain='';self.started=time.time();self.before=None
  self.log=gzip.open(directory/'actions.jsonl.gz','wt');self.first_graph=restore_graph(snapshot['graph'])
 def streams(self,rng,seed,attempt):return counter_rng(self.ds,seed,0,attempt,'proposal'),counter_rng(self.ds,seed,0,attempt,'selection','M')
 def before_attempt(self,graph,target_rows,v):self.diagnostics={};return False
 def after_attempt(self,graph,target_rows,v):
  p=v['selected'];k=v['completed'];self.count+=1
  rec={'attempt':k,'time':v['time_fraction'],'selected':{'action':p.action,'family':p.family,'old':p.old_edges,'new':p.new_edges},'selected_index':v['selected_index'],'candidate_count':len(v['candidates']),
   'candidates_sha256':digest([{'action':x.action,'family':x.family,'old':x.old_edges,'new':x.new_edges} for x in v['candidates']]),'role_error':v['current_role_error']}
  if self.count<=8 or self.count%256==0:
   rec['candidates']=[{'action':x.action,'family':x.family,'old':x.old_edges,'new':x.new_edges} for x in v['candidates']];rec['scores']=v['scores'].detach().cpu().tolist()
  self.chain=digest({'previous':self.chain,'record':rec});self.log.write(json.dumps({'chain':self.chain,'record':rec},separators=(',',':'))+'\n')
  if self.count%128==0 or k==10000:
   self.log.flush();atomic(self.directory/'status.json',{'stage':'generation','completed_absolute':k,'suffix_attempts':self.count,'wall_seconds':time.time()-self.started})
  return self.limit is not None and self.count>=self.limit
 def failure(self,e):atomic(self.directory/'sampler_failure.json',{'type':type(e).__name__,'message':str(e),'completed_suffix':self.count})

def generate(out,ds,prefix_seed,checkpoint,limit=None):
 torch.set_num_threads(1);start=time.time();cpu=time.process_time();out.mkdir(parents=True,exist_ok=False)
 manifest=json.loads((HISTORY/'manifest.json').read_text());ref=manifest['references'][ds]
 source=HISTORY/ds/f'seed{prefix_seed}/prefix/transition.json';snap=read_snapshot(source)
 model,prior,_=load_mixed_edit_checkpoint(checkpoint);role=ExchangeableRoleDegreePrior.load(OLD/ref['assets']['role'])
 config=dict(snap['config']);config.update(phase_policy='switch_only',selection_mode='model',fixed_role_steps=None,device='cpu',sampling_backend='cpu')
 assert config['steps']==10000 and config['model_time_start_step']==0
 runtime=Journal(out,snap,ds,prefix_seed,limit)
 atomic(out/'binding.json',{'dataset':ds,'prefix_seed':prefix_seed,'training_checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'source_transition':str(source),'source_transition_sha256':sha(source),'start_completed':snap['completed'],'target_hash':snap['target_hash'],'config':config,'stage2_only':True})
 try:result=sample_role_guided_edit_diffusion(prior,model,role,runtime=runtime,**config)
 finally:runtime.log.close()
 graph=result.graph;graph.validate()
 assert all(np.array_equal(graph.family_degrees(f),runtime.first_graph.family_degrees(f)) for f in graph.family_specs)
 atomic(out/'final_graph.json',graph_payload(graph));atomic(out/'report.json',{'completed':True,'suffix_attempts':runtime.count,'expected_suffix_attempts':min(limit,10000-snap['completed']) if limit else 10000-snap['completed'],'action_chain':runtime.chain,'same_per_node_relation_degree':True,'wall_seconds':time.time()-start,'cpu_seconds':time.process_time()-cpu,'new_graph':limit is None,'engineering_smoke':limit is not None,'final_sha256':sha(out/'final_graph.json')})
 return runtime

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',choices=ASSETS,required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--prefix-seed',type=int,required=True);a=p.parse_args()
 try:
  verify_manifest(a.run);generate(a.run/'generation'/a.dataset/f'train{a.seed}'/a.arm/f'prefix{a.prefix_seed}',a.dataset,a.prefix_seed,a.run/'fits'/a.dataset/f'seed{a.seed}'/a.arm/'mixed_edit_diffusion.pt');verify_manifest(a.run)
 except BaseException as e:atomic(a.run/f'failure_generate_{a.dataset}_{a.seed}_{a.arm}_{a.prefix_seed}.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
