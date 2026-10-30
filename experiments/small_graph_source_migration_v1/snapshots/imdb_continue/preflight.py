"""CPU parent/cost/queue verification, no model forward or optimizer step."""
import sys,statistics,time
from common import *
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(SOURCE/'package'))
import torch,numpy as np
from prepare import load_case
from normal2 import pair_inputs
from probe import input_hash
from runtime_contract import source_binding as scientific_binding
def main():
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'tmp').mkdir();case,cfg=load_case('IMDB');parents={};hashes={};costs=[]
 assert scientific_binding()[1]==SOURCE_HASH
 for job in train_jobs():
  d=PARENT/job['id'];p=d/'checkpoint_u3000.pt';h=sha(p);side=read(p.with_suffix('.sha256.json'));assert side==dict(sha256=h,updates=3000)
  assert read(d/'summary.json')['status']=='complete' and read(d/'sampling_isolation_u3000.json')['passed']
  x=torch.load(p,map_location='cpu');b=x['binding']
  assert x['updates']==x['ema']['updates']==3000 and x['ema']['decay']==.99 and x['optimizer']['state']
  assert b['source_sha256']==SOURCE_HASH and (b['dataset'],b['train_seed'],b['arm'])==('IMDB',job['train_seed'],job['arm'])
  assert b['config']==dict(cfg,seed=job['train_seed']) and b['lambda_count']==(0 if job['arm']=='CE' else .001)
  assert set(x['rng'])=={'numpy','global_numpy','python','torch','device'}
  assert all(torch.isfinite(t).all() for t in x['model'].values() if t.is_floating_point()) and all(torch.isfinite(t).all() for t in x['ema']['shadow'].values())
  rng=np.random.default_rng();rng.bit_generator.state=x['rng']['numpy'];inputs=pair_inputs(case,dict(cfg,seed=job['train_seed']),3001,rng)
  hashes[job['id']]=input_hash(inputs);parents[job['id']]=dict(path=str(p),sha256=h,sidecar_sha256=sha(p.with_suffix('.sha256.json')),binding=b)
  rows=[readline for readline in (d/'training.jsonl').read_text().splitlines()];times=[__import__('json').loads(s)['total_update_seconds'] for s in rows[-500:]]
  chain=[read(f)['seconds'] for f in sorted(d.glob('u3000/ema_0.99/*/binding.json'))]
  costs.append(dict(id=job['id'],update_seconds_mean=statistics.mean(times),update_seconds_p95=sorted(times)[475],chain_seconds=chain))
  del x
 for s in [2026102911,2026102931]:assert hashes[identity(s,'CE')]==hashes[identity(s,'CE_COUNT')]
 compute=max(c['update_seconds_p95']*3000+sum(c['chain_seconds']) for c in costs)
 planned=(compute+180)*1.25
 if time.time()+planned>SCIENCE:raise RuntimeError('remaining budget cannot contain fixed4 tasks8 chains plus reserve')
 save(OUT/'preflight.json',dict(status='passed',parents=parents,next_input3001_hashes=hashes,source_sha256=source_binding()[1],costs=costs,
  planned_remaining_science_seconds=planned,planned_science_finish_epoch=time.time()+planned,planned_delivery_epoch=time.time()+planned+1800,
  scientific_parent_source_sha256=SOURCE_HASH,temporary_optimizer_steps=0,model_forwards=0,training_jobs=train_jobs(),sampling_jobs=jobs()))
 print(json.dumps(dict(status='passed',parents=4,new_updates=12000,chains=8,planned_science_minutes=planned/60,latest_allowed_science=SCIENCE)))
if __name__=='__main__':main()
