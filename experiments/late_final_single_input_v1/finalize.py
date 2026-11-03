"""CPU evidence/plots and small local archive, with large checkpoint SHA index."""
import datetime,hashlib,json,os,pathlib,shutil,subprocess,time,traceback,zipfile
from common_run import ROOT,HERE,SCI,OUT,read,save,sha,now
def main():
 ledger=read(OUT/'ledger.json');tasks=read(OUT/'tasks.json');pairing=[];failures=[]
 for ds in ['IMDB','PubMed']:
  for seed in sorted({j['train_seed'] for j in tasks if j['dataset']==ds}):
   paths={a:OUT/'run'/f'{ds}_{seed}_{a}_late'/'training.jsonl' for a in ['A','E']};logs={a:[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else [] for a,p in paths.items()};n=min(map(len,logs.values()))
   inp=all(logs['A'][i]['input_sha256']==logs['E'][i]['input_sha256'] for i in range(n));streams=all(logs['A'][i]['training_streams_before']==logs['E'][i]['training_streams_before'] and logs['A'][i]['training_streams_after']==logs['E'][i]['training_streams_after'] for i in range(n))
   pairing.append(dict(dataset=ds,train_seed=seed,compared_updates=n,expected_updates=3000,input_pairing=inp,all_training_streams_pairing=streams,complete_both=n==3000))
   if not inp or not streams:failures.append(dict(dataset=ds,train_seed=seed,reason='input/global training RNG pairing mismatch'))
 save(OUT/'final_pairing_audit.json',dict(pairs=pairing,failures=failures))
 import csv
 valrows=[];relations=[]
 for j in tasks:
  p=OUT/'run'/j['task_id']/'validation.json'
  if not p.exists():continue
  for node in read(p):
   for kind in ['raw','ema_0_99']:
    for r in node[kind]['rows']:
     identity=dict(dataset=j['dataset'],train_seed=j['train_seed'],arm=j['arm'],updates=node['updates'],readout=kind,t=r['t'],input_sha256=r['input_sha256'])
     valrows.append(dict(identity,CE_sum_over_S=r['ce_sum_over_S']))
     for f in r['family']:relations.append(dict(identity,**f,p0_minus_target=f['expected_edges']-f['true_edges']))
 def writecsv(name,rows):
  with (OUT/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
 writecsv('validation_total.csv',valrows);writecsv('validation_relations.csv',relations)
 # Plots are presentation only. A plot import failure cannot replace scientific failure reasons.
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  rows=read(OUT/'evaluation/decision_table.json');metrics=[('absolute_error','edges'),('distance','main_incident4_W1'),('distance','main_joint_degree_TV'),('distance','triangle_composition_TV')]
  fig,axes=plt.subplots(2,4,figsize=(14,6))
  for di,ds in enumerate(['IMDB','PubMed']):
   seeds=sorted({j['train_seed'] for j in tasks if j['dataset']==ds})
   for mi,(kind,metric) in enumerate(metrics):
    ax=axes[di,mi]
    for ai,arm in enumerate(['A','E']):
     vv=[next((r['value'] for r in rows if r['dataset']==ds and r['train_seed']==s and r['arm']==arm and r['kind']==kind and r['metric']==metric),None) for s in seeds]
     if all(x is not None for x in vv):ax.plot(range(2),vv,'o-',label=arm+' late')
    ax.set_xticks(range(2),[str(s) for s in seeds],rotation=20);ax.set_title(ds+' '+metric);ax.legend()
  fig.tight_layout();fig.savefig(OUT/'quality_per_training_seed.png',dpi=160);plt.close(fig)
  fig,axes=plt.subplots(2,2,figsize=(10,6))
  for di,ds in enumerate(['IMDB','PubMed']):
   seeds=sorted({j['train_seed'] for j in tasks if j['dataset']==ds})
   for si,seed in enumerate(seeds):
    ax=axes[di,si]
    for arm in ['A','E']:
     p=OUT/'run'/f'{ds}_{seed}_{arm}_late'/'training.jsonl';rr=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
     if rr:
      x=[r['update'] for r in rr];y=[r['loss'] for r in rr];ax.plot(x,y,alpha=.12)
      smooth=[sum(y[max(0,i-49):i+1])/len(y[max(0,i-49):i+1]) for i in range(len(y))];ax.plot(x,smooth,label=arm+' CE trailing50')
    ax.set_title(f'{ds} {seed}');ax.legend()
  fig.tight_layout();fig.savefig(OUT/'training_CE_raw_and_trailing50.png',dpi=160);plt.close(fig)
 except BaseException as e:save(OUT/'plot_failure.json',dict(at=now(),error=repr(e)))
 historical=ROOT/'results/frozen_late_method_audit_20261102T121837'
 save(OUT/'historical_panels_SEPARATE.json',dict(historical_factorial=dict(directory=str(historical),graphs=32,setting='1000normal2/raw/R1',not_new_matched_evidence=True),external=dict(directory=str(ROOT/'results/hgen_192_full_reference_20261030T091112'),graphs=16,not_identical_support_or_training_budget=True)))
 # Checkpoint/image artifacts remain local; small review ZIP excludes private tensors.
 index={}
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name not in ['SHA256_INDEX.json'] and '.tmp' not in p.name and not p.name.endswith('.log') and p.name not in ['watchdog_state.json','launcher_state.json']:
   if time.time()>=ledger['deadline_epoch']-45:raise TimeoutError('hash/pack final reserve')
   index[str(p.relative_to(OUT))]=dict(sha256=sha(p),bytes=p.stat().st_size)
 save(OUT/'SHA256_INDEX.json',index)
 leases=[read(p) for p in (OUT/'leases').glob('*.json')];hours=sum((x.get('released_epoch',time.time())-x['acquired_epoch'])/3600 for x in leases)
 receipt=dict(at=now(),elapsed_minutes=(time.time()-ledger['start_epoch'])/60,GPU_hours=hours,within_wall=time.time()<=ledger['deadline_epoch'],within_GPU_hours=hours<=25,source_manifest_sha256=sha(OUT/'source_manifest.json'),pairing=pairing,pairing_failures=failures,completion=read(OUT/'completion.json'),temporary_optimizer_steps=6,no_additional_candidates=True,no_multigraph_experiments=True)
 save(OUT/'FINAL_RECEIPT.json',receipt)
 review=ROOT/'research_branches/late_final_single_input_release_20261102'
 if review.exists():raise FileExistsError('release already exists')
 (review/'experiments/late_final_single_input_v1').mkdir(parents=True)
 for p in HERE.glob('*.py'):shutil.copy2(p,review/'experiments/late_final_single_input_v1'/p.name)
 dest=review/'results'/OUT.name;dest.mkdir(parents=True)
 for p in OUT.rglob('*'):
  if p.is_file() and p.suffix not in ['.pt','.npz'] and p.stat().st_size<10*1024**2 and 'temporary' not in p.parts and 'engineering' not in p.parts and 'prepared' not in p.parts:
   d=dest/p.relative_to(OUT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
 for cmd in [['git','init','--quiet'],['git','symbolic-ref','HEAD','refs/heads/codex/late-final-single-input-results'],['git','add','experiments','results'],['git','-c','user.name=Codex','-c','user.email=codex@local','commit','--quiet','-m','Complete frozen A/E late-CN single-input matched panel']]:subprocess.run(cmd,cwd=review,check=True)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=review,text=True).strip();save(OUT/'release_version.json',dict(commit=commit,directory=str(review),external_upload=False))
 archive=OUT.parent/(OUT.name+'_review.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in review.rglob('*'):
   if p.is_file() and '.git' not in p.parts:z.write(p,str(p.relative_to(review)))
  z.write(OUT/'release_version.json','release_version.json')
 with zipfile.ZipFile(archive) as z:
  if z.testzip() is not None:raise ValueError('ZIP CRC')
 save(OUT/'archive_receipt.json',dict(file=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,large_weights_and_original_graphs_remain_at=str(OUT/'run')))
 print(json.dumps(dict(commit=commit,archive=str(archive),GPU_hours=hours,pairing=pairing)))
if __name__=='__main__':main()
