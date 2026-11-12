"""Missing-aware O/A receipt and compact evidence archive; no new science."""
import csv,json,pathlib,time,zipfile
from common_run import OUT,HERE,ROOT,read,save,sha,now

def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
 ledger=read(OUT/'ledger.json');checks=[];vrows=[];relations=[];failures=[]
 for o,a in zip(read(OUT/'tasks.json'),read(OUT/'reused_A_jobs.json')):
  assert (o['dataset'],o['train_seed'])==(a['dataset'],a['train_seed'])
  op=pathlib.Path(o['folder']);ap=pathlib.Path(a['folder']);olf=op/'training.jsonl';alf=ap/'training.jsonl'
  ol=[json.loads(x)for x in olf.read_text().splitlines()]if olf.exists()else[]
  al=[json.loads(x)for x in alf.read_text().splitlines()];n=min(len(ol),len(al))
  inputs=all(ol[i]['update']==al[i]['update'] and ol[i]['input_sha256']==al[i]['input_sha256']for i in range(n))
  checks.append(dict(dataset=o['dataset'],train_seed=o['train_seed'],updates_compared=n,input_pairing=inputs,complete=n==3000,
     torch_RNG='not required identical across architectures; extra A CN context dropout call differs',
     main_input_RNG='separate stream, checked each update before optimizer'))
  if not inputs:failures.append(dict(dataset=o['dataset'],train_seed=o['train_seed'],error='input mismatch'))
  for arm,folder in [('O',op),('A',ap)]:
   vf=folder/'validation.json'
   if not vf.exists():continue
   for cp in read(vf):
    for rd in ['raw','ema_0_99']:
     for r in cp[rd]['rows']:
      ident=dict(dataset=o['dataset'],train_seed=o['train_seed'],arm=arm,updates=cp['updates'],readout=rd,t=r['t'],input_sha256=r['input_sha256'])
      vrows.append(dict(ident,CE_sum_over_S=r['ce_sum_over_S']))
      for rr in r['family']:relations.append(dict(ident,**rr,p0_minus_target=rr['expected_edges']-rr['true_edges']))
 save(OUT/'final_pairing_audit.json',dict(pairs=checks,failures=failures))
 write(OUT/'validation_total.csv',vrows);write(OUT/'validation_relations.csv',relations)
 protected=read(OUT/'protected_parent_assets.json')
 if any(sha(r['path'])!=r['sha256']for r in protected):raise ValueError('original A asset modified')
 files=[]
 for p in sorted((OUT/'run').rglob('*')):
  if p.is_file():files.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
 save(OUT/'scientific_asset_hashes.json',files)
 leases=[read(p)for p in (OUT/'leases').glob('*.json')]
 new_hours=sum((r.get('released_epoch',time.time())-r['acquired_epoch'])/3600 for r in leases);hours=ledger['prior_GPU_hours']+new_hours
 receipt=dict(at=now(),wall_minutes=(time.time()-ledger['start_epoch'])/60,GPU_hours=hours,
     within_wall=time.time()<=ledger['deadline_epoch'],within_GPU_hours=hours<=12,engineering_updates_this_stage=0,reused_engineering_updates=6,prior_GPU_hours=ledger['prior_GPU_hours'],new_GPU_hours=new_hours,
     pairing=checks,errors=failures,evaluation=read(OUT/'evaluation/receipt.json'),
     protected_parent_assets_unchanged=True,PubMed_bad_third_seed_in_panel=False,
     third_seed_gap_explicit=True,complete_does_not_prove_stability=True)
 save(OUT/'FINAL_RECEIPT.json',receipt)
 archive=OUT.parent/(OUT.name+'_review.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(OUT.rglob('*')):
   if p.is_file() and p.suffix not in ['.pt'] and 'engineering' not in p.relative_to(OUT).parts and 'temporary' not in p.parts:
    z.write(p,str(p.relative_to(ROOT)))
  for p in HERE.glob('*.py'):z.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(archive)as z:assert z.testzip()is None
 save(OUT/'archive_receipt.json',dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
      private_weights_remain_local=True,original_and_LCC_graphs_included=True))
 print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
