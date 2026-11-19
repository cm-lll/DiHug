from r2_generate import *
def main(out):
 out.mkdir(parents=True,exist_ok=False);records={}
 for ds in ASSETS:
  p=OLD/ASSETS[ds]['directory']/'mixed_edit_diffusion.pt';r=generate(out/ds,ds,100,p,64)
  # Verified PR3B has both an access ledger and the distinct attempt ledger.
  old=HISTORY/ds/'seed100/r0/M/attempts.jsonl'
  assert old.is_file()
  with old.open() as f:history=[json.loads(next(f))['record'] for _ in range(64)]
  with gzip.open(out/ds/'actions.jsonl.gz','rt') as f:new=[json.loads(line)['record'] for line in f]
  assert len(new)==64
  for a,b in zip(new,history):
   assert a['attempt']==b['attempt'] and a['time']==b['time'] and a['selected']==b['selected'] and a['selected_index']==b['selected_index']
   assert a['candidates_sha256']==digest(b['candidates'])
  report=json.loads((out/ds/'report.json').read_text());records[ds]={'matched_old_attempts':64,'wall_seconds':report['wall_seconds'],'cpu_seconds':report['cpu_seconds'],'per_attempt_seconds':report['wall_seconds']/64}
 atomic(out/'gate.json',{'passed':True,'records':records,'scope':'64 consecutive historical attempts per reference, exact candidates, selected edits and time; not entire generator replay','new_scientific_samples':0})
 print(json.dumps(records),flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 try:main(a.out)
 except BaseException as e:atomic(a.out/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
