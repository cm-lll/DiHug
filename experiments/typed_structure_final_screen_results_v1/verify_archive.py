"""Verify public payload and rerun read-only graph/arithmetic audit on CPU."""
import argparse,pathlib,zipfile,subprocess,sys,json,time
from audit import sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--scratch',required=True);a=p.parse_args()
 root=pathlib.Path(a.root).resolve();scratch=pathlib.Path(a.scratch).resolve();scratch.mkdir(exist_ok=False);started=time.time()
 lines=(root/'SHA256SUMS').read_text().splitlines()
 for line in lines:
  expected,name=line.split('  ',1);f=(root/name).resolve();assert root in f.parents and f.is_file() and not f.is_symlink();assert sha(f)==expected,name
 with zipfile.ZipFile(root/'runtime_source.zip') as z:
  for member in z.infolist():
   assert scratch in (scratch/member.filename).resolve().parents
   assert (member.external_attr>>16)&0o170000 != 0o120000
  z.extractall(scratch)
 receipt=scratch/'public_audit.json'
 subprocess.run([sys.executable,str(root/'audit.py'),'--run',str(root/'run'),'--source',str(scratch/'typed_structure_final_screen_v1'),'--output',str(receipt),'--public-only'],check=True,timeout=180)
 out=dict(status='passed',payload_hashes=len(lines),public_audit=json.loads(receipt.read_text()),seconds=time.time()-started)
 (scratch/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status='passed',payload_hashes=len(lines),seconds=out['seconds'])))
if __name__=='__main__':main()
