"""One reservation, one frozen ablation; no source mutation or new task search."""
import sys,json,time,subprocess
from pathlib import Path
def main():
 out=Path(sys.argv[1]).resolve();key=sys.argv[2];p=json.loads((out/'protocol.json').read_text());job=next(x for x in p['jobs'] if x['id']==key)
 folder=out/key;folder.mkdir(exist_ok=False);row=dict(p['parents'][job['parent']],reward=job['reward'],cached_complete=job['cached_complete']);ds=job['dataset']
 q=dict(p,parents={ds:row},terminal_plan={ds:job['terminal_plan']});(folder/'protocol.json').write_text(json.dumps(q,indent=2))
 subprocess.run([sys.executable,str(Path(__file__).with_name('worker.py')),str(folder),ds,'T_eager'],check=True)
 (folder/'complete.json').write_text(json.dumps(dict(status='complete',at_epoch=time.time(),new_updates=job['new_updates'],graphs=4)))
if __name__=='__main__':main()
