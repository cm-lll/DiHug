"""Start exactly one autonomous, bounded queue after preflight."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/single_step_formal_resume_20261111'
assert json.loads((OUT/'preflight_receipt.json').read_text())['passed']
lock=os.open(str(OUT/'launch.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(lock,str(time.time()).encode());os.close(lock)
with (OUT/'supervisor.log').open('x') as log:p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('supervise.py'))],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(supervisor_pid=p.pid,launch_epoch=time.time());(OUT/'launcher.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
