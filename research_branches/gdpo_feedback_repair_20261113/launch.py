"""Independent hard watchdog for the sole owned diagnostic process group."""
import os, sys, json, time, subprocess, signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/gdpo_feedback_repair_20261113'
OUT.mkdir(exist_ok=True)
start=time.time()
science=1794502750.0  # 2026-11-13 00:59:10 +08:00
# Absolute stop also read from the frozen manifest to avoid a timezone ambiguity.
p=json.loads((OUT/'protocol.json').read_text());science=p['science_stop_epoch']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',ALLOCATION_EPOCH=str(start),SCIENCE_STOP_EPOCH=str(science))
log=(OUT/'gpu.log').open('w')
child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('gradient_audit.py'))],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(OUT/'launch.json').write_text(json.dumps(dict(pid=child.pid,supervisor_pid=os.getpid(),GPU=0,allocation_epoch=start,science_stop_epoch=science),indent=2))
reason='completed'
try: child.wait(timeout=max(.1,min(science-start,1800)))
except subprocess.TimeoutExpired:
    reason='watchdog_limit';os.killpg(child.pid,signal.SIGTERM)
    try: child.wait(timeout=5)
    except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
finally:
    log.close();end=time.time()
    (OUT/'release.json').write_text(json.dumps(dict(reason=reason,exit_code=child.returncode,start_epoch=start,end_epoch=end,GPU_hours=(end-start)/3600,owned_process_exited=True),indent=2))
