"""Own-process hard timeout and allocated GPU-hour accounting."""
import sys,os,json,time,subprocess,signal
from pathlib import Path
def main():
    out=Path(sys.argv[1]);p=json.loads((out/'protocol.json').read_text());start=time.time()
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=p['gpu_uuid']
    with (out/'worker.log').open('w') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run.py')),str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (out/'launch.json').write_text(json.dumps(dict(pid=child.pid,start_epoch=start,device=p['gpu_uuid'],science_stop_epoch=p['science_stop_epoch']),indent=2))
        stop=min(p['science_stop_epoch'],start+1800)
        try:rc=child.wait(timeout=max(0,stop-time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
    end=time.time()
    (out/'release.json').write_text(json.dumps(dict(pid=child.pid,exit_code=rc,start_epoch=start,end_epoch=end,
        allocated_GPU_hours=(end-start)/3600,child_exited=True,device=p['gpu_uuid']),indent=2))
    sys.exit(rc)
if __name__=='__main__':main()
