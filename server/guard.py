"""One owned GPU process group, idle-device check and independent hard deadline."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,required=True);p.add_argument('--minutes',type=float,required=True);p.add_argument('--gpu-hours',type=float,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command or min(a.minutes,a.gpu_hours)<=0 or a.receipt.exists():p.error('new receipt, command and positive explicit budgets required')
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=20)
    rows=[l.split(',') for l in text.splitlines()];row=next(r for r in rows if int(r[0])==a.gpu)
    if float(row[2])!=0:raise RuntimeError('requested device is occupied; refusing to preempt')
    uuid=row[1].strip();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=uuid
    start=time.time();deadline=start+min(a.minutes*60,a.gpu_hours*3600);child=None;reason=None
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('guard terminated')))
    try:
        child=subprocess.Popen(command,env=env,start_new_session=True)
        while child.poll() is None:
            if time.time()>=deadline-30:reason='wall/GPU budget; reserve 30 seconds for saving';break
            time.sleep(1)
    except BaseException as exc:reason=repr(exc);raise
    finally:
        if child is not None:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=max(.01,min(30,deadline-time.time())))
                except subprocess.TimeoutExpired:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait()
            end=time.time();a.receipt.parent.mkdir(parents=True,exist_ok=True)
            a.receipt.write_text(json.dumps(dict(start_epoch=start,end_epoch=end,seconds=end-start,occupied_GPU_hours=(end-start)/3600,device=uuid,exit_code=child.returncode,stop_reason=reason,owned_process_exited=True),indent=2)+'\n')
    raise SystemExit(child.returncode if child is not None else 1)
if __name__=='__main__':main()
