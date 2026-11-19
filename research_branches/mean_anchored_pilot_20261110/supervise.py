"""Independent wall/RSS watchdog for this CPU-only worker."""
import os,sys,time,json,subprocess,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results'/('mean_anchored_pilot_20261110'+('_attempt2' if len(sys.argv)>1 else ''))
OUT.mkdir(exist_ok=False)
start=time.time();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PILOT_OUT=str(OUT))
original_start=json.loads((ROOT/'results/mean_anchored_pilot_20261110/protocol.json').read_text())['start_epoch'] if len(sys.argv)>1 else start
protocol=dict(start_epoch=start,scientific_limit_seconds=600,worker_limit_seconds=660,memory_RSS_limit_MiB=2048,CPU_threads=1,GPU_hours=0,states=28,c=.5,metric='Euclidean logit',direction_draws=128,evaluation_draws=128,arms=['unchanged','original','average','cagrad','same_norm_average'],step_rule='eta=norm(original saved alpha1 logit displacement)/norm(mean five-risk gradient)',support='preserve fixed zero/one',scope='old development states; no network, optimizer or reverse chain',stop_on_solver_failure='retain failure, stop affected state',no_parameter_search=True)
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
peak=0;reason=None
with (OUT/'worker.log').open('w') as log:
    p=subprocess.Popen([__import__('os').environ.get('DIHUG_SPARSE_BLOCK_PYTHON', __import__('sys').executable),str(Path(__file__).with_name('run.py'))],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while p.poll() is None:
        try:
            status=Path(f'/proc/{p.pid}/status').read_text();rss=int(next(x.split()[1] for x in status.splitlines() if x.startswith('VmRSS:')))/1024;peak=max(peak,rss)
        except (FileNotFoundError,StopIteration):pass
        if time.time()-original_start>660:reason='wall_limit'
        if peak>2048:reason='RSS_limit'
        if reason:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10);break
        time.sleep(.5)
receipt=dict(start_epoch=start,end_epoch=time.time(),seconds=time.time()-start,worker_pid=p.pid,exit_code=p.returncode,stop_reason=reason,peak_RSS_MiB=peak,GPU_hours=0,worker_reaped=True)
(OUT/'supervisor.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
