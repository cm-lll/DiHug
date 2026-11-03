"""Frozen A/E-late single-input runtime; scientific modules are read-only."""
import datetime, hashlib, json, pathlib, sys, time, os
ROOT=pathlib.Path(__file__).resolve().parents[2]
HERE=pathlib.Path(__file__).resolve().parent
SCI=ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1'
OUT=pathlib.Path(os.environ['DIHUG_O_CN_RUN']).resolve()
if OUT.parent!=ROOT/'results' or not OUT.name.startswith('o_cn_matched_formal_'):raise ValueError('isolated formal output binding required')
# Runtime entry names (especially launch.py) must resolve to this isolated ticket;
# the frozen package also contains a historical launch.py with a different API.
sys.path[:0]=[str(HERE),str(SCI/'package'),str(SCI)]
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def check_time():
 if (OUT/'STOP_SCIENCE').exists() or time.time()>=read(OUT/'ledger.json')['science_stop_epoch']-45:raise TimeoutError('scientific stop/save reserve')
def cfg_for(base,seed):return dict(base,seed=seed,inputs_per_update=1,rounds=1,formal_authorized=True,engineering_only=False,arms=['O'],max_updates_per_dataset_arm=3000,runtime='o_cn_matched_normal1_v1',source_note='Frozen late-CN science; new single-input CE runtime; final dictionaries approval-bound',query_contract='R1 complete movable slots; CE sum/full schema S; one input/update')
def spawn_gpu(args,gpu,identifier):
 import os,subprocess
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
          OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TMPDIR=str(OUT/'temporary'))
 log=(OUT/'logs'/f'{identifier}.log').open('x');began=time.time()
 p=subprocess.Popen([sys.executable,'-u',*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text().split()[21]
 lease=OUT/'leases'/f'{identifier}.json';save(lease,dict(pid=p.pid,proc_start_ticks=ticks,GPU=gpu,id=identifier,acquired_epoch=began))
 return p,log,lease
def release(lease):
 r=read(lease);r['released_epoch']=time.time();save(lease,r)
