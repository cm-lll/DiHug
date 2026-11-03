"""Isolated generation runtime; the scientific implementation remains frozen."""
import datetime,hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2]
HERE=pathlib.Path(__file__).resolve().parent
SCI=ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1'
OLD=ROOT/'experiments/late_final_single_input_v1'
TASKOUT=ROOT/'results/d_late_ard_formal_20261103T095019'
OUT=TASKOUT/'generation'
sys.path.extend([str(OLD),str(SCI/'package'),str(SCI)])
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def check_time():
 if (TASKOUT/'STOP_SCIENCE').exists() or time.time()>=read(TASKOUT/'ledger.json')['science_stop_epoch']-45:raise TimeoutError('scientific stop/save reserve')
def cfg_for(base,seed):
 return dict(base,seed=seed,inputs_per_update=1,rounds=1,formal_authorized=True,engineering_only=False,arms=['A','E'],max_updates_per_dataset_arm=3000,runtime='late_final_single_input_v1',source_note='Frozen late-CN science; new single-input CE runtime; final dictionaries approval-bound',query_contract='R1 complete movable slots; CE sum/full schema S; one input/update')
