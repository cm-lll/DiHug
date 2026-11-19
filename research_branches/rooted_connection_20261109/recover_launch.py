"""Preserve zero-update attempt; correct only resource accounting, same clock/budget."""
import json,time,hashlib,shutil,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SRC=Path(__file__).resolve().parent;OUT=ROOT/'results/rooted_connection_20261109T015142'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    dest=OUT/'failures/attempt1';dest.mkdir(parents=True,exist_ok=False)
    assert (OUT/'evaluation_exit.json').exists()
    assert not any(p.stat().st_size for p in (OUT/'run').glob('*/training.jsonl'))
    for p in (OUT/'run').glob('*/progress.json'):assert read(p)['updates']==0
    for p in (OUT/'leases').glob('*.json'):assert 'released_epoch' in read(p)
    for name in ['run','evaluation','STOP_SCIENCE','launcher_summary.json','launcher_progress.json','watchdog.json','evaluation_exit.json','delivery.json','RESULT_ZH.md','quality_screen.json']:
        p=OUT/name
        if p.exists():p.rename(dest/name)
    for name in ['source_binding.json','ledger.json','approval.json','budget.json','execution_source_amendment.json','formal_launch_receipt.json','frozen_execution_source.tar.gz']:
        shutil.copy2(OUT/name,dest/name)
    shutil.copy2(OUT/'logs/evaluation.log',dest/'evaluation.log')
    for name in ['run','evaluation']:(OUT/name).mkdir()
    ledger=read(OUT/'ledger.json');ledger.update(GPUworker_RSS_limit_MiB=8192,GPUworkers_RSS_total_MiB=65536,
        resource_amendment='CPU feature-only cap2GiB retained; total GPU worker includes CUDA runtime/model checkpoints, separately8GiB each64GiB total; host available494GiB. Science wall/GPUhour unchanged.')
    (OUT/'ledger.json').write_text(json.dumps(ledger,indent=2))
    old=read(OUT/'source_binding.json');changed={p:(h,sha(Path(p))) for p,h in old.items() if h!=sha(Path(p))}
    assert set(changed)=={str(SRC/'runtime/supervise_formal.py')},changed
    for p in SRC.rglob('*.py'):old[str(p)]=sha(p)
    (OUT/'source_binding.json').write_text(json.dumps(old,indent=2))
    with tarfile.open(OUT/'frozen_execution_source_attempt2.tar.gz','w:gz') as tar:
        for p in old:tar.add(p,arcname=str(Path(p).relative_to(ROOT)))
    record=dict(epoch=time.time(),formal_updates_first_attempt=0,changed=changed,reason='watchdog applied CPU-only2GiB cap to GPU worker totalRSS3.59GiB; host available494GiB',
        scientific_code_unchanged=True,all_old_leases_retained_and_charged=True,original_deadlines_unchanged=True,recovery='same sealed initial state; no completed formal updates discarded')
    (OUT/'resource_repair.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
if __name__=='__main__':main()
