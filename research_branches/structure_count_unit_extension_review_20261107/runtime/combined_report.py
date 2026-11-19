"""Merge completed CPU evaluations; new seed remains separately identified."""
import json,csv,statistics,hashlib
from pathlib import Path
from .formal_runtime import OUT,read,save
def main():
    parent=Path(read(OUT/'ledger.json')['parent_run']);dest=OUT/'combined_evaluation';dest.mkdir(exist_ok=True)
    sources=[];rows=[];means=[];deltas=[]
    for directory,label in [(parent,'previous_two_development_seeds'),(OUT,'prospective_new_seed')]:
        receipt=directory/'evaluation/receipt.json'
        if not receipt.exists():
            sources.append(dict(root=str(directory),status='missing evaluation'));continue
        rec=read(receipt);sources.append(dict(root=str(directory),receipt=rec,receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest()))
        for name,target in [('per_graph_all_metrics',rows),('per_model_four_sample_mean',means),('paired_UNIT_minus_PB',deltas)]:
            f=directory/'evaluation'/(name+'.json')
            if f.exists():target.extend(dict(r,panel=label) for r in read(f))
    grouped={}
    for r in means:
        key=tuple(r[k] for k in ['dataset','arm','version','kind','metric'])
        grouped.setdefault(key,[]).append(r)
    across=[]
    for key,items in grouped.items():
        vals=[r['mean'] for r in items]
        assert len(set(r['train_seed'] for r in items))==len(items),'duplicate model average'
        complete=len(items)==3 and all(v is not None for v in vals)
        across.append(dict(zip(['dataset','arm','version','kind','metric'],key),mean=statistics.mean(vals) if complete else None,training_seed_std=statistics.stdev(vals) if complete else None,training_seeds=[r['train_seed'] for r in items],training_seed_means=vals,expected_training_seeds=3,complete=complete,limitation='two development seeds plus one prospective seed; four samples are not independent trained models'))
    for name,data in [('per_graph_all_metrics',rows),('per_model_four_sample_mean',means),('paired_UNIT_minus_PB',deltas),('across_three_training_seeds',across)]:
        save(dest/(name+'.json'),data)
        if data:
            with (dest/(name+'.csv')).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)));writer.writeheader();writer.writerows(data)
    save(dest/'sources.json',sources)
    print('Combined CPU report saved; missing/null rows retained.')
if __name__=='__main__':main()
