"""Read-only CPU interpretation tables for the fixed effect trial.

This script never imports a model or changes the execution protocol.
"""
import csv
import hashlib
import json
import statistics as st
import sys
import time
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def table(path, rows):
    save(path.with_suffix('.json'), rows)
    if rows:
        with path.with_suffix('.csv').open('w') as f:
            writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)


def main(out):
    protocol = load(out / 'protocol.json')
    validation, groups, execution, paired, missing = [], [], [], [], []
    for parent in protocol['parents']:
        model = parent['id']
        common = dict(model=model, dataset=parent['dataset'], seed=parent['train_seed'])
        manifests = []
        for arm in ['CE', 'calibrated_micro_macro']:
            folder = out / model / arm
            if not (folder / 'result.json').exists():
                missing.append(dict(**common, arm=arm, reason='result missing'))
                continue
            data = load(folder / 'result.json')
            manifests.append(load(folder / 'inputs.json'))
            updates = data['updates']
            complete = load(folder / 'completion.json') if (folder / 'completion.json').exists() else None
            execution.append(dict(**common, arm=arm, updates=len(updates),
                update_seconds_mean=st.mean(r['seconds'] for r in updates) if updates else None,
                update_seconds_sum=sum(r['seconds'] for r in updates),
                unclipped_gradient_mean=st.mean(r['gradient_norm'] for r in updates) if updates else None,
                unclipped_gradient_max=max((r['gradient_norm'] for r in updates), default=None),
                gradient_clip_count=sum(r['gradient_norm'] > 1 for r in updates),
                recovery=data.get('recovery'), completion=complete,
                validation_overlaps_train=data.get('validation_overlaps_train')))
            reads = [(0, 'raw_parent_EMA3000', data['baseline'])] if data.get('baseline') else []
            reads += [(int(step), readout, rows) for step, dd in data['evaluations'].items() for readout, rows in dd.items() if readout in ('raw', 'ema')]
            for step, readout, rows in reads:
                for index, row in enumerate(rows):
                    identity = dict(**common, arm=arm, step=step, readout=readout,
                                    t=row['t'], input_index=index, input_hash=row['input_hash'])
                    validation.append(dict(**identity, **{k:row[k] for k in ('CE','positive_CE','negative_CE')},
                                           **{'block_'+k:v for k,v in row['blocks'].items()}))
                    for relation in row['relations']:
                        groups.append(dict(**identity, **relation))
        if len(manifests) == 2:
            assert manifests[0] == manifests[1], ('paired inputs differ', model)
    for row in validation:
        if row['arm'] != 'calibrated_micro_macro':
            continue
        base = next((r for r in validation if r['model']==row['model'] and r['arm']=='CE'
                     and r['step']==row['step'] and r['readout']==row['readout'] and r['input_hash']==row['input_hash']), None)
        assert base is not None, ('validation pair missing', row)
        paired.append(dict(**{k:row[k] for k in ('model','dataset','seed','step','readout','t','input_index','input_hash')},
            **{k+'_delta':row[k]-base[k] for k in ('CE','positive_CE','negative_CE','block_degree','block_closure','block_propagation')}))
    table(out / 'validation_per_input', validation)
    table(out / 'validation_per_relation', groups)
    table(out / 'validation_paired', paired)
    save(out / 'training_execution_audit.json', execution)
    save(out / 'receipt_missing.json', missing)
    allocation = load(out / 'allocations.json')
    stages = {}
    for row in allocation:
        stage = 'calibration' if row['tag']=='calibration' else 'training' if row['tag'].startswith('train_') else 'generation'
        seconds = (row.get('end_epoch') or time.time()) - row['start_epoch']
        stages[stage] = stages.get(stage, 0) + seconds / 3600
    save(out / 'resource_by_stage.json', dict(GPU_hours=stages,
        total=sum(stages.values()), all_released=all(r.get('end_epoch') is not None for r in allocation),
        meaning='Allocation start through observed process exit; includes loading, saving and waiting.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(validation_rows=len(validation), relation_rows=len(groups), paired_rows=len(paired), models=len(execution), missing=missing)))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
