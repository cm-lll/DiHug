"""Aggregate eight complete, content-bound CPU trajectory checks; no generation."""
import confirmation_common as contract
import confirmation_prefix_v2 as producer
from round01 import b, Path, ROOT, SRC

OUT = producer.OUT
BASE = OUT.parent
FINAL = BASE / 'prefix_chunk_finalized'

def check_manifest(folder):
    for path, expected in b.read(folder / 'manifest.json')['files'].items():
        assert b.sha(path) == expected, path

def main():
    b.setup()
    assert not (OUT / 'acceptance.json').exists()
    assert not (OUT / 'summary.json').exists()
    check_manifest(OUT)
    jobs = [(ds, sid, arm) for ds in producer.DATASETS
            for sid in producer.SEEDS for arm in producer.ARMS]
    rows, receipts, files = [], [], [Path(__file__), SRC / 'CONFIRMATION_PREFIX_CHUNKS.md']
    files += [OUT / 'manifest.json', OUT / 'completion.json', OUT / 'interruption.json']
    for index, (ds, sid, arm) in enumerate(jobs):
        folder = BASE / 'prefix_chunks' / str(index)
        check_manifest(folder)
        assert not (folder / 'failure.json').exists()
        receipt = b.read(folder / 'acceptance.json')
        assert receipt['passed'] and receipt['index'] == index
        assert (receipt['dataset'], receipt['seed'], receipt['arm']) == (ds, sid, arm)
        assert receipt['seconds'] <= 300
        assert receipt['feature_error'] <= 1e-5 and receipt['logit_error'] <= 2e-5
        assert receipt['new_graphs'] == receipt['new_training_updates'] == 0
        scientific = OUT / 'outputs' / ds / f'{sid}_{arm}'
        row = b.read(scientific / 'report.json')
        assert receipt['replayed_opportunities'] == row['steps']
        rows.append(row); receipts.append(receipt)
        files += [folder / 'manifest.json', folder / 'acceptance.json']
        files += [Path(p) for p in b.read(folder / 'manifest.json')['files']]
    total = sum(v['replayed_opportunities'] for v in receipts)
    seconds = sum(v['seconds'] for v in receipts)
    assert total == b.read(OUT / 'completion.json')['scientific_opportunities'] == 30432
    assert seconds <= 2400
    FINAL.mkdir(parents=True, exist_ok=False)
    contract.bind(FINAL, files, new_graphs=0, new_training_updates=0,
                  source=str(OUT), independent_chunks=8)
    summary = dict(rows=rows, all_C_arrived=all(r['arrived'] for r in rows if r['arm']=='C'),
                   all_D_arrived=all(r['arrived'] for r in rows if r['arm']=='D'),
                   new_original_MLP_fits=0, complete_two_stage=False,
                   independent_finalization=str(FINAL))
    contract.put(FINAL, 'summary.json', summary)
    contract.verify(FINAL)
    outputs = {str(p): b.sha(p) for p in sorted(set(files))}
    receipt = dict(passed=True, replayed_opportunities=total, independent_chunks=8,
                   feature_error=max(r['feature_error'] for r in receipts),
                   logit_error=max(r['logit_error'] for r in receipts), seconds=seconds,
                   new_graphs=0, new_training_updates=0, outputs=outputs)
    contract.put(FINAL, 'acceptance.json', receipt)
    delegated = dict(receipt, acceptance_source=str(FINAL/'acceptance.json'),
                     acceptance_source_sha256=b.sha(FINAL/'acceptance.json'),
                     independent_finalization=True, original_worker_exit_code=143)
    delegated['outputs'] = dict(outputs)
    for path in [FINAL/'manifest.json', FINAL/'summary.json', FINAL/'acceptance.json']:
        delegated['outputs'][str(path)] = b.sha(path)
    contract.put(OUT, 'summary.json', summary)
    contract.put(OUT, 'acceptance.json', delegated)
    print('Eight prefixes accepted:', total, 'opportunities;', seconds, 'CPU seconds', flush=True)

if __name__ == '__main__':
    main()
