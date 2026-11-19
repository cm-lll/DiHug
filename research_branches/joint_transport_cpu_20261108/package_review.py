"""Package local evidence only; never include parent weights or datasets."""
import datetime, hashlib, json, time, zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
home = Path(__file__).parent
out = root / 'results/joint_transport_cpu_20261108'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
protocol, receipt = read(out / 'protocol.json'), read(out / 'receipt.json')
now = time.time()
assert receipt['status'] == 'complete'
assert now < protocol['delivery_stop_epoch']
delivery = dict(status='complete_stopped', start_epoch=protocol['start_epoch'],
                scientific_end_epoch=receipt['end_epoch'], report_end_epoch=now,
                wall_minutes=(now-protocol['start_epoch'])/60,
                scientific_worker_seconds=receipt['seconds'],
                worker_peak_RSS_MiB=receipt['peak_RSS_MiB'], CPU_threads=1,
                GPU_hours=0, model_calls=0, updates=0, T1000_chains=0,
                formal_training_authorized=False, missing=[], scientific_failures=[],
                maintenance_notes=['Initial documentation patch used a wrong heading; rejected without applying; corrected.'],
                decision='Do not advance this frozen joint-transport score to training.',
                sources={str(p.relative_to(root)):sha(p) for p in sorted(home.glob('*.py'))},
                source_git_head='df5c8898d91422b5fced712e718fc8f32290de73',
                source_git_commit_is_not_new_work=True,
                timezone='Asia/Shanghai')
(out / 'delivery.json').write_text(json.dumps(delivery, indent=2))
files = sorted(p for p in out.iterdir() if p.is_file()) + sorted(home.glob('*.py'))
files += [root / 'research_branches/hard_structure_rebar_cpu_20261108/objective.py',
          root / 'research_branches/hard_structure_edge_direction_cpu_20261108/metrics.py']
(out / 'review_file_hashes.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in files}, indent=2))
files.append(out / 'review_file_hashes.json')
archive = out.with_name(out.name + '_review.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, str(p.relative_to(root)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
record = dict(path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive),
              files=len(files), wall_minutes=delivery['wall_minutes'])
(out / 'archive_receipt.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
