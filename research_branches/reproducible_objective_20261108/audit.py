"""Read-only completeness/identity audit; no torch or model invocation."""
import hashlib,json,sys,math
from pathlib import Path
OUT=Path(sys.argv[1]);ROOT=Path(__file__).resolve().parents[2]
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
r=read(OUT/'receipt.json');assert r['status']=='complete'
u=read(OUT/'updates.json');p=read(OUT/'probes.json');b=read(OUT/'asset_binding.json')
assert len(u)==128 and len(p)==108 and r['updates']==128 and r['forwards']==108
assert all(all(z['first_step_exact_against_previous'].values()) for z in r['gates']) and len(r['gates'])==4
for ds in ['IMDB','PubMed']:
    a={z['offset']:z for z in u if z['dataset']==ds and z['arm']=='CE'}
    c={z['offset']:z for z in u if z['dataset']==ds and z['arm']=='PS'}
    assert set(a)==set(c)==set(range(1,33))
    assert all((a[k]['input_sha256'],a[k]['t'])==(c[k]['input_sha256'],c[k]['t']) for k in a)
for z in b:assert sha(z['parent'])==z['parent_sha256']
old=read(ROOT/'results/joint_output_formal_projected_20261108T043000/source_binding.json')
assert all(sha(path)==h for path,h in old.items())
for z in p:
    assert all(math.isfinite(z[k]) for k in ['CE','positive_CE','negative_CE','expected_edges'])
    assert math.isclose(sum(q['expected_edges'] for q in z['relations']),z['expected_edges'],rel_tol=1e-6,abs_tol=1e-3)
lease=read(OUT/'lease.json');previous=read(OUT.parent/'lease.json')
assert lease['returncode']==0 and previous['returncode']==2
hours=lease['gpu_hours']+previous['gpu_hours'];assert hours<=.25
assert lease['end_epoch']<=read(OUT/'protocol.json')['science_stop_epoch']
result=dict(status='passed',updates=len(u),evaluation_forwards=len(p),paired_training_inputs=64,first_step_full_state_exact=4,
            original_source_files_unchanged=len(old),parents_unchanged=2,gpu_hours_including_zero_update_first_attempt=hours,
            diagnostic_input_duplicates=[dict(dataset=z['dataset'],**q) for z in b for q in z['inputs'] if q['duplicates_historical_input']],
            local_artifacts={str(q):sha(q) for q in OUT.glob('*.pt')},
            limitations=['No complete 32-step independent rerun in this ticket','PS-trained parent/Adam state; not a pure-CE historical control','Conditional predictions, no generated graphs'])
(OUT/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in result.items() if k!='local_artifacts'},ensure_ascii=False))
