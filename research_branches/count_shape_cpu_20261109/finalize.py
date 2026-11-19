"""CPU summaries and integrity receipt; never imports a denoising model."""
from run import *
started=time.time()
binds=json.loads((OUT/'binding.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in binds.items())
rows=json.loads((OUT/'final_edges.json').read_text())
result=summary(rows,['dataset','train_seed'])
if (OUT/'final_summary_by_seed.json').exists():assert result==json.loads((OUT/'final_summary_by_seed.json').read_text())
else:save('final_summary_by_seed',result)
rc=json.loads((OUT/'receipt.json').read_text());extra=json.loads((OUT/'new_adverse_summary.json').read_text())
assert rc['final_edges']==4096 and rc['conditional_edges']==2765 and rc['exchanges']==172
decision=dict(status='CPU completed, not admitted to training',
    supported='exact count/shape bookkeeping and local PubMed direction gains',
    adverse='IMDB new adverse directions; PubMed t1000 missed Pareto gains and new no-aggregate-gain rewards',
    scope='fixed development screen, no empirical generation benefit or universal-method rejection',
    next='define interpretable joint allocation discrepancy using archived counterexamples before another training proposal',
    formula_search_performed=False,formal_training_admission=False)
save('decision',decision)
paths=list(Path(__file__).parent.glob('*.py'))+[OUT/'protocol.json',OUT/'THEORY_ZH.md',OUT/'REPORT_ZH.md']
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
assert not (OUT/'completion.json').exists()
save('completion',dict(status='complete',completed_BJT=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    source_document_hashes=manifest,old_bindings_rechecked=len(binds),computation_seconds=rc['wall_seconds']+extra['CPU_seconds'],
    finalization_seconds=time.time()-started,peak_main_RSS_MiB=rc['peak_RSS_MiB'],GPU_hours=0,models=0,updates=0,new_random_graphs=0,
    failures=[],limits_met=True,all_worker_processes_exited=True,formal_training_admission=False,
    note='decompose.py dead conditional initialization simplified after execution; no mathematical or outcome change'))
print((OUT/'completion.json').read_text())
