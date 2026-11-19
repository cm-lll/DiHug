from run import *
start=time.time();b=json.loads((OUT/'binding.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in b.items())
main=json.loads((OUT/'receipt.json').read_text());extra=json.loads((OUT/'case_ledger_summary.json').read_text())
assert main['final_slots']==4096 and main['conditional_slots']==2765
rs=json.loads((OUT/'case_ledger.json').read_text());remaining=[r for r in rs if r['joint_change']<-EPS]
save('remaining_case_metric_summary',[dict(metric=k,better=sum(r[k+'_change']<-EPS for r in remaining),worse=sum(r[k+'_change']>EPS for r in remaining),unchanged=sum(abs(r[k+'_change'])<=EPS for r in remaining)) for k in METRICS])
save('decision',dict(status='diagnostic complete; exact matching not recommended as a training replacement',
    evidence='18/23 IMDB previous adverse cases remain rewarded;14 improve rooted pathW1 while sacrificing principal metrics',
    next='single CPU output-path ablation with both candidate/reference path coordinates zeroed; freeze original width/scales/directions/no renormalization',
    remaining_uncertainty='objective tradeoffs are not proven cause of neural training instability; no new actual gradient/update or generation result',
    formal_training_admission=False,next_ablation_executed=False))
paths=list(Path(__file__).parent.glob('*.py'))+[OUT/'THEORY_ZH.md',OUT/'REPORT_ZH.md',OUT/'protocol.json']
assert not (OUT/'completion.json').exists()
save('completion',dict(status='complete',completed_BJT=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},old_bindings_rechecked=len(b),
    computation_seconds=main['wall_seconds']+extra['CPU_seconds'],finalization_seconds=time.time()-start,
    peak_RSS_MiB=max(main['peak_RSS_MiB'],extra['peak_RSS_MiB']),failures=[],GPU_hours=0,model_calls=0,updates=0,new_random_graphs=0,
    all_worker_processes_exited=True,formal_training_admission=False))
print((OUT/'completion.json').read_text())
