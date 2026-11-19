import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from worker import *
PREVIOUS=ROOT/'results/same_input_update_check_20261110'
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists();assert not list(OUT.glob('PubMed_*'));p=json.loads((PREVIOUS/'protocol.json').read_text());bindings=json.loads((PREVIOUS/'bindings.json').read_text())
    for path,h in bindings.items():assert rt.sha(path)==h,path
    for name in ('audit.json','delivery_receipt.json','decision_summary.json','evaluation_summary.csv','risk_changes.csv','relations.csv','node_groups.csv','REPORT_ZH.md'):
        bindings[str(PREVIOUS/name)]=rt.sha(PREVIOUS/name)
    for row in p['tasks'][:2]:
        folder=PREVIOUS/f"{row['dataset']}_{row['seed']}";assert json.loads((folder/'receipt.json').read_text())['status']=='completed'
        for name in ('checks.json','updates.json','receipt.json'):bindings[str(folder/name)]=rt.sha(folder/name)
    for row in p['tasks'][2:]:assert not (PREVIOUS/f"{row['dataset']}_{row['seed']}").exists(),'Previously missing condition now exists; do not duplicate'
    a=(ROOT/'research_branches/same_input_update_check_20261110/worker.py').read_text();b=Path(__file__).with_name('worker.py').read_text()
    assert a.replace("OUT=ROOT/'results/same_input_update_check_20261110'","OUT=ROOT/'results/same_input_pubmed_completion_20261110'").rstrip('\n')==b.rstrip('\n')
    assert (ROOT/'research_branches/same_input_update_check_20261110/evaluate_cpu.py').read_text().rstrip('\n')==Path(__file__).with_name('evaluate_cpu.py').read_text().rstrip('\n')
    p.update(execute_indices=[2,3],previous_IMDB_results=str(PREVIOUS),planned_calls=dict(training_forwards=2,same_input_after_forwards=2,other_input_after_forwards=12,VJPs=10,temporary_updates=2,reverse_chains=0),budgets=dict(wall_minutes_from_GPU_start=5,science_seconds_from_GPU_start=210,GPU_hours=.05,GPUs=1,CPU_threads=1,RSS_stop=None),budget_note='New approved completion window; original ended window and missing GPU accounting remain preserved and are not repaired by this ticket.')
    for path in Path(__file__).parent.glob('*'):
        if path.suffix in ('.py','.md'):bindings[str(path)]=rt.sha(path)
    rt.save(OUT/'protocol.json',p);rt.save(OUT/'bindings.json',bindings);rt.save(OUT/'cpu_gate.json',dict(passed=True,worker_change_only_output_path=True,evaluation_source_identical=True,solver_and_state_gates_reused=True,missing_PubMed_assets_confirmed=True));rt.save(OUT/'freeze.json',dict(epoch=time.time(),protocol_sha256=rt.sha(OUT/'protocol.json'),bindings_sha256=rt.sha(OUT/'bindings.json')))
    print(json.dumps(dict(output=str(OUT),execution_indices=[2,3],GPU_hours=.05,wall_minutes=5)))
if __name__=='__main__':main()
