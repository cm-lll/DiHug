import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from worker import *
def main():
    OUT.mkdir(exist_ok=False)
    old=ROOT/'results/same_input_pubmed_completion_20261110';p=json.loads((old/'protocol.json').read_text());bindings=json.loads((old/'bindings.json').read_text())
    for path,h in bindings.items():assert rt.sha(path)==h,path
    for row in p['tasks']:
        src=ROOT/('results/same_input_update_check_20261110' if row['dataset']=='IMDB' else 'results/same_input_pubmed_completion_20261110')/f"{row['dataset']}_{row['seed']}"
        row['response_source']=str(src);assert json.loads((src/'checks.json').read_text())['evaluation_state_unchanged']
        for path in src.glob('*'):
            if path.is_file():bindings[str(path)]=rt.sha(path)
        bindings[str(src.parent/'evaluation_summary.csv')]=rt.sha(src.parent/'evaluation_summary.csv')
    p.update(arms=['saved_direction_alpha_0.25','saved_direction_alpha_0.5'],updates_each=0,optimizer='no optimizer calls; immutable parent AdamW/EMA/RNG',alphas=[.25,.5],planned_calls=dict(forwards=56,backwards=0,updates=0,reverse_chains=0),budgets=dict(wall_seconds=480,GPU_seconds=288,GPUs=1),interpretation='Read-only response curve. Cached endpoints. No step selection, adaptive backtracking or new training rule.')
    for path in Path(__file__).parent.glob('*.py'):bindings[str(path)]=rt.sha(path)
    rt.save(OUT/'protocol.json',p);rt.save(OUT/'bindings.json',bindings);rt.save(OUT/'freeze.json',dict(epoch=time.time(),protocol_sha256=rt.sha(OUT/'protocol.json')))
if __name__=='__main__':main()
