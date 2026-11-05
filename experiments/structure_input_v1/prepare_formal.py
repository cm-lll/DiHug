"""CPU identity intake only. Does not construct a model or initialize CUDA."""
import os,json,hashlib,shutil,datetime
from pathlib import Path
def main():
    root=Path(__file__).resolve().parents[2];assets=Path(os.environ['STRUCTURE_ASSETS']);out=Path(os.environ['STRUCTURE_RUN']);out.mkdir(exist_ok=False)
    def read(p):return json.loads(p.read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
    begin=datetime.datetime.fromisoformat('2026-11-06T04:58:33+08:00').timestamp()
    save(out/'ledger.json',dict(start_epoch=begin,science_stop_epoch=begin+210*60,deadline_epoch=begin+240*60,GPU_hours_limit=12,
         ordinary_recompute_tolerance_passed=False,user_authorized_proceed_with_recorded_variation=True))
    base=assets/'results/cn_multiseed_20261104T042546';registry=read(root/'review/structure_input_v1/ASSET_BINDINGS.json')['assets'];tasks=[];audit=[]
    original=read(base/'tasks.json');ss=[2026102821,2026102822,2026110231,2026110232];contracts=read(base/'prepared/sampling_contracts.json')
    for gpu,(ds,seed) in enumerate([('IMDB',2026110411),('IMDB',2026110412),('PubMed',2026110421),('PubMed',2026110422)]):
        job=next(x for x in original if x['dataset']==ds and x['train_seed']==seed and x['arm']=='A');tid=job['task_id'];folder=base/'run'/tid
        match=next(x for x in registry if x['kind']=='checkpoint_bytes_only' and x['dataset']==ds and x['seed']==seed and x['update']==0)
        assert sha(folder/'initial.pt')==match['actual_sha256']
        tasks.append(dict(task_id=tid,dataset=ds,train_seed=seed,arm='STRUCTURE',gpu=gpu,config=job['config'],initial=str(folder/'initial.pt'),initial_sha256=sha(folder/'initial.pt')))
        for name in ('initial.pt','final.pt','readout_ema_0.99.pt'):
            p=folder/name;receipt=p.with_suffix('.sha256.json');assert sha(p)==read(receipt)['sha256'];audit.append(dict(path=str(p),sha256=sha(p)))
        c=base/'prepared'/f'input_contract_{ds}_{seed}.json';assert len(read(c))==3000
        (out/'prepared').mkdir(exist_ok=True);shutil.copyfile(c,out/'prepared'/c.name)
        logs=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()];expected=read(c)
        assert len(logs)==3000 and all(x['input_sha256']==y['sha256'] for x,y in zip(logs,expected))
        for sample in ss:
            sf=folder/'ema_0.99'/f'sample_{sample}';b=read(sf/'binding.json');gp=sf/'graph.npz'
            assert b['dataset']==ds and b['train_seed']==seed and b['sample_seed']==sample and b['updates']==3000 and b['configuration']=='A_late'
            assert sha(gp)==b['graph_sha256'] and b['model_sha256']==sha(folder/'readout_ema_0.99.pt')
            steps=[json.loads(x) for x in (sf/'steps.jsonl').read_text().splitlines()];cs=contracts[f'{ds}_{sample}']['steps']
            assert len(steps)==1000 and all(x['uniform_sha256']==y['uniform_sha256'] and x['query_plan_sha256']==y['query_plan_sha256'] and x['exact_coverage'] for x,y in zip(steps,cs))
            for version in ('original','lcc'):
                v=read(base/'evaluation'/tid/str(sample)/(version+'.json'));assert v['graph_binding']['graph_sha256']==sha(gp)
            audit.append(dict(path=str(gp),sha256=sha(gp),all1000_sampling_pairs=True))
    save(out/'tasks.json',tasks);save(out/'asset_audit.json',dict(passed=True,models=4,reused_graphs=16,files=audit))
    save(out/'prepared/sampling_contracts.json',contracts)
    save(out/'approval.json',dict(authorized=True,scope='4 candidate models x3000 normal1 CE; 4 T1000 per model',samples=ss,source_base='01024668cb5d22d601b622b36d4381940a458cc8',launcher_pid=None))
    for n in ('run','logs','leases','engineering'):(out/n).mkdir(exist_ok=True)
    print(json.dumps(dict(output=str(out),models=4,reused_graphs=16,passed=True)))
if __name__=='__main__':main()
