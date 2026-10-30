"""Optional standalone recomputation of delivered graphs; CPU metrics only.

python -B recompute_cpu.py --root EXTRACTED_PACKAGE --output NEW_CPU_DIRECTORY
Uses existing numpy/networkx. No torch, model or installation is required.
"""
import argparse, importlib.util, json, pathlib, time
import numpy as np
def module(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve();out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    source=root/'frozen_evaluation_source';M=module(source/'lcc_eval/metrics.py','cpu_metrics');E=module(source/'evaluation_fields.py','cpu_fields');v=source/'lcc_eval/vendor';F=M.frozen(v/'frozen_metrics.py',v/'aligned_evaluate.py',v/'hgen_utils.py')
    names={'IMDB':['Director','Actor','Movie','Genre'],'PubMed':['Gene','Disease','Chemical','Species']};refs={};doc=json.loads((root/'evaluation.json').read_text())
    for d in {r['dataset'] for r in doc['records']}:
        z=np.load(root/f'reference/{d}.npz',allow_pickle=False);g=M.make_graph(z['types'],z['edges'].T,z['schema']);st=M.stats(g,F,names[d],lambda:None);refs[d]=(g,st)
    for r in doc['records']:
        d=r['dataset'];cell=f"{d}_s{r['train_seed']}_{r['arm']}";tag=f"{cell}_u{r['checkpoint']}_ema_0.99_{r['sample_seed']}"
        path=root/'run'/cell/f"u{r['checkpoint']}"/'ema_0.99'/f"sample_{r['sample_seed']}"/'graph.npz' if r['version']=='original' else root/'run/derived_lcc_evaluation'/tag/'lcc.npz'
        z=np.load(path,allow_pickle=False);g=M.make_graph(z['types'],z['edges'],z['schema'],z['new_to_original'] if 'new_to_original' in z else None);st=M.stats(g,F,names[d],lambda:None);rg,rs=refs[d];co=M.compare(g,st,rg,rs,F,names[d]);raw,dist=E.scalar(st,rs,co,names[d]);reference,_=E.scalar(rs,rs,M.compare(rg,rs,rg,rs,F,names[d]),names[d])
        x=dict(identity={k:r[k] for k in ('dataset','train_seed','arm','checkpoint','sample_seed','version')},statistics=st,comparison=co,raw=raw,reference=reference,absolute_error={k:abs(v-reference[k]) if v is not None and reference.get(k) is not None else None for k,v in raw.items()},distribution_error=dist)
        (out/f'{tag}_{r["version"]}.json').write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
    (out/'receipt.json').write_text(json.dumps(dict(records=len(doc['records']),model_forwards=0,training_updates=0,new_chains=0,epoch=time.time()),indent=2))
if __name__=='__main__':main()
