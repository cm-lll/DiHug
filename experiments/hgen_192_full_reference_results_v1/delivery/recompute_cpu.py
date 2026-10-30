"""Portable delivered-graph recomputation. No model import, training or new graph.

python recompute_cpu.py --root EXTRACTED_PACKAGE --output NEW_DIRECTORY
"""
import argparse,pathlib,json,importlib.util,time
import numpy as np
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve();out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 src=root/'frozen_evaluation_source';M=module(src/'lcc_eval/metrics.py','delivered_metric');E=module(src/'evaluation_fields.py','delivered_fields');v=src/'lcc_eval/vendor';F=M.frozen(v/'frozen_metrics.py',v/'aligned_evaluate.py',v/'hgen_utils.py');names={'IMDB':['Director','Actor','Movie','Genre'],'PubMed':['Gene','Disease','Chemical','Species']};refs={}
 doc=json.loads((root/'evaluation.json').read_text());start=time.time()
 for d in ('IMDB','PubMed'):
  z=np.load(root/'reference'/(d+'.npz'));g=M.make_graph(z['types'],z['edges'].T,z['schema']);st=M.stats(g,F,names[d],lambda:None);refs[d]=(g,st)
 for r in doc['records']:
  d,s,ss,version=(r[k] for k in ('dataset','train_seed','sample_seed','version'));tag=f'{d}_s{s}_{ss}';path=root/'run'/f'{d}_s{s}'/f'sample_{ss}'/'graph.npz' if version=='original' else root/'derived'/tag/'lcc.npz'
  z=np.load(path);g=M.make_graph(z['types'],z['edges'],z['schema'],z['new_to_original'] if 'new_to_original' in z else None);st=M.stats(g,F,names[d],lambda:None);rg,rs=refs[d];co=M.compare(g,st,rg,rs,F,names[d]);raw,dist=E.scalar(st,rs,co,names[d]);rv,_=E.scalar(rs,rs,M.compare(rg,rs,rg,rs,F,names[d]),names[d]);val=dict(raw=raw,reference=rv,absolute_error={k:abs(x-rv[k]) if x is not None and rv.get(k) is not None else None for k,x in raw.items()},distribution_error=dist,statistics=st,comparison=co)
  (out/(tag+'_'+version+'.json')).write_text(json.dumps(val,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
 (out/'receipt.json').write_text(json.dumps(dict(records=len(doc['records']),seconds=time.time()-start,model_calls=0,training=0,new_graphs=0),indent=2))
if __name__=='__main__':main()
