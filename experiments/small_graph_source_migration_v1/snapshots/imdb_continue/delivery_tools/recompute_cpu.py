"""Portable CPU-only recomputation from delivered graphs and frozen pure metrics.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python recompute_cpu.py --root . --output new_cpu_recompute
No network/model/torch import, dependency installation or original server paths.
"""
import argparse,pathlib,importlib.util,json,time
import numpy as np
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve();out=pathlib.Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 source=root/'frozen_evaluation_source';M=module(source/'lcc_eval/metrics.py','frozen_cpu_metric');fields=module(source/'evaluation_fields.py','frozen_cpu_fields')
 v=source/'lcc_eval/vendor';F=M.frozen(v/'frozen_metrics.py',v/'aligned_evaluate.py',v/'hgen_utils.py');names=['Director','Actor','Movie','Genre']
 z=np.load(root/'reference/IMDB.npz',allow_pickle=False);ref=M.make_graph(z['types'],z['edges'].T,z['schema']);rs=M.stats(ref,F,names,lambda:None)
 plan=json.loads((root/'evaluation.json').read_text());count=0
 for old in plan['records']:
  cell=f"IMDB_s{old['train_seed']}_{old['arm']}";tag=f"{cell}_u{old['checkpoint']}_ema_0.99_{old['sample_seed']}"
  if old['version']=='original':path=root/'run'/cell/f"u{old['checkpoint']}"/'ema_0.99'/f"sample_{old['sample_seed']}"/'graph.npz'
  else:path=root/'run/derived_lcc_evaluation'/tag/'lcc.npz'
  z=np.load(path,allow_pickle=False);g=M.make_graph(z['types'],z['edges'],z['schema'],z['new_to_original'] if 'new_to_original' in z else None)
  st=M.stats(g,F,names,lambda:None);comp=M.compare(g,st,ref,rs,F,names);raw,dist=fields.scalar(st,rs,comp,names)
  original_ref,_=fields.scalar(rs,rs,M.compare(ref,rs,ref,rs,F,names),names)
  val=dict(identity={k:old[k] for k in ('dataset','train_seed','arm','checkpoint','sample_seed','version')},statistics=st,comparison=comp,raw=raw,reference=original_ref,
   absolute_error={k:None if v is None or original_ref.get(k) is None else abs(v-original_ref[k]) for k,v in raw.items()},distribution_error=dist)
  (out/f"{tag}_{old['version']}.json").write_text(json.dumps(val,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()));count+=1
 (out/'receipt.json').write_text(json.dumps(dict(records=count,model_forwards=0,training_updates=0,epoch=time.time()),indent=2));print('CPU records recomputed',count)
if __name__=='__main__':main()

