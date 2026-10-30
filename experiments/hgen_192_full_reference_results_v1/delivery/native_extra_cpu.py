"""Run an already-existing author metric using the already-installed package.

Separate supplement: never substitutes values in the frozen common evaluator.
"""
import ast,contextlib,io,json,pathlib,math,time,warnings,hashlib
import numpy as np
import powerlaw
ROOT=pathlib.Path(__file__).resolve().parent
SOURCE=pathlib.Path('/data2/lyh/gnn_project/DiHug/research_branches/hgen_192_full_reference/experiments/hgen_192_full_reference_v1/vendor/hgen/utils.py')
def run():
 node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='statistics_power_law_alpha')
 tree=ast.Module(body=[node]);env=dict(np=np,powerlaw=powerlaw)
 if hasattr(ast,'TypeIgnore'):tree.type_ignores=[]
 exec(compile(tree,str(SOURCE),'exec'),env)
 doc=json.loads((ROOT/'evaluation.json').read_text());rows=[]
 for r in [dict(dataset=d,train_seed=None,sample_seed=None,version='reference') for d in ('IMDB','PubMed')]+doc['records']:
  d,s,ss,v=(r[k] for k in ('dataset','train_seed','sample_seed','version'))
  path=ROOT/'data'/d/'reference.npz' if v=='reference' else ROOT/'run'/f'{d}_s{s}'/f'sample_{ss}'/'graph.npz' if v=='original' else ROOT/'derived'/f'{d}_s{s}_{ss}'/'lcc.npz'
  z=np.load(path);es=z['edges'].T if v=='reference' else z['edges'];a=np.zeros((len(z['types']),len(z['types'])),np.int64);a[es[0],es[1]]=1;a[es[1],es[0]]=1
  out=io.StringIO();result=None;reason=None;tic=time.time()
  with warnings.catch_warnings(record=True) as caught,contextlib.redirect_stdout(out),contextlib.redirect_stderr(out):
   warnings.simplefilter('always')
   try:
    result=float(env['statistics_power_law_alpha'](a))
    if not math.isfinite(result):result=None;reason='nonfinite_native_fit'
   except Exception as e:reason=repr(e)
  rows.append(dict(dataset=d,train_seed=s,sample_seed=ss,version=v,value=result,reason=reason,seconds=time.time()-tic,warnings=[str(w.message) for w in caught],output=out.getvalue(),graph_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 receipt=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),function='statistics_power_law_alpha',formula='powerlaw.Fit(degrees,xmin=max(min(degrees),1)).power_law.alpha; all other library defaults unchanged',package_version=getattr(powerlaw,'__version__','unknown'),is_separate_supplement=True,common_frozen_metric_replaced=False,rows=rows)
 (ROOT/'native_HGEN_powerlaw_supplement.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
 print(len(rows),'native author powerlaw rows',sum(r['value'] is None for r in rows),'undefined/failed')
if __name__=='__main__':run()
