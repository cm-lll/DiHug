"""Read-only child-process imports of all three actual worker entry modules."""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE='175373c83202b6c8b80aa570e0092c4652031ce6'
SCI=ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1'

def verify_sources():
    rows=[]
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',SOURCE],cwd=ROOT,text=True).splitlines()
    for name in paths:
        if not name.endswith('.py'):continue
        expected=subprocess.check_output(['git','show',SOURCE+':'+name],cwd=ROOT)
        data=(ROOT/name).read_bytes();assert expected==data,name
        rows.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),matches_fixed_source=True))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    result=dict(source=SOURCE,files=verify_sources(),imports=[],GPU_used=False)
    child='''
import hashlib,importlib.util,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);name=sys.argv[2];sci=root/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1'
entry=root/'experiments'/name/('generation_worker.py' if name=='d_late_ard_runtime_v1' else 'worker_run.py')
sys.path[:0]=[str(entry.parent),str(sci/'package'),str(sci)]
spec=importlib.util.spec_from_file_location('checked_worker',entry);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import torch,boundary_transactions,integration
rows={}
for key in ('common_run','single_input','sampling','model_factory','boundary_transactions','integration','model_variants'):
 p=pathlib.Path(sys.modules[key].__file__).resolve();assert p.is_relative_to(root)
 rows[key]=dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert m.TrainingBoundary is boundary_transactions.TrainingBoundary
assert m.ema_validation is integration.ema_validation
assert not torch.cuda.is_initialized()
print(json.dumps(dict(worker=name,entry=str(entry.relative_to(root)),modules=rows,CUDA_initialized=False)))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',DIHUG_O_CN_RUN=str(ROOT/'results/o_cn_matched_formal_binding_only'))
    for name in ('o_cn_matched_execute_v1','late_final_single_input_v1','d_late_ard_runtime_v1'):
        p=subprocess.run([sys.executable,'-c',child,str(ROOT),name],cwd=ROOT,env=env,text=True,capture_output=True)
        assert p.returncode==0,p.stderr
        result['imports'].append(json.loads(p.stdout))
    result['status']='passed'
    path=pathlib.Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print('passed:',len(result['files']),'fixed Python files;',len(result['imports']),'separate worker imports')

if __name__=='__main__':main()
