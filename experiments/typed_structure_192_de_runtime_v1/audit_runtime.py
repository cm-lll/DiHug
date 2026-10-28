"""Static semantic/byte audit only. No model construction or backend access."""
import argparse
import ast
import hashlib
import json
import pathlib

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--current',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    old=pathlib.Path(a.baseline);new=pathlib.Path(a.current)
    allowed={'admission.py','launch.py','worker.py','experiment_contract.py'};same={};changed={}
    for path in (old/'package').rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:continue
        relative=path.relative_to(old/'package');target=new/'package'/relative
        assert target.is_file(),str(relative)
        row=dict(old=sha(path),new=sha(target));row['equal']=row['old']==row['new']
        if not row['equal']:
            assert relative.as_posix() in allowed,'unapproved scientific source change: '+str(relative)
            changed[relative.as_posix()]=row
        else:same[relative.as_posix()]=row
    dictionary={}
    for path in (old/'dictionaries').rglob('*'):
        if path.is_file():
            relative=path.relative_to(old);assert sha(path)==sha(new/relative);dictionary[str(relative)]=sha(path)
    assert sha(old/'proposal.json')==sha(new/'proposal.json'),'frozen protocol changed'
    unchanged_functions={}
    for filename,names in [('experiment_contract.py',['read_dictionary','initialization','reset_training_streams']),('worker.py',['atomic_checkpoint','validation']),('launch.py',['resource_check'])]:
        left=functions(old/'package'/filename);right=functions(new/'package'/filename)
        for name in names:
            assert left[name]==right[name],filename+':'+name;unchanged_functions[filename+':'+name]=True
    left=ast.parse((old/'package/worker.py').read_text());right=ast.parse((new/'package/worker.py').read_text())
    for target in ['position','t']:
        find=lambda tree:[ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id==target]
        assert find(left)==find(right) and find(left),'scientific training/reverse loop differs'
        unchanged_functions['worker:for '+target]=True
    helper=functions(new/'summary_metrics.py');reference=functions(old/'reference/summarize_oabc.py')
    for name in ['scalars','finite','reason','graph_summary','strict_mean','paired']:
        assert helper[name]==reference[name],name
        unchanged_functions['metric_helper:'+name]=True
    out=dict(status='passed',baseline_commit='89922be56927412dbdb2b5f9e6f36f8264f027d4',unchanged_files=same,runtime_only_changes=changed,dictionary_sha256=dictionary,protocol_sha256=sha(new/'proposal.json'),unchanged_semantics=unchanged_functions,limits='static equivalence and prior engineering evidence reused; no new network/GPU verification')
    with pathlib.Path(a.output).open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(dict(status='passed',unchanged_files=len(same),runtime_changed=len(changed),dictionary_files=len(dictionary),semantic_checks=len(unchanged_functions))))
if __name__=='__main__':main()
