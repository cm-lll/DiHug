"""Execute actual save/milestone and summary code without model construction."""
import ast, copy, hashlib, inspect, json, os, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import torch
torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'research_branches/single_step_formal_20261111'
RESULT=ROOT/'results/single_step_formal_resume_20261111'
RESULT.mkdir(exist_ok=True)
source=(HERE/'worker.py').read_text();tree=ast.parse(source)
old_tree=ast.parse((OLD/'worker.py').read_text())
results=[]
def record(name):results.append(dict(test=name,passed=True))
def fn(name,t=tree):return next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
for name in ('inputs','setup','update','snapshot','restoring_weights','evaluation','sample'):
    assert ast.dump(fn(name),include_attributes=False)==ast.dump(fn(name,old_tree),include_attributes=False),name
record('scientific_input_update_loss_evaluation_sampling_functions_unchanged')
with tempfile.TemporaryDirectory(prefix='dihug_resume_cpu_') as tmp:
    tmp=Path(tmp);(tmp/'protocol.json').write_text('{}')
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def save(p,x):Path(p).write_text(json.dumps(x))
    state={'model':{'weight':torch.tensor([1.,2.])},'optimizer':{'state':{0:{'step':torch.tensor(2500.),'exp_avg':torch.tensor([.1,.2]),'exp_avg_sq':torch.tensor([.01,.04])}}},'ema':{'updates':2500,'shadow':{'weight':torch.tensor([.9,1.9])}},'rng':{'torch':torch.get_rng_state()},'updates':2500}
    class Boundary:
        completed=2500
        def checkpoint(self,binding):return {**copy.deepcopy(state),'binding':binding}
    rt=SimpleNamespace(checkpoint=lambda path,obj:torch.save(obj,path),save=save,sha=sha)
    ns=dict(rt=rt,OUT=tmp)
    exec(compile(ast.Module(body=[fn('snapshot')],type_ignores=[]),'<actual snapshot>','exec'),ns)
    sig=inspect.signature(ns['snapshot'])
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='snapshot']
    for n in calls:sig.bind(*([None]*len(n.args)),**{k.arg:None for k in n.keywords})
    record('all_snapshot_call_sites_match_actual_signature')
    old_bad=next(n for n in ast.walk(old_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='snapshot' and len(n.args)==6)
    try:sig.bind(*([None]*len(old_bad.args)))
    except TypeError:pass
    else:raise AssertionError('original failure not reproduced')
    record('original_milestone_missing_argument_reproduced')
    condition=next(n for n in ast.walk(fn('main')) if isinstance(n,ast.If) and ast.unparse(n.test)=='u in (2500, 3000)')
    observed=[]
    ns.update(folder=tmp,z={'binding':{'test':True}},b=Boundary(),row={'dataset':'fixture','seed':1},B=4,K=16,cfg=None,case=None,m=None,o=None,e=None,rng=None,evaluation=lambda *args:observed.append(args[1]))
    for u in (2500,3000):
        ns['u']=u;ns['b'].completed=u;state['updates']=u;state['ema']['updates']=u
        exec(compile(ast.Module(body=[condition],type_ignores=[]),'<actual milestone>','exec'),ns)
        f=tmp/f'checkpoint_u{u}.pt';payload=torch.load(f,map_location='cpu',weights_only=False)
        assert payload['state']['updates']==payload['state']['ema']['updates']==u
        assert torch.equal(payload['state']['model']['weight'],state['model']['weight'])
        assert torch.equal(payload['state']['rng']['torch'],state['rng']['torch'])
        assert json.loads(f.with_suffix('.sha256.json').read_text())['sha256']==sha(f)
    assert observed==['u2500','u3000']
    record('actual_u2500_u3000_save_and_evaluation_dispatch')
    ns['snapshot'](tmp,'stopped_complete.pt',ns['z'],ns['b'],ns['row'],4,16)
    assert (tmp/'stopped_complete.sha256.json').exists()
    record('exception_save_path_still_works')
    proto=json.loads((ROOT/'results/single_step_formal_20261111/protocol.json').read_text())
    for mode in ('missing','failed','complete','wrong_identity'):
        out=tmp/mode;out.mkdir();(out/'protocol.json').write_text(json.dumps(proto))
        for task in proto['tasks']:
            d=out/task['id'];d.mkdir()
            if mode=='missing':continue
            receipt=dict(task='wrong' if mode=='wrong_identity' else task['id'],status='completed' if mode=='complete' else 'failed')
            (d/'receipt.json').write_text(json.dumps(receipt))
            if mode=='complete':
                rows=[]
                for ss in proto['sample_seeds']:
                    for view in ('original','LCC'):rows.append(dict(dataset=task['parent']['dataset'],train_seed=task['parent']['seed'],arm=task['id'],B=task['B'],K=task['K'],sample_seed=ss,view=view,edge_error=1.,degree_W1=.2,joint_TV=.3,triangle_error=2.,MP3_error=3.,lcc=192))
                (d/'per_graph.json').write_text(json.dumps(rows))
        proc=subprocess.run([sys.executable,str(HERE/'finalize.py'),'--out',str(out)],capture_output=True,text=True,timeout=30)
        if mode=='wrong_identity':assert proc.returncode!=0 and 'task identity mismatch' in proc.stderr
        else:
            assert proc.returncode==0,proc.stderr
            completion=json.loads((out/'completion.json').read_text())
            assert completion['complete_tasks']==(8 if mode=='complete' else 0)
            assert completion['graphs']==(32 if mode=='complete' else 0)
            assert len(json.loads((out/'task_status.json').read_text()))==8
        record('finalizer_'+mode)
    # Check resume loop starts after the saved completed boundary, not at 2000.
    loops=[ast.unparse(n.iter) for n in ast.walk(fn('main')) if isinstance(n,ast.For)]
    assert 'range(2501, 3001)' in loops and 'range(2001, 3001)' not in loops
    record('resume_range_preserves_first_500_updates')
(RESULT/'cpu_repair_tests.json').write_text(json.dumps(dict(tests=results,passed=len(results),GPU_hours=0,model_calls=0,limitations='synthetic save and dispatcher tests, not GPU restore or replay'),indent=2))
print(json.dumps(results,indent=2))
