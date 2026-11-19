"""Bounded eight-GPU task DAG; schedules only this frozen panel's own children."""
import os,sys,time,json,subprocess,concurrent.futures,traceback,heapq
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'results/typed_edit_complexity_20261116';SCRIPT=Path(__file__).resolve().parent
START=time.time();DEADLINE=START+24*3600;DS=['imdb','pubmed','freebase','acm'];SEEDS=list(range(810,818));PREFIX=list(range(100,108))
def read(p):return json.loads(p.read_text())
def put(name,obj):
    p=RUN/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2));os.replace(q,p)
def invoke(args,name,gpu=None):
    env=dict(os.environ)
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (RUN/'logs'/f'{name}.log').open('x') as f:
        r=subprocess.run([sys.executable,*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,DEADLINE-time.time()))
    if r.returncode:raise RuntimeError(f'{name} failed with exit {r.returncode}')
def main():
    assert read(RUN/'panel_gate.json')['passed']
    put('launch.json',{'pid':os.getpid(),'started_epoch':START,'deadline_epoch':DEADLINE,'gpu_ids':list(range(8))})
    put('status.json',{'stage':'cache','completed':0,'total':64})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(invoke,[SCRIPT/'run.py','cache','--dataset',ds,'--pair',j],f'cache_{ds}_{j}') for ds in DS for j in range(16)]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):f.result();put('status.json',{'stage':'cache','completed':i,'total':64,'seconds':time.time()-START})
    for ds in DS:invoke([SCRIPT/'run.py','merge','--dataset',ds],f'merge_{ds}')
    pending=[];seen=set();completed=set();requests=[];serial=0;availability={}
    def add(stage,ds,arm,seed,prefix=-1,suffix=False):
        nonlocal serial
        key=(stage,ds,arm,seed,prefix,suffix)
        if key in seen:return
        seen.add(key);serial+=1
        priority=0 if stage=='generate' and arm=='M' and seed==810 and not suffix else 1 if stage=='fit' else 4 if suffix else 3
        heapq.heappush(pending,(priority,serial,key))
        if stage=='generate':
            requests.append({'dataset':ds,'arm':arm,'training_seed':None if arm=='N' else seed,'prefix':prefix,'kind':'suffix' if suffix else 'full',
                'output_path':str(RUN/'outputs'/ds/('suffix' if suffix else 'full')/arm/f'train{seed}'/f'prefix{prefix}')})
    def eligible_suffix(ds):
        for prefix in PREFIX:
            producer=('generate',ds,'M',810,prefix,False)
            if producer not in completed:continue
            p=RUN/'prefixes'/ds/f'seed{prefix}'/'binding.json'
            if not p.exists():availability[f'{ds}/{prefix}']={'available':False,'reason':'no_natural_arrival'};continue
            binding=read(p);good=binding['remaining_steps']>0;availability[f'{ds}/{prefix}']={'available':good,**binding}
            if not good:continue
            add('generate',ds,'N',0,prefix,True)
            for seed in SEEDS:
                for arm in ['M','F','W']:
                    if ('fit',ds,arm,seed,-1,False) in completed:add('generate',ds,arm,seed,prefix,True)
    def execute(key,gpu):
        stage,ds,arm,seed,prefix,suffix=key;args=[SCRIPT/'run.py',stage,'--dataset',ds,'--arm',arm,'--seed',seed]
        if stage=='generate':args+=['--prefix',prefix]
        if suffix:args+=['--suffix']
        invoke(args,f'{stage}_{ds}_{arm}_{seed}_{prefix}_{suffix}',gpu)
    for ds in DS:
        add('fit',ds,'M',810)
    for seed in SEEDS:
        for ds in DS:
            for arm in ['M','F','W']:add('fit',ds,arm,seed)
    for ds in DS:
        for prefix in PREFIX:add('generate',ds,'N',0,prefix)
    running={};free=list(range(8))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        while pending or running:
            if time.time()>DEADLINE:raise TimeoutError('24-hour fixed panel deadline')
            while pending and free:
                _,_,key=heapq.heappop(pending);gpu=free.pop(0);running[pool.submit(execute,key,gpu)]=(key,gpu)
            done,_=concurrent.futures.wait(running,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                key,gpu=running.pop(f);f.result();completed.add(key);free.append(gpu)
                stage,ds,arm,seed,prefix,suffix=key
                if stage=='fit':
                    for prefix in PREFIX:add('generate',ds,arm,seed,prefix)
                eligible_suffix(ds)
            put('generation_ledger.json',{'requests':requests,'prefix_availability':availability})
            put('status.json',{'stage':'fit_and_generation','fits_completed':sum(k[0]=='fit' for k in completed),
                'full_completed':sum(k[0]=='generate' and not k[-1] for k in completed),
                'suffix_completed':sum(k[0]=='generate' and k[-1] for k in completed),
                'pending':len(pending),'running':[{'key':v[0],'gpu':v[1]} for v in running.values()],'seconds':time.time()-START})
    assert sum(k[0]=='fit' for k in completed)==96 and sum(k[0]=='generate' and not k[-1] for k in completed)==800
    put('status.json',{'stage':'independent_acceptance','seconds':time.time()-START})
    invoke([SCRIPT/'accept.py','training'],'accept_training');invoke([SCRIPT/'accept.py','generation'],'accept_generation')
    invoke([SCRIPT/'finish.py'],'finish_tables_and_inserts')
    put('completion.json',{'completed':True,'fits':96,'formal_updates':451200,'wall_seconds':time.time()-START,'generation':read(RUN/'generation_acceptance.json')})
    put('status.json',{'stage':'completed','wall_seconds':time.time()-START})
try:main()
except BaseException as e:
    put('failure_controller.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'wall_seconds':time.time()-START})
    put('status.json',{'stage':'failed','message':str(e)});raise
