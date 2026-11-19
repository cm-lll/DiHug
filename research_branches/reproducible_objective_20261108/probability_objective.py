"""CPU reevaluation of frozen PS objective on exported probabilities; no network."""
import os,sys,json,time,resource,importlib.util
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(sys.argv[1]).resolve()
spec=importlib.util.spec_from_file_location('old_worker_cpu',ROOT/'results/output_boundary_replay_20261108T081654/worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
w.torch.set_num_threads(1)
start=time.time();rows=[]
tasks=w.read(ROOT/'results/output_boundary_replay_20261108T081654/tasks.json')
probes=w.read(OUT/'probes.json')
for task in tasks:
    ds,cfg=task['dataset'],task['row']['config']
    path=ROOT/w.g.SCI.relative_to(w.g.ROOT)/'package'/cfg['datasets'][ds]['file']
    with w.np.load(path,allow_pickle=False) as z:case=w.g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
    obj=w.make_objective(case,cfg,ds,'PS')
    fixed=w.torch.load(OUT/f'{ds}_fixed_inputs.pt',map_location='cpu')['inputs']
    for row in probes:
        if row['dataset']!=ds or row['offset'] not in [0,32]:continue
        inp=next(inp for panel,t,seed,inp in fixed if panel==row['panel'] and t==row['t'])
        file=OUT/f"{ds}_{row['arm']}_{row['offset']}_{row['readout']}_{row['panel']}_{row['t']}.npz"
        with w.np.load(file,allow_pickle=False) as z:p=w.torch.from_numpy(z['probability'])
        with w.torch.no_grad():value,detail=obj(p,inp['query'],inp['f'],inp,logits=None)
        rows.append(dict(**{k:row[k] for k in ['dataset','seed','arm','offset','readout','panel','t']},PS=float(value),unweighted_CE=row['CE'],CE_plus_PS=row['CE']+float(value)))
assert not w.torch.cuda.is_initialized()
w.fr.save(OUT/'probability_objective_cpu.json',dict(rows=rows,seconds=time.time()-start,max_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,model_calls=0,cuda_initialized=False))
print(json.dumps(dict(rows=len(rows),seconds=time.time()-start)))
