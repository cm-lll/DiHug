"""Read saved probabilities/logs only. No model, sampling, fitting or autograd."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,csv,time,hashlib,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'results/single_step_formal_20261111'
RUN=ROOT/'results/single_step_formal_resume_20261111'
OUT=ROOT/'results/single_step_failure_audit_20261111'
OUT.mkdir(exist_ok=True)
start=time.time();proto=json.loads((RUN/'protocol.json').read_text())
validation=[];relations=[];training=[];trajectory=[];checks=[];bindings={}
def read(p):
    raw=p.read_bytes();bindings[str(p.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)
def jsonlines(p):
    raw=p.read_bytes();bindings[str(p.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return [json.loads(v) for v in raw.splitlines()]
def save(name,rows):
    (OUT/(name+'.json')).write_text(json.dumps(rows,indent=2,allow_nan=False))
    if isinstance(rows,list) and rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (OUT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
for task in proto['tasks']:
    ident=dict(task=task['id'],dataset=task['parent']['dataset'],seed=task['parent']['seed'],B=task['B'],K=task['K'])
    initial={}
    for u,base in [(2000,OLD),(2500,RUN),(3000,RUN)]:
        d=base/task['id'];v=read(d/f'validation_u{u}.json')
        for r in v:
            key=(r['readout'],r['t']);f=d/f"validation_u{u}_{key[0]}_t{key[1]}.npz"
            bindings[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
            with np.load(f) as z:p=z['probability'];y=z['labels'];fam=z['families'];ids=z['ids']
            if u==2000:initial[key]=(r['input_hash'],y.copy(),fam.copy(),ids.copy())
            old=initial[key];assert r['input_hash']==old[0] and np.array_equal(y,old[1]) and np.array_equal(fam,old[2]) and np.array_equal(ids,old[3])
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            pos=y==1;neg=~pos
            pc=[(a['positive_n'],a['positive_CE']) for a in r['relations'] if a['positive_n']]
            nc=[(a['negative_n'],a['negative_CE']) for a in r['relations'] if a['negative_n']]
            with np.errstate(divide='ignore'):logempty=float(np.log1p(-p).sum())
            vrow=dict(**ident,update=u,readout=key[0],t=key[1],slots=len(p),reference_edges=int(y.sum()),expected_edges=float(p.sum()),
                positive_mean_probability=float(p[pos].mean()),negative_mean_probability=float(p[neg].mean()),
                expected_false_negative=float((1-p[pos]).sum()),expected_false_positive=float(p[neg].sum()),
                zero_prob_slots=int((p==0).sum()),one_prob_slots=int((p==1).sum()),below_1e8_slots=int((p<1e-8).sum()),
                mean_sigmoid_derivative=float((p*(1-p)).mean()),empty_clean_graph_probability=float(np.exp(logempty)),
                CE=r['CE'],positive_CE=sum(n*c for n,c in pc)/sum(n for n,c in pc),negative_CE=sum(n*c for n,c in nc)/sum(n for n,c in nc))
            validation.append(vrow)
            for a in r['relations']:
                f=a['family'];mask=fam==f
                relations.append(dict(**ident,update=u,readout=key[0],t=key[1],**a,expected_edges=float(p[mask].sum()),reference_edges=int(y[mask].sum())))
    checks.append(dict(task=task['id'],matched_validation_inputs=36,finite_probabilities=True))
    rows=jsonlines(RUN/task['id']/'training.jsonl')
    for lo in range(2001,3001,100):
        x=[r for r in rows if lo<=r['update']<lo+100];assert len(x)==100
        risks=np.array([[s['risks'] for s in r['records']] for r in x]);ce=np.array([[s['CE'] for s in r['records']] for r in x]);gr=np.array([r['gradient_norm'] for r in x])
        training.append(dict(**ident,start=lo,end=lo+99,CE_mean=float(ce.mean()),gradient_median=float(np.median(gr)),gradient_max=float(gr.max()),gradient_below_1e8=int((gr<1e-8).sum()),
            **{name+'_mean':float(risks[:,:,i].mean()) for i,name in enumerate(['count','joint','present_tri','absent_tri','components'])},objective_mean=float(risks.mean())))
    for s in proto['sample_seeds']:
        for r in jsonlines(RUN/task['id']/f'sample_{s}'/'trajectory.jsonl'):trajectory.append(dict(**ident,sample_seed=s,**r))
save('validation',validation);save('relations',relations);save('training_windows',training);save('trajectories',trajectory);save('checks',checks)
save('receipt',dict(seconds=time.time()-start,model_calls=0,backwards=0,new_graphs=0,GPU_hours=0,validation_rows=len(validation),source_bindings=bindings))
for task in proto['tasks']:
    print('\n',task['id'])
    for u in (2000,2500,3000):
        x=[r for r in validation if r['task']==task['id'] and r['update']==u and r['readout']=='EMA']
        raw=[r for r in validation if r['task']==task['id'] and r['update']==u and r['readout']=='raw']
        print(u,'EMA mu',[round(r['expected_edges'],5) for r in x], 'raw mu',[round(r['expected_edges'],5) for r in raw], 'EMA exactzero',[r['zero_prob_slots'] for r in x])
    x=[r for r in trajectory if r['task']==task['id']]
    print('trajectory',[(t,round(statistics.mean(r['total_edges'] for r in x if r['input_t']==t),2)) for t in (1000,750,500,250,100,50,10,1)])
    for lo in (2001,2101,2401,2901):
        r=next(r for r in training if r['task']==task['id'] and r['start']==lo)
        print('window',lo,'risks',[round(r[k+'_mean'],5) for k in ('count','joint','present_tri','absent_tri','components')],'gradient_med/max',r['gradient_median'],r['gradient_max'])
