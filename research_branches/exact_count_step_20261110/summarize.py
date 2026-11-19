"""CPU-only three-way paired report; no new inference or optimization."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,csv,time,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from exact_count import exact
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/exact_count_step_20261110_attempt2';OLD=ROOT/'results/hard_clean_zero_safe_20261110'
torch.set_num_threads(1)
def write(name,rows):
    if not rows:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def identical(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    return a==b
def main():
    start=time.time();protocol=json.loads((OUT/'protocol.json').read_text());receipt=json.loads((OUT/'receipt.json').read_text())
    rows=[];samples=[];relations=[];aggregate=[];audits=[];costnames=['relation_count','joint_degree','present_triangles','absent_triangles','components']
    for task in protocol['tasks']:
        name=task['task'];folder=OUT/name
        if not (folder/'result.json').exists():continue
        info=json.loads((folder/'result.json').read_text());oldinfo=json.loads((OLD/name/'result.json').read_text())
        oldstate=torch.load(OLD/name/'temporary_complete.pt',map_location='cpu',weights_only=False)
        newstate=torch.load(folder/'temporary_complete.pt',map_location='cpu',weights_only=False)
        assert identical(oldstate['state']['rng'],newstate['state']['rng'])
        assert identical(oldstate['auxiliary_rng'],newstate['auxiliary_rng'])
        assert info['main_rng_sha256']==oldinfo['main_rng_sha256'] and info['input_sha256']==oldinfo['input_sha256']
        assert newstate['state']['updates']==newstate['state']['ema']['updates']==oldstate['state']['updates']
        del oldstate,newstate
        inputs=torch.load(task['evaluation_file'],map_location='cpu',weights_only=False)['inputs'];order=np.argsort(inputs[0]['ids']);q=inputs[0]['query'][:,order];f=inputs[0]['f'][order]
        definition=json.loads((ROOT/'results/hard_clean_pilot_v2_20261110'/name/'score_definition.json').read_text());target=np.array(definition['reference']['counts'])
        n=len(definition['reference']['degree'])
        reports={label:json.loads((path/(stage+'.json')).read_text()) for label,path,stage in [('before',folder,'before'),('old_update',OLD/name,'after'),('exact_update',folder,'after')]}
        max_before_p=0.;before_bitdiff=0
        for j,t in enumerate(protocol['times']):
            oldbase=np.load(OLD/name/f'before_t{t}.npz');newbase=np.load(folder/f'before_t{t}.npz')
            max_before_p=max(max_before_p,float(abs(oldbase['probability']-newbase['probability']).max()));before_bitdiff+=int(np.count_nonzero(oldbase['bits']!=newbase['bits']))
            for label,path,stage in [('before',folder,'before'),('old_update',OLD/name,'after'),('exact_update',folder,'after')]:
                data=np.load(path/f'{stage}_t{t}.npz');b=data['bits'];p=data['probability'];uniform=np.random.default_rng(task['seed']+810001+100+j).random(b.shape)
                assert np.array_equal(uniform<p,b)
                er=reports[label][j];assert er['input_sha256']==reports['before'][j]['input_sha256'] and er['t']==t
                records=[]
                for k,bits in enumerate(b):
                    a=np.zeros((n,n),dtype=np.uint8);u,v=q[:,bits.astype(bool)];a[u,v]=1;a[v,u]=1
                    degree=a.sum(1);nc,comp=connected_components(csr_matrix(a),directed=False)
                    stat=dict(edges=int(bits.sum()),triangles=int(((a@a)*a).sum()//6),components=int(nc),LCC=float(np.bincount(comp).max()/n),isolates=int((degree==0).sum()),hub=int(degree.max()))
                    samples.append(dict(task=name,dataset=task['dataset'],seed=task['seed'],state=label,t=t,sample=k,**stat));records.append(stat)
                item=dict(task=name,dataset=task['dataset'],seed=task['seed'],state=label,t=t,samples=len(b),
                    CE=er['CE'],positive_CE=er['positive_CE'],negative_CE=er['negative_CE'],exact_count_risk=float(exact(p,f,target)[0].sum()))
                for stat in records[0]:
                    vals=[x[stat] for x in records];item[stat]=float(np.mean(vals));item[stat+'_std']=float(np.std(vals,ddof=1))
                for k,cn in enumerate(costnames):item['cost_'+cn]=float(data['costs'][:,k].mean())
                rows.append(item)
                for rel in er['relations']:relations.append(dict(task=name,state=label,t=t,**rel))
        result=dict(task=name,dataset=task['dataset'],seed=task['seed'],update=info['update'],training_t=info['t'],
            old_gradient_norm=oldinfo['new_gradient_norm'],new_gradient_norm=info['new_gradient_norm'],
            old_count_parameter_gradient_norm=oldinfo['individual_norms'][0],new_count_parameter_gradient_norm=info['individual_norms'][0],
            old_AdamW_norm=oldinfo['actual_AdamW_norm'],new_AdamW_norm=info['actual_AdamW_norm'],
            old_coefficients=oldinfo['coefficients'],new_coefficients=info['coefficients'],states={})
        for label in reports:
            rr=[r for r in rows if r['task']==name and r['state']==label]
            result['states'][label]={key:float(np.mean([r[key] for r in rr])) for key in ('edges','triangles','components','LCC','isolates','hub','CE','positive_CE','negative_CE','exact_count_risk',*[f'cost_{c}' for c in costnames])}
        result['worse_than_before_times']={metric:sum(a[metric]>b[metric]+1e-12 for a,b in zip([r for r in rows if r['task']==name and r['state']=='exact_update'],[r for r in rows if r['task']==name and r['state']=='before'])) for metric in ['cost_joint_degree','cost_present_triangles','cost_components','positive_CE']}
        aggregate.append(result);audits.append(dict(task=name,input_RNG_and_auxiliary_RNG_identical=True,EMA_counter_correct=True,save_load_exact=info['save_reload_boundary_exact'],
            before_probability_recompute_max_abs=max_before_p,before_hard_bit_differences=before_bitdiff,ordinary_GPU_bitwise_recomputation_not_claimed=True))
    write('paired_by_time.csv',rows);write('graphs.csv',samples);write('relations.csv',relations)
    oldlease=json.loads((ROOT/'results/exact_count_step_20261110/supervisor.json').read_text());lease=json.loads((OUT/'supervisor.json').read_text())
    result=dict(status=receipt['status'],conditions=aggregate,audit=audits,actual_calls={k:receipt[k] for k in ('forwards','backwards','optimizer_updates')},
        total_GPU_seconds=oldlease['occupied_seconds']+lease['occupied_seconds'],total_GPU_hours=(oldlease['occupied_seconds']+lease['occupied_seconds'])/3600,
        peak_GPU_reserved_MiB=receipt['peak_reserved_MiB'],first_attempt_forwards=0,reverse_chains=0,formal_training=False,
        accounting_budget=json.loads((OUT/'budget.json').read_text()),CPU_summary_seconds=time.time()-start)
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(aggregate,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
