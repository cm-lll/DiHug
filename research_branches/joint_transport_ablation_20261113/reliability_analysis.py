"""CPU-only scoring and finite-panel gradient reliability; no model imports.

Disjoint groups are diagnostic samples from each frozen model. They are not
independent training seeds. Stable eager directions do not imply good updates.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,csv,traceback,itertools,resource,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import heat_increment_cpu as inc
from increment_reward_audit import metrics as organization_metrics
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research_branches/new_datasets_20261112'))
import metrics as M
OLD=ROOT/'research_branches/seed_coverage_repair_release/experiments/seed_coverage_repair_v1/lcc_eval'
F=M.frozen(OLD/'vendor/frozen_metrics.py',OLD/'vendor/aligned_evaluate.py',OLD/'vendor/hgen_utils.py')

def write(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)
def table(path,rows):
    write(path.with_suffix('.json'),rows)
    if rows:
        with path.with_suffix('.csv').open('w') as f:
            w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def coefficient(cost,ix):
    out=np.zeros(len(cost),float);x=np.asarray(cost)[ix];k=len(ix)
    out[ix]=.01*(x-x.mean())/(k-1)
    assert abs(out.sum())<1e-12
    return out
def product(a,b,G):return float(a@G@b)
def norm(a,G):
    x=product(a,a,G)
    assert x>=-1e-8*max(1.,np.linalg.norm(G)*np.dot(a,a)),x
    return float(np.sqrt(max(0.,x)))
def cosine(a,b,G):
    den=norm(a,G)*norm(b,G)
    return float(np.clip(product(a,b,G)/den,-1,1)) if den>1e-20 else None

def main():
    out=Path(sys.argv[1]);start=time.time();p=json.loads((out/'protocol.json').read_text())
    ev=out/'evaluation';ev.mkdir(exist_ok=True);refs={};scores={};done={};failures={}
    def check():
        if time.time()>p['delivery_epoch']-60:raise TimeoutError('delivery reserve')
    inventory=[]
    for row in p['models']:
        ds=row['dataset'];names=row['config']['datasets'][ds]['type_names']
        if ds not in refs:
            z=np.load(row['data']);e=z['edges'];e=e if e.shape[0]==2 else e.T
            ref=M.make_graph(z['types'],e,z['schema']);rs=M.stats(ref,F,names,check)
            refs[ds]=(ref,rs,names);scores[ds]=inc.assets.load(ds,row)
        for plan in row['reused']+row['new_trajectories']:
            old=plan['index']<6;directory=Path(plan['path']) if old else out/row['id']/f'chain_{plan["index"]:02d}_{plan["seed"]}'
            inventory.append(dict(model=row['id'],dataset=ds,train_seed=row['train_seed'],index=plan['index'],sample_seed=plan.get('sample_seed',plan.get('seed')),
                                  source='reused6' if old else 'new10',path=str(directory)))
    write(out/'analysis_inventory.json',inventory);write(out/'reference_metrics.json',{ds:v[1] for ds,v in refs.items()})
    def one(item):
        check();ds=item['dataset'];sc=scores[ds];ref,rs,names=refs[ds];directory=Path(item['path'])
        summary=json.loads((directory/'summary.json').read_text());gp=directory/'graph.npz'
        assert summary['status']=='complete' and summary['graph_sha256']==inc.assets.sha(gp)
        z=np.load(gp);assert np.array_equal(z['types'],sc.types) and np.array_equal(z['schema'],ref['schema'])
        b=inc.assets.bits(sc,z);ms,h,d=inc.matrices(sc,b);cost,parts,_=inc.solved(ms,1,'increment')
        org=organization_metrics(sc,b,d,{'LCC':rs['lcc']})
        g=M.make_graph(z['types'],z['edges'],z['schema']);stat=M.stats(g,F,names,check);cmp=M.compare(g,stat,ref,rs,F,names)
        lc,mapping=M.lcc(g);ls=M.stats(lc,F,names,check);lcmp=M.compare(lc,ls,ref,rs,F,names)
        full=dict(identity=item,score=cost,type_score_parts=parts,organization=org,graph_sha256=inc.assets.sha(gp),
                  raw=dict(statistics=stat,comparison=cmp),LCC=dict(statistics=ls,comparison=lcmp,mapping=mapping))
        write(ev/f'{item["model"]}_{item["index"]:02d}.json',full)
        rr=[]
        for view,st,co in [('raw',stat,cmp),('LCC',ls,lcmp)]:
            rr.append(dict(**item,view=view,nodes=st['nodes'],edges=st['edges'],reference_edges=rs['edges'],edge_abs_error=abs(st['edges']-rs['edges']),
                degree_W1=co['main_incident_W1'],joint_TV=co['main_joint_degree_TV'],triangles=st['triangles'],reference_triangles=rs['triangles'],
                triangle_abs_error=abs(st['triangles']-rs['triangles']),triangle_composition_TV=co['triangle_composition_TV'],
                MP2=st['mp'][1]['total'],MP2_abs_error=abs(st['mp'][1]['total']-rs['mp'][1]['total']),MP2_TV=co['MP_composition_TV'][1],
                MP3=st['mp'][2]['total'],MP3_abs_error=abs(st['mp'][2]['total']-rs['mp'][2]['total']),MP3_TV=co['MP_composition_TV'][2],
                LCC=st['lcc'],components=st['components'],isolates=st['isolates'],maximum_degree=st['maximum_degree'],
                score=cost['score'] if view=='raw' else None,degree_cost=cost['degree'] if view=='raw' else None,heat_cost=cost['heat'] if view=='raw' else None,
                anonymous_joint_L1=org['joint_relation_degree_L1'] if view=='raw' else None,
                low_fraction_error=org['type_low_fraction_error'] if view=='raw' else None,hub_error=org['type_top10_degree_mean_error'] if view=='raw' else None))
        return rr
    while True:
        check()
        for item in inventory:
            key=(item['model'],item['index'])
            if key in done or key in failures or not (Path(item['path'])/'summary.json').exists():continue
            try:done[key]=one(item)
            except Exception:failures[key]=dict(**item,traceback=traceback.format_exc())
            table(out/'per_graph',[r for rr in done.values() for r in rr]);write(out/'evaluation_failures.json',list(failures.values()))
        write(out/'CPU_progress.json',dict(epoch=time.time(),evaluated_graphs=len(done),failures=len(failures)))
        if (out/'gpu_finished.json').exists():break
        time.sleep(5)
    # One last sweep catches the final completed graph just before GPU exit.
    for item in inventory:
        key=(item['model'],item['index'])
        if key not in done and key not in failures and (Path(item['path'])/'summary.json').exists():
            try:done[key]=one(item)
            except Exception:failures[key]=dict(**item,traceback=traceback.format_exc())
    rows=[r for rr in done.values() for r in rr];table(out/'per_graph',rows)
    summary=[];associations=[];reliability=[];pairrows=[];groups=[];influence=[];components=[];missing=[]
    for row in p['models']:
        check();mid=row['id'];raw=sorted([r for r in rows if r['model']==mid and r['view']=='raw'],key=lambda r:r['index'])
        for view in ('raw','LCC'):
            for source in ('all16','reused6','new10'):
                rr=[r for r in rows if r['model']==mid and r['view']==view and (source=='all16' or r['source']==source)]
                sm=dict(model=mid,dataset=row['dataset'],train_seed=row['train_seed'],view=view,source=source,samples=len(rr))
                for key in ('score','degree_cost','heat_cost','edges','edge_abs_error','degree_W1','joint_TV','triangles','triangle_abs_error','triangle_composition_TV','MP2_abs_error','MP2_TV','MP3_abs_error','MP3_TV','LCC','isolates','anonymous_joint_L1','low_fraction_error','hub_error'):
                    vals=[r[key] for r in rr if r[key] is not None]
                    sm[key+'_mean']=float(np.mean(vals)) if vals else None;sm[key+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
                summary.append(sm)
        if len(raw)!=16 or any(not (out/mid/f'vector_{i:02d}.npy').exists() for i in range(16)):
            missing.append(dict(model=mid,graphs=len(raw),gradient_vectors=sum((out/mid/f'vector_{i:02d}.npy').exists() for i in range(16))));continue
        costs=np.array([r['score'] for r in raw]);dc=np.array([r['degree_cost'] for r in raw]);hc=np.array([r['heat_cost'] for r in raw])
        assert np.allclose(costs,dc+hc,rtol=1e-12,atol=1e-12)
        vectors=[np.load(out/mid/f'vector_{i:02d}.npy',mmap_mode='r') for i in range(16)]
        layout=json.loads(next((out/mid).glob('layout_*.json')).read_text())
        G=np.zeros((16,16));by={g:np.zeros((16,16)) for g in ('CN','joint_head','degree','backbone')}
        raw_energy=0.;center_energy=0.
        for part in layout:
            check()
            for a in range(part['start'],part['end'],200000):
                b=min(a+200000,part['end']);v=np.array([x[a:b] for x in vectors],dtype=np.float64)
                raw_energy+=float(np.sum(v*v));v-=v.mean(0,keepdims=True);gg=v@v.T
                center_energy+=float(np.trace(gg));G+=gg;by[part['group']]+=gg
        assert np.allclose(G,sum(by.values()),rtol=1e-10,atol=1e-7)
        np.savez(out/mid/'centered_gram.npz',all_parameters=G,**by)
        full=coefficient(costs,list(range(16)));c4=[coefficient(costs,g) for g in row['four_groups']];c8=[coefficient(costs,g) for g in row['eight_groups']]
        for k,ixs,cs in [(4,row['four_groups'],c4),(8,row['eight_groups'],c8),(16,[list(range(16))],[full])]:
            for j,(ix,c) in enumerate(zip(ixs,cs)):
                groups.append(dict(model=mid,K=k,group=j,indices=json.dumps(ix),cost_mean=float(costs[ix].mean()),cost_sd=float(costs[ix].std(ddof=1)),gradient_norm=norm(c,G),cosine_to_full_shared_samples=cosine(c,full,G)))
            for a,b in itertools.combinations(range(len(cs)),2):
                for name,gram in [('all',G)]+list(by.items()):
                    pairrows.append(dict(model=mid,K=k,group_a=a,group_b=b,parameter_group=name,cosine=cosine(cs[a],cs[b],gram),norm_a=norm(cs[a],gram),norm_b=norm(cs[b],gram),groups_disjoint=True))
        for i in range(16):
            other=coefficient(costs,[j for j in range(16) if j!=i]);delta=full-other
            influence.append(dict(model=mid,index=i,sample_seed=raw[i]['sample_seed'],cost=float(costs[i]),leave_one_out_cosine=cosine(other,full,G),
                                  relative_direction_change=norm(delta,G)/norm(full,G) if norm(full,G)>1e-20 else None))
        cd=coefficient(dc,list(range(16)));ch=coefficient(hc,list(range(16)))
        assert np.allclose(full,cd+ch,rtol=1e-10,atol=1e-13)
        for name,gram in [('all',G)]+list(by.items()):
            components.append(dict(model=mid,parameter_group=name,full_norm=norm(full,gram),degree_norm=norm(cd,gram),heat_norm=norm(ch,gram),degree_heat_cosine=cosine(cd,ch,gram)))
        for metric in ('edge_abs_error','degree_W1','joint_TV','triangle_abs_error','MP2_abs_error','MP3_abs_error','anonymous_joint_L1','low_fraction_error','hub_error','isolates'):
            vals=np.array([r[metric] for r in raw],float)
            rho=float(spearmanr(costs,vals).statistic) if np.std(vals)>0 and np.std(costs)>0 else None
            associations.append(dict(model=mid,metric=metric,samples=16,cost_error_spearman=rho,
                                     meaning='negative means lower score associates with larger measured error; descriptive only'))
        cos4=[cosine(a,b,G) for a,b in itertools.combinations(c4,2)];cos4=[c for c in cos4 if c is not None]
        reliability.append(dict(model=mid,dataset=row['dataset'],train_seed=row['train_seed'],
            disjoint4_cosine_mean=float(np.mean(cos4)) if cos4 else None,disjoint4_cosine_min=min(cos4) if cos4 else None,
            disjoint4_cosine_max=max(cos4) if cos4 else None,disjoint4_negative_pairs=sum(c<0 for c in cos4),
            disjoint8_cosine=cosine(c8[0],c8[1],G),full16_gradient_norm=norm(full,G),
            old6_new10_cosine=cosine(coefficient(costs,list(range(6))),coefficient(costs,list(range(6,16))),G),
            centered_vector_energy_fraction=center_energy/raw_energy if raw_energy else None,
            leave_one_out_min_cosine=min(x['leave_one_out_cosine'] for x in influence if x['model']==mid and x['leave_one_out_cosine'] is not None),
            maximum_relative_leave_one_out_change=max(x['relative_direction_change'] for x in influence if x['model']==mid and x['relative_direction_change'] is not None)))
    for name,data in [('model_summary',summary),('gradient_reliability',reliability),('disjoint_group_cosines',pairrows),('gradient_groups',groups),('trajectory_influence',influence),('component_gradients',components),('score_metric_associations',associations)]:table(out/name,data)
    write(out/'analysis_missing.json',missing);write(out/'evaluation_failures.json',list(failures.values()))
    assert 'torch' not in sys.modules
    write(out/'analysis_receipt.json',dict(status='complete' if len(done)==64 and len(reliability)==4 and not failures else 'partial',
        graphs=len(done),models_with_reliability=len(reliability),seconds=time.time()-start,CPU_seconds=time.process_time(),
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,model_calls=0,GPU_hours=0,failures=list(failures.values()),missing=missing))
if __name__=='__main__':
    try:main()
    except BaseException:
        out=Path(sys.argv[1]);write(out/'analysis_failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()));raise
