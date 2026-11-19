"""Independent CPU verification and transparent summary of archived switches."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,time,hashlib,resource,itertools,sys,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/degree_preserving_switch_20261109'
START=time.monotonic()
spec=importlib.util.spec_from_file_location('sw',ROOT/'research_branches/degree_preserving_switch_20261109/run_cpu.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)

def load(name):return json.loads((OUT/(name+'.json')).read_text())
def save(name,obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(obj,list) and obj:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(obj[0]));w.writeheader();w.writerows(obj)

def independent_connectivity(q,bits,n):
    parent=list(range(n));size=[1]*n
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for u,v in q[:,np.flatnonzero(bits)].T:
        x,y=root(int(u)),root(int(v))
        if x!=y:parent[x]=y;size[y]+=size[x]
    sizes=[size[i] for i in range(n) if root(i)==i]
    return dict(LCC=max(sizes),components=len(sizes),isolates=sum(s==1 for s in sizes))

def main():
    binding=load('binding');assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in binding.items())
    rows=load('moves');plans=load('frozen_candidates');bases=load('bases');cosines=load('replicate_cosines')
    inp=next(x for t,seed,x in S.torch.load(S.H.OLD/'PubMed_fixed_inputs.pt',map_location='cpu',weights_only=False)['inputs'] if t==1000)
    q,f,y=map(np.asarray,(inp['query'],inp['f'],inp['labels']));S.FAMILIES=f
    with np.load(ROOT/'results/rooted_connection_20261109T015142/PubMed_cpu_cost_panel.npz') as z:types=z['types'].copy()
    core=S.H.RootedScore(types,q,f,y,10);sc=S.H.ExplicitScore(core)
    channels=np.full((4,4),-1,int)
    for k,(a,b) in enumerate(core.triangle_pairs):channels[a,b]=channels[b,a]=10+k
    aggreg=[];classrows=[];unique_checks=0;finite_checks=0;examples=0
    for plan,base in zip(plans,bases):
        tag=plan['state'];rs=[r for r in rows if r['state']==tag]
        with np.load(S.H.FULL/(tag+'.npz')) as z:p=z['probability'].copy();bits=z['bits'].copy()
        z=np.log(p)-np.log1p(-p);original_conn=independent_connectivity(q,bits,len(types));seen=set()
        for r in rs:
            move=tuple(r[k] for k in ['remove0','remove1','add0','add1']);e0,e1,a0,a1=move
            assert move in map(tuple,next(v['moves'] for v in plan['parts'] if v['replicate']==r['replicate'] and v['relation']==r['relation']))
            if move in seen:continue
            seen.add(move);unique_checks+=1
            assert bits[e0]==bits[e1]==1 and bits[a0]==bits[a1]==0
            assert len(set(q[:,[e0,e1]].ravel()))==4
            assert all(f[x]==r['relation'] for x in move)
            before=np.bincount(q[:,[e0,e1]].ravel(),minlength=len(types));after=np.bincount(q[:,[a0,a1]].ravel(),minlength=len(types))
            assert np.array_equal(before,after)
            b=bits.copy();b[[e0,e1]]=0;b[[a0,a1]]=1
            conn=independent_connectivity(q,b,len(types))
            assert conn['LCC']-original_conn['LCC']==r['LCC_change'] and conn['components']-original_conn['components']==r['components_change']
            assert conn['isolates']==original_conn['isolates']
            w0=p[e0]*p[e1]*(1-p[a0])*(1-p[a1]);w1=(1-p[e0])*(1-p[e1])*p[a0]*p[a1]
            assert abs(w1/(w0+w1)-r['q'])<1e-12
            s=z[a0]+z[a1]-z[e0]-z[e1];delta=r['score_change']
            fd=delta*(S.H.sigmoid(s+1e-5)-S.H.sigmoid(s-1e-5))/2e-5
            assert abs(fd-r['h'])<1e-10;finite_checks+=1
        good=[r for r in rs if r['score_change']<-1e-12];mass=sum(abs(r['h']) for r in good)
        costs=[r for r in good if r['triangle_composition_TV_change']>1e-12]
        aggreg.append(dict(state=tag,base_triangles=base['triangle_count'],reference_triangles=float(sc.reftri.sum()),
            **S.stats(rs),rewarded_composition_cost_fraction=len(costs)/len(good) if good else None,
            rewarded_composition_cost_feedback_fraction=sum(abs(r['h']) for r in costs)/mass if mass else None,
            mean_replica_cosine=float(np.mean([c['cosine'] for c in cosines if c['state']==tag])),
            any_connectivity_change=sum(r['LCC_change']!=0 or r['components_change']!=0 for r in rs)))
        # One deterministic first composition-cost example per background; not a new candidate selection.
        if costs:
            r=costs[0];b=bits.copy();b[[r['remove0'],r['remove1']]]=0;b[[r['add0'],r['add1']]]=1
            rb=S.independent_raw(q,bits,types,10,channels);ra=S.independent_raw(q,b,types,10,channels)
            tb=sc.triangle_counts(rb)[0];ta=sc.triangle_counts(ra)[0]
            assert abs((abs(ta-sc.reftri).sum()-abs(tb-sc.reftri).sum())/sc.triangle_scale-r['typed_triangle_L1_change'])<1e-12
            for j,cl in enumerate(sc.classes):
                classrows.append(dict(state=tag,replicate=r['replicate'],relation=r['relation'],position=r['position'],triangle_class=str(cl),
                    reference_count=float(sc.reftri[j]),before_count=float(tb[j]),after_count=float(ta[j]),
                    reference_fraction=float(sc.reftri[j]/sc.reftri.sum()),before_fraction=float(tb[j]/tb.sum()),after_fraction=float(ta[j]/ta.sum())))
            examples+=1
    assert len(rows)==load('receipt')['expected_occurrences']==1824
    assert unique_checks==load('receipt')['unique_scored_moves']
    good=[r for r in rows if r['score_change']<-1e-12]
    save('aggregate',aggreg);save('composition_cost_examples',classrows)
    save('overall',dict(**S.stats(rows),rewarded_composition_cost_fraction=sum(r['triangle_composition_TV_change']>1e-12 for r in good)/len(good),
        rewarded_composition_cost_feedback_fraction=sum(abs(r['h']) for r in good if r['triangle_composition_TV_change']>1e-12)/sum(abs(r['h']) for r in good),
        mean_replica_cosine=float(np.mean([r['cosine'] for r in cosines])),
        not_independent_observations='Repeated candidates and graphs shared. These counts and cosines are descriptive, not significance or efficacy.',
        same_degree_does_not_imply_same_expected_degree_after_update=True))
    save('independent_audit',dict(status='passed',bound_files=len(binding),unique_degree_and_unionfind_checks=unique_checks,
        conditional_probability_and_finite_difference_checks=finite_checks,composition_cost_examples=examples,
        seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),GPU_hours=0))
    print((OUT/'aggregate.json').read_text());print((OUT/'independent_audit.json').read_text())

if __name__=='__main__':main()
