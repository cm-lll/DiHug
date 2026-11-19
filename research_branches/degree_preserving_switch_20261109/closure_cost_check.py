"""Exact cancellation of degree terms; audit-only optimization, no new objective."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time,resource,importlib.util,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/degree_preserving_switch_20261109'
START=time.monotonic()
spec=importlib.util.spec_from_file_location('sw',ROOT/'research_branches/degree_preserving_switch_20261109/run_cpu.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)

def closure_only(sc,raw):
    value=0.;nt=len(sc.cache)
    for c in sc.cache.values():
        tr=raw[c['ids'],sc.core.r:sc.core.base_width][:,c['tm']]
        if tr.shape[-1]:value+=float((abs(np.sort(tr,axis=0)-c['rt'])/c['tri_scale']).mean())/nt
    typed=float(abs(sc.triangle_counts(raw)[0]-sc.reftri).sum())/sc.triangle_scale
    return (value+typed)/8

def main():
    inp=next(x for t,seed,x in S.torch.load(S.H.OLD/'PubMed_fixed_inputs.pt',map_location='cpu',weights_only=False)['inputs'] if t==1000)
    q,f,y=map(np.asarray,(inp['query'],inp['f'],inp['labels']))
    with np.load(ROOT/'results/rooted_connection_20261109T015142/PubMed_cpu_cost_panel.npz') as z:types=z['types'].copy()
    core=S.H.RootedScore(types,q,f,y,10);sc=S.H.ExplicitScore(core)
    jd=S.H.jdmod.JointDegree(dict(types=types,query=q,families=f,reference=y))
    channels=np.full((4,4),-1,int)
    for k,(a,b) in enumerate(core.triangle_pairs):channels[a,b]=channels[b,a]=10+k
    rows=json.loads((OUT/'moves.json').read_text());bench=[];errors=[];cache_states=[];matched=0
    for tag in dict.fromkeys(r['state'] for r in rows):
        with np.load(S.H.FULL/(tag+'.npz')) as z:bits=z['bits'].copy()
        with S.torch.no_grad():raw=core.features(S.torch.from_numpy(bits)).numpy()[:,:core.base_width]
        adj=S.H.adjacency(q,bits,len(types));base=closure_only(sc,raw);seen=set();selected=[]
        for row in [r for r in rows if r['state']==tag]:
            key=tuple(row[k] for k in ['remove0','remove1','add0','add1'])
            if key in seen:continue
            seen.add(key);ar=raw.copy();aa=adj.copy()
            for slot in key:S.H.flip(ar,aa,q,types,channels,slot,row['relation'])
            err=abs(closure_only(sc,ar)-base-row['score_change']);assert err<1e-12;errors.append(err);matched+=1
            if len(selected)<32:selected.append((key,row['relation']))
        # Fixed first 32 distinct proposals, no effect selection; include incremental features and connectivity in both timings.
        original=S.H.describe(sc,jd,raw,adj)['score'];vals={};times={}
        for method in ['full','closure_only']:
            tick=time.perf_counter();out=[]
            for key,r in selected:
                ar=raw.copy();aa=adj.copy()
                for slot in key:S.H.flip(ar,aa,q,types,channels,slot,r)
                if method=='full':res=S.H.describe(sc,jd,ar,aa);d=res['score']-original;conn={k:res[k] for k in ['LCC','components','isolates']}
                else:d=closure_only(sc,ar)-base;conn=S.H.connectivity(aa)
                out.append((d,conn))
            times[method]=time.perf_counter()-tick;vals[method]=out
        assert all(abs(a[0]-b[0])<1e-12 and a[1]==b[1] for a,b in zip(vals['full'],vals['closure_only']))
        bench.append(dict(state=tag,moves=len(selected),full_seconds=times['full'],closure_only_seconds=times['closure_only'],
            speedup=times['full']/times['closure_only'],fast_milliseconds_per_move=1000*times['closure_only']/len(selected)))
    fast=sum(b['closure_only_seconds'] for b in bench)/sum(b['moves'] for b in bench)
    result=dict(status='passed',unique_score_equivalence_checks=matched,max_score_difference=max(errors),benchmarks=bench,
        overall_full_to_fast_ratio=sum(b['full_seconds'] for b in bench)/sum(b['closure_only_seconds'] for b in bench),
        hypothetical_160_moves_per_update_seconds=fast*160,hypothetical_3000_updates_CPU_minutes=fast*160*3000/60,
        limits='One timing pass, full then fast; potential cache/order bias. Includes raw feature delta, closure scoring, connectivity. Excludes model, hard-background draws, proposal enumeration, parameter VJP, GPU transfer/synchronization. Not end-to-end training throughput.',
        seconds=time.monotonic()-START,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),GPU_hours=0)
    (OUT/'closure_cost_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
