"""Compare numpy audit terms with the actual prior score classes, CPU only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,hashlib,time,resource,signal
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/component_target_cpu_20261109/verified'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    tick=time.monotonic();signal.alarm(120);resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    src=['research_branches/component_target_cpu_20261109/run_cpu.py','research_branches/explicit_structure_cpu_20261109/explicit_score.py','research_branches/joint_degree_distance_20261109/run.py']
    a,e,j=[module(n,p) for n,p in zip(['audit','explicit','joint'],src)]
    # Minimal ndarray protocol adapter for the historical score's .numpy() calls;
    # no Torch tensors, model or alternative feature definition.
    class Arr(np.ndarray):
        def numpy(self):return np.asarray(self)
    def wrap(x):return np.asarray(x).view(Arr)
    rows=json.loads((OUT/'flips.json').read_text());bases=json.loads((OUT/'backgrounds.json').read_text());plan=json.loads((ROOT/'results/anonymous_teacher_protocol_20261109/next_cpu_proposal.json').read_text());errs=[]
    for ds in ['IMDB','PubMed']:
        p=ROOT/f'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1/package/assets/{ds}.npz';t,s,ref=a.load(p)
        q=np.array(np.triu_indices(len(t),1));rm={tuple(v):r for r,v in enumerate(s)};f=np.array([rm[tuple(sorted(t[[u,v]]))] for u,v in q.T]);bits=ref[q[0],q[1]].astype(float)
        rf=a.features(ref,t,s);core=SimpleNamespace(reference=bits,r=10,n=192,base_width=20,type_values=range(4),triangle_pairs=a.PAIRS,
            nodes={k:wrap(np.flatnonzero(t==k)) for k in range(4)},masks={k:wrap(np.r_[(s==k).any(1),np.ones(10,bool)]) for k in range(4)},
            features=lambda b:wrap(np.concatenate(rf[:2],axis=1)))
        score=e.ExplicitScore(core);joint=j.JointDegree(dict(types=t,query=q,families=f,reference=bits))
        for br in [b for b in bases if b['dataset']==ds]:
            aa=ref if br['background']=='reference' else a.load(next(z['file'] for z in plan['archived_graphs'] if z['dataset']==ds and z['train_seed']==br['train_seed']))[2]
            ff=a.features(aa,t,s)
            def old_terms(feat):
                terms=score.terms(np.concatenate(feat[:2],axis=1));jt=joint.terms(feat[0])[0]['joint']
                return {**{k:float(terms[k][0]) for k in ['relation_W1','low','hub','node_triangle_W1','typed_triangle_L1']},'joint':jt,'old_F':float(terms['total'][0])+(jt-float(terms['joint_TV'][0]))/8}
            base=old_terms(ff);errs.extend(abs(base[k]-br[k]) for k in base)
            for r in [r for r in rows if r['dataset']==ds and r['background']==br['background']]:
                u,v,fam,pr=[r[k] for k in ['u','v','relation','present']];sign=1-2*pr;df=[x.copy() for x in ff];df[0][[u,v],fam]+=sign
                for w in np.flatnonzero(aa[u]&aa[v]):
                    for n,i,z in [(u,v,w),(v,u,w),(w,u,v)]:df[1][n,a.PI[tuple(sorted(t[[i,z]]))]]+=sign
                aft=old_terms(df)
                errs.extend(abs(aft[k]-base[k]-r[k+'_delta']) for k in base)
    assert max(errs)<1e-10,max(errs)
    result=dict(status='passed',comparisons=len(errs),max_error=max(errs),seconds=time.monotonic()-tick,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in src},scope='Actual historical score classes; pure numpy protocol adapter, no model, same-assistant independent implementation check')
    (OUT/'historical_term_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
