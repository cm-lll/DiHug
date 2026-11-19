"""Bounded CPU test of frozen production objectives in saved output space.

No denoiser, learned-parameter update, graph sampling, or production modification.
The test deliberately precedes a repair decision: do not call a tradeoff a bug.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import sys,json,time,csv,resource,traceback,signal,importlib.util
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import closure_semantics_audit as audit
from objective import descriptors,distribution_errors
torch.set_num_threads(1)
ROOT=audit.ROOT
BALANCED='--balanced' in sys.argv
OUT=ROOT/('results/closure_output_scale_candidate_20261116' if BALANCED else 'results/closure_gradient_cpu_20261116')
spec=importlib.util.spec_from_file_location('saved_risk',ROOT/'research_branches/analytic_joint_risk_20261113/core.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

def json_scalar(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=json_scalar))
def table(name,rows):
    dump(name+'.json',rows)
    if rows:
        with (OUT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def cloud(x,y):
    c=((x[:,None]-y[None,:])**2).mean(2);i,j=linear_sum_assignment(c);return float(c[i,j].mean())

def main():
    start=time.time();OUT.mkdir(exist_ok=False);bindings={}
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('600 second CPU wall limit')));signal.alarm(600)
    def bind(p): bindings[str(p)]=audit.sha(p);return Path(p)
    def read(p):return json.loads(bind(p).read_text())
    proto=read(audit.SRC/'protocol.json');manifest=read(audit.PAIRED/'input_manifest.json')
    sigma=read(ROOT/'results/hetero_micro_macro_calibration_20261116/frozen_scales.json')['sigma_squared']
    coeff=read(audit.SRC/'gradient_coefficients.json')['coefficients'];contexts={}
    dump('protocol.json',dict(scope='all32 frozen free/normal inputs, four seeds, candidate128 saved p0',
        directions=['CE','closure','macro','combined'],step='one unconstrained diagnostic logit descent, max absolute logit displacement0.01; no search, no fixed edge counts',
        denominator='CE sum over active slots / full unordered slot count; fixed support contributes zero',
        semantics='same frozen targets/scales/weights as production; p recovered as float64 logit of saved float32 probability, not re-evaluated network',
        endpoints='original and four local output-only directions, not AdamW or model parameter changes',
        purpose='test signed typed closure feedback before choosing any production repair',CPU_seconds_max=600,threads=1,GPU_hours=0,
        scale_candidate=dict(enabled=BALANCED,rule='k=||g_CE||2/||g_macro||2; stop-gradient k; g_new=g_CE+k*g_macro; old relative macro block weights unchanged',
            no_parameter_search=True,not_fixed_global_objective=True,zero_gradient='stop, do not silently amplify zero',
            prior_panel_development=True,step_max_logit=.01,only_output_space=True),
        limitations='Existing development states; parent free states are off-policy for candidate; descriptors soft conditional expectations, nonlinear comparisons are not expected hard loss'))
    for ds in ['IMDB','PubMed']:
        par=next(r for r in proto['parents'] if r['dataset']==ds);z=np.load(bind(par['data']));ty=z['types'];schema=z['schema'];n=len(ty)
        plan=audit.obj.make_plan(schema,len(np.unique(ty)));e=z['edges'];e=e if e.shape[0]==2 else e.T
        lookup={tuple(r):i for i,r in enumerate(schema)};ref=np.zeros((len(schema),n,n))
        for i,j in e.T:ref[lookup[tuple(sorted((ty[i],ty[j])))],i,j]=ref[lookup[tuple(sorted((ty[i],ty[j])))],j,i]=1
        rd,rc,_=audit.counts(ref,ty,plan);frozen=read(ROOT/'results/hetero_micro_macro_cpu_20261116_v2'/(ds+'_frozen.json'))
        ff={int(t):{b:dict(scale=torch.tensor(v['scale'],dtype=torch.float64),omega=torch.tensor(v['omega'],dtype=torch.float64),target=torch.tensor(np.array(v['target_real'])+1j*np.array(v['target_imag']))) for b,v in bs.items()} for t,bs in frozen.items()}
        deg=read(ROOT/'results/analytic_joint_risk_20261113'/(ds+'_frozen_target.json'))
        contexts[ds]=(ty,schema,plan,ref,rd,rc,ff,deg)
    rows=[];channelrows=[];gradrows=[];checks=[]
    for ent in manifest:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>3500:raise MemoryError('3.5GiB observed RSS stop')
        ds=ent['dataset'];ty,schema,plan,ref,rd,rc,ff,deg=contexts[ds];z=np.load(bind(ent['input']))
        assert audit.sha(ent['input'])==ent['input_sha256']
        prob=np.load(bind(audit.PAIRED/(ent['key']+'_calibrated_micro_macro_prediction.npz')))['p'].astype(float)
        assert np.isfinite(prob).all() and (prob>0).all() and (prob<1).all(), 'do not clip saturated probabilities silently'
        q=z['query'];f=z['f'];fq=z['fullquery'];fam=z['fullfamily'];ids=z['ids'];prior=z['prior']
        basefull=prior[fam].copy();basefull[ids]=prob;other=np.ones(len(fam),bool);other[ids]=False
        assert np.all((basefull[other]==0)|(basefull[other]==1))
        np.testing.assert_array_equal(fq[:,ids],q);np.testing.assert_array_equal(fam[ids],f)
        lab=torch.tensor(ref[f,q[0],q[1]],dtype=torch.float64);l=torch.tensor(np.log(prob)-np.log1p(-prob),requires_grad=True)
        def calc(logit):
            pp=logit.sigmoid();tp=torch.tensor(basefull).scatter(0,torch.tensor(ids),pp)
            P=tp.new_zeros(ref.shape);P[fam,fq[0],fq[1]]=tp;P[fam,fq[1],fq[0]]=tp
            desc=descriptors(P,ty,plan,check=False)
            errs=[r for r in distribution_errors(desc,ty,ff) if r['block']!='degree']
            _,parts=core.risks(tp,ty,fq,fam,deg);errs += [dict(type=r['type'],block='degree',error=r['risk']) for r in parts]
            blocks={b:torch.stack([r['error']/(4*sigma[ds][str(r['type'])][b]) for r in errs if r['block']==b]).mean() for b in coeff}
            ce=torch.nn.functional.binary_cross_entropy_with_logits(logit,lab,reduction='sum')/len(fam)
            macro=sum(coeff[b]*blocks[b] for b in coeff)
            return dict(CE=ce,closure=blocks['closure'],macro=macro,combined=ce+macro),desc,tp
        vals,desc,tp=calc(l);keys=['CE','macro'] if BALANCED else ['CE','closure','macro','combined']
        grads={k:torch.autograd.grad(vals[k],l,retain_graph=i<len(keys)-1)[0].detach() for i,k in enumerate(keys)}
        if BALANCED:
            assert grads['CE'].norm()>0 and grads['macro'].norm()>0
            multiplier=float(grads['CE'].norm()/grads['macro'].norm())
            grads['combined']=grads['CE']+grads['macro']
            grads['balanced']=grads['CE']+multiplier*grads['macro']
        assert all(torch.isfinite(g).all() for g in grads.values())
        np.testing.assert_allclose(grads['combined'],grads['CE']+grads['macro'],rtol=1e-8,atol=1e-12)
        # Finite difference against the actual frozen scalar, not counts alone.
        target='balanced' if BALANCED else 'combined'
        unit=-grads[target]/grads[target].abs().max();h=1e-4
        with torch.no_grad():
            pv=calc(l+h*unit)[0];mv=calc(l-h*unit)[0]
            plus=pv['CE']+multiplier*pv['macro'] if BALANCED else pv['combined']
            minus=mv['CE']+multiplier*mv['macro'] if BALANCED else mv['combined']
        finite=float((plus-minus)/(2*h));analytic=float((grads[target]*unit).sum())
        rel=abs(finite-analytic)/max(1e-10,abs(analytic));assert rel<2e-4,(ent['key'],rel)
        checks.append(dict(key=ent['key'],finite_difference=finite,analytic=analytic,relative_error=rel))
        ident={k:ent[k] for k in ['key','dataset','train_seed','sample_seed','kind','t']}
        for k,g in grads.items():
            def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm())) if a.norm()*b.norm()>0 else None
            gradrows.append(dict(**ident,objective=k,norm=float(g.norm()),cosine_CE=cosine(g,grads['CE']),cosine_closure=cosine(g,grads['closure']) if 'closure' in grads else None,
                macro_multiplier=multiplier if BALANCED else 1.0))
        # Freeze current-Xt strata for all local directions, never reference IDs.
        X=np.zeros((len(ty),len(ty)));X[fq[0],fq[1]]=z['xt'];X+=X.T;xd=X.sum(1);hub=np.zeros(len(ty),bool)
        for typ in np.unique(ty):
            ix=np.flatnonzero(ty==typ);hub[ix[np.argsort(xd[ix],kind='stable')[-max(1,int(np.ceil(len(ix)*.1))):]]]=True
        low=(xd<=1)&~hub
        for arm in (['balanced'] if BALANCED else ['original','CE','closure','macro','combined']):
            ll=l.detach() if arm=='original' else l.detach()-.01*grads[arm]/grads[arm].abs().max()
            with torch.no_grad():v,d,p=calc(ll)
            dc=d['closure'].numpy();dd=d['degree'].numpy();x=np.concatenate([dd,dc],1);rx=np.concatenate([rd,rc],1)
            typed=0.;w2=0.;nr=0
            for typ in np.unique(ty):
                ix=ty==typ;scale=ff[int(typ)]['closure']['scale'].numpy();w2+=cloud(x[ix]/scale,rx[ix]/scale)/len(np.unique(ty))
                for col,rels in enumerate(plan['closures']):
                    # Keep only feasible channels; include reference-zero but possible closure channels.
                    r,s,u=rels
                    if not (np.any(ref[r]) and np.any(ref[s]) and np.any(ref[u])):continue
                    a=int(typ);er=schema[r];eu=schema[u]
                    if a not in er or a not in eu:continue
                    b=er[1] if er[0]==a else er[0];c=eu[1] if eu[0]==a else eu[0]
                    if sorted((b,c))!=list(schema[s]):continue
                    target=float(rc[ix,col].sum());pred=float(dc[ix,col].sum());sc=float(np.maximum(np.sqrt(np.mean(rc[ix,col]**2)),1))
                    typed+=abs(pred-target)/ix.sum()/sc;nr+=1
                    channelrows.append(dict(**ident,arm=arm,type=a,relations=str(rels),nodes=int(ix.sum()),reference=target,predicted=pred,signed_error=pred-target,absolute_error=abs(pred-target),normalized_error=abs(pred-target)/ix.sum()/sc))
            active=p.numpy()[ids];yy=lab.numpy().astype(bool)
            out=dict(**ident,arm=arm,**{k:float(value) for k,value in v.items()},triangles=float(dc.sum()/3),reference_triangles=float(rc.sum()/3),triangle_abs_error=abs(float((dc.sum()-rc.sum())/3)),
                typed_closure_normalized_MAE=typed/nr,joint_W2=w2,expected_edges=float(p.sum()),reference_edges=float(ref.sum()/2),
                positive_CE=float(torch.nn.functional.softplus(-ll)[lab.bool()].mean()),negative_CE=float(torch.nn.functional.softplus(ll)[~lab.bool()].mean()),
                missed_mass=float((1-active[yy]).sum()),false_mass=float(active[~yy].sum()),
                expected_relation_degree_sorted_MAE=float(np.mean([np.abs(np.sort(dd[ty==a],axis=0)-np.sort(rd[ty==a],axis=0)).mean() for a in np.unique(ty)])),
                fixed_current_hub_degree_MAE=float(np.abs(dd[hub].sum(1)-rd[hub].sum(1)).mean()) if hub.any() else None,
                fixed_current_low_degree_MAE=float(np.abs(dd[low].sum(1)-rd[low].sum(1)).mean()) if low.any() else None,
                group_note='hub/low node-index error only for local conditional prediction on known reference labels, not anonymous generation accuracy',
                macro_multiplier=multiplier if BALANCED else 1.0,
                frozen_local_candidate_loss=float(v['CE']+multiplier*v['macro']) if BALANCED else None)
            rows.append(out)
        table('rows',rows);dump('progress.json',dict(complete_inputs=len(checks),seconds=time.time()-start))
    table('channels',channelrows);table('gradients',gradrows);dump('checks.json',checks)
    if BALANCED:
        parent=ROOT/'results/closure_gradient_cpu_20261116'
        previous=read(parent/'rows.json');before=read(parent/'bindings.json')
        for p,hsh in before.items():
            if p!=str(Path(__file__).resolve()):assert audit.sha(p)==hsh,p
        summary=[];base={r['key']:r for r in previous if r['arm']=='original'};old={r['key']:r for r in previous if r['arm']=='combined'}
        for seed in sorted({r['train_seed'] for r in rows}):
            for kind in ['free','normal']:
                rr=[r for r in rows if r['train_seed']==seed and r['kind']==kind];o=dict(train_seed=seed,kind=kind,n=len(rr))
                for m in ['CE','closure','combined','triangle_abs_error','typed_closure_normalized_MAE','joint_W2','positive_CE','negative_CE','expected_relation_degree_sorted_MAE','fixed_current_hub_degree_MAE','fixed_current_low_degree_MAE']:
                    for label,baseline in [('start',base),('old_combined',old)]:
                        delta=[r[m]-baseline[r['key']][m] for r in rr if r[m] is not None and baseline[r['key']][m] is not None]
                        o[m+'_vs_'+label+'_improved']=int(sum(x<-1e-10 for x in delta));o[m+'_vs_'+label+'_mean_delta']=float(np.mean(delta)) if delta else None
                summary.append(o)
        table('summary',summary)
        bind(__file__);dump('bindings.json',bindings)
        assert all(audit.sha(p)==h for p,h in bindings.items())
        dump('receipt.json',dict(status='complete',seconds=time.time()-start,peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            inputs=32,output_directions=32,finite_difference_checks=32,network_calls=0,GPU_hours=0,model_updates=0,new_graphs=0,production_modified=False))
        signal.alarm(0);return
    summary=[]
    metrics=['CE','closure','combined','triangle_abs_error','typed_closure_normalized_MAE','joint_W2','positive_CE','negative_CE','expected_relation_degree_sorted_MAE']
    for seed in sorted({r['train_seed'] for r in rows}):
        for kind in ['free','normal']:
            for arm in ['CE','closure','macro','combined']:
                rr=[r for r in rows if r['train_seed']==seed and r['kind']==kind and r['arm']==arm]
                base={r['key']:r for r in rows if r['arm']=='original'}
                o=dict(train_seed=seed,kind=kind,arm=arm,n=len(rr))
                for m in metrics:
                    ds=[r[m]-base[r['key']][m] for r in rr];o[m+'_improved']=sum(x<-1e-10 for x in ds);o[m+'_mean_delta']=float(np.mean(ds))
                summary.append(o)
    table('summary',summary)
    for p in [__file__,audit.__file__,ROOT/'research_branches/hetero_micro_macro_20261116/objective.py',ROOT/'research_branches/analytic_joint_risk_20261113/core.py']:bind(p)
    dump('bindings.json',bindings)
    assert all(audit.sha(k)==v for k,v in bindings.items())
    dump('receipt.json',dict(status='complete',CPU_seconds=time.time()-start,peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,inputs=len(checks),output_directions=128,finite_difference_checks=len(checks),network_calls=0,GPU_hours=0,model_updates=0,new_graphs=0,production_modified=False))
    signal.alarm(0)

def recover_report():
    """Resume only JSON/CSV summarization after numpy-scalar serialization error."""
    rows=json.loads((OUT/'rows.json').read_text());grads=json.loads((OUT/'gradients.json').read_text());checks=json.loads((OUT/'checks.json').read_text())
    assert len(rows)==160 and len(grads)==128 and len(checks)==32
    base={r['key']:r for r in rows if r['arm']=='original'};summary=[]
    metrics=['CE','closure','combined','triangle_abs_error','typed_closure_normalized_MAE','joint_W2','positive_CE','negative_CE','expected_relation_degree_sorted_MAE']
    for seed in sorted({r['train_seed'] for r in rows}):
        for kind in ['free','normal']:
            for arm in ['CE','closure','macro','combined']:
                rr=[r for r in rows if r['train_seed']==seed and r['kind']==kind and r['arm']==arm]
                o=dict(train_seed=seed,kind=kind,arm=arm,n=len(rr))
                for m in metrics:
                    delta=[r[m]-base[r['key']][m] for r in rr];o[m+'_improved']=int(sum(x<-1e-10 for x in delta));o[m+'_mean_delta']=float(np.mean(delta))
                summary.append(o)
    table('summary',summary)
    proto=json.loads((audit.SRC/'protocol.json').read_text());manifest=json.loads((audit.PAIRED/'input_manifest.json').read_text());paths=[Path(__file__),Path(audit.__file__),ROOT/'research_branches/hetero_micro_macro_20261116/objective.py',ROOT/'research_branches/analytic_joint_risk_20261113/core.py',audit.SRC/'protocol.json',audit.SRC/'gradient_coefficients.json',audit.PAIRED/'input_manifest.json',ROOT/'results/hetero_micro_macro_calibration_20261116/frozen_scales.json']
    for ds in ['IMDB','PubMed']:
        paths += [Path(next(r for r in proto['parents'] if r['dataset']==ds)['data']), ROOT/'results/hetero_micro_macro_cpu_20261116_v2'/(ds+'_frozen.json'),ROOT/'results/analytic_joint_risk_20261113'/(ds+'_frozen_target.json')]
    for e in manifest:
        assert audit.sha(e['input'])==e['input_sha256']
        paths += [Path(e['input']),audit.PAIRED/(e['key']+'_calibrated_micro_macro_prediction.npz')]
    # Reuse the immediately prior audit's pre-existing hashes wherever available.
    old=json.loads((ROOT/'results/closure_score_attribution_20261116/sources.json').read_text())
    verified=[]
    for p in paths:
        if str(p) in old:
            assert audit.sha(p)==old[str(p)],str(p);verified.append(str(p))
    dump('bindings.json',{str(p):audit.sha(p) for p in paths})
    progress=json.loads((OUT/'progress.json').read_text())
    dump('receipt.json',dict(status='complete_after_report_recovery',CPU_seconds_through_last_input=progress['seconds'],peak_MiB=None,
        peak_memory_note='Main process exited at report serialization before writing final resource receipt; not reconstructed.',
        inputs=32,output_directions=128,finite_difference_checks=32,network_calls=0,GPU_hours=0,model_updates=0,new_graphs=0,production_modified=False,
        failure='numpy int64 summary count serialization; numerical computations complete and saved before error',recomputed_scientific_inputs=0,
        source_bindings='Postrun hashes, plus prior-audit shared-asset and all input-manifest hash checks; initial full binding manifest was not saved before report failure',prior_audit_hashes_rechecked=len(verified)))

if __name__=='__main__':
    try:
        if '--summarize' in sys.argv:recover_report()
        else:main()
    except BaseException:
        if OUT.exists():dump('failure.json',dict(traceback=traceback.format_exc()))
        raise
