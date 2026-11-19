"""Saved-state algebra, exact count-risk analysis; NO neural/optimizer calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv,time,signal,resource,itertools,gc,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/hard_clean_zero_safe_20261110'
OUT=ROOT/'results/count_adam_audit_cpu_20261110'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def table(p,rows):
    if not rows:return
    with Path(p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def norm(x):return float(np.linalg.norm(x))
def cosine(x,y):
    den=norm(x)*norm(y);return float(np.dot(x,y)/den) if den>0 else None
def loo(p,b,c):
    centered=c-c[0];K=len(b)
    return ((K*centered-centered.sum(0))/(K-1)).T@(b-p)/K
def exact(p,f,targets):
    R=len(targets);grad=np.zeros_like(p);risks=[]
    for r,m in enumerate(targets):
        ix=f==r;pp=p[ix];mu=pp.sum();var=(pp*(1-pp)).sum();scale=R*max(1,m)**2
        risks.append(((mu-m)**2+var)/scale)
        grad[ix]=pp*(1-pp)*(2*(mu-m)+1-2*pp)/scale
    return np.array(risks),grad
def tests():
    p=np.array([.13,.27,.45,.61,.78,.91]);f=np.array([0,0,0,1,1,2]);target=np.array([1,1,0])
    b=np.array(list(itertools.product([0,1],repeat=6)),float)
    c=np.column_stack([((b[:,f==r].sum(1)-m)/max(1,m))**2/len(target) for r,m in enumerate(target)])
    mass=np.prod(np.where(b,p,1-p),1);risk,g=exact(p,f,target)
    exg=(mass*c.sum(1))@(b-p)
    np.testing.assert_allclose(mass@c,risk,atol=1e-14,rtol=1e-13)
    np.testing.assert_allclose(exg,g,atol=1e-14,rtol=1e-13)
    return dict(passed=True,enumerated_graphs=64,max_gradient_error=float(abs(exg-g).max()),
        reference_zero_and_multiple_relations_included=True)

def main():
    start=time.time();OUT.mkdir(exist_ok=False)
    def stop(*_):raise RuntimeError('CPU 180s / 2GiB supervision stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(180)
    receipt=dict(start_epoch=start,GPU_hours=0,model_calls=0,backwards=0,optimizer_steps=0,new_graph_draws=0)
    summaries=[];relation_rows=[];parameter_rows=[]
    try:
        protocol=json.loads((SOURCE/'protocol.json').read_text());bindings={str(Path(__file__)):sha(__file__)}
        for task in protocol['tasks']:
            for p in [Path(task['stop_file']),Path(task['evaluation_file']),Path(task['data']),
                      SOURCE/task['task']/'temporary_complete.pt',SOURCE/task['task']/'result.json',SOURCE/task['task']/'failed_batch_replay.npz',
                      ROOT/'results/hard_clean_pilot_v2_20261110'/task['task']/'score_definition.json']:
                bindings[str(p)]=sha(p)
        for p in [SOURCE/'protocol.json',ROOT/'research_branches/hard_clean_zero_safe_20261110/check_real.py',ROOT/'research_branches/hard_clean_pilot_v2_20261110/objective.py']:
            bindings[str(p)]=sha(p)
        dump(OUT/'protocol.json',dict(bindings=bindings,CPU_threads=1,memory_GiB=2,wall_seconds=180,GPU=0,
            count_risk='mean_r ((N_r-m_r)/max(1,m_r))^2; exact independent Bernoulli expectation, no objective change',
            exact_margin_derivative='p_e(1-p_e) [2(mu_r-m_r)+1-2p_e] / (R max(1,m_r)^2)',
            Adam_analysis='algebra of saved pre/post moments and parameters, never optimizer.step or neural calls',
            inferred_gradient='(m_new-beta1*m_old)/(1-beta1), approximate due saved FP32 rounding',
            decomposition='history/current numerator contributions share ACTUAL post-update denominator; not independent causal effects',
            denominator_comparison='hold actual m_new fixed, substitute bias-corrected previous v_old; arithmetic sensitivity only, not a training alternative',
            no_training_change=True,no_new_sampling=True))
        dump(OUT/'tests.json',tests())
        import torch
        torch.set_num_threads(1)
        for task in protocol['tasks']:
            folder=SOURCE/task['task'];info=json.loads((folder/'result.json').read_text())
            saved=np.load(folder/'failed_batch_replay.npz');p=saved['probability'];b=saved['bits'].astype(float)
            inp=torch.load(task['evaluation_file'],map_location='cpu',weights_only=False)['inputs'][0]
            order=np.argsort(inp['ids']);f=inp['f'][order];np.testing.assert_array_equal(inp['ids'][order],saved['ids'])
            definition=json.loads((ROOT/'results/hard_clean_pilot_v2_20261110'/task['task']/'score_definition.json').read_text())
            targets=np.array(definition['reference']['counts']);R=len(targets)
            counts=np.column_stack([b[:,f==r].sum(1) for r in range(R)])
            rc=((counts-targets)/np.maximum(1,targets))**2/R
            np.testing.assert_allclose(rc.sum(1),saved['costs'][:,0],atol=1e-12,rtol=1e-13)
            er,eg=exact(p,f,targets);lg=loo(p,b,rc);total=lg.sum(0);on=np.zeros_like(p)
            for r,m in enumerate(targets):
                ix=f==r;on[ix]=lg[r,ix];pp=p[ix];own=lg[r].copy();own[~ix]=0
                relation_rows.append(dict(task=task['task'],dataset=task['dataset'],seed=task['seed'],relation=r,
                    target=int(m),slots=int(ix.sum()),expected_count=float(pp.sum()),count_variance=float((pp*(1-pp)).sum()),
                    sampled_count_mean=float(counts[:,r].mean()),sampled_risk=float(rc[:,r].mean()),exact_risk=float(er[r]),
                    normalization=1/(R*max(1,m)**2),exact_margin_norm=norm(eg[ix]),
                    LOO_margin_norm=norm(lg[r]),LOO_own_relation_norm=norm(lg[r,ix]),LOO_other_relation_norm=norm(lg[r,~ix]),
                    total_LOO_vs_exact_cosine=cosine(total[ix],eg[ix]),
                    exact_positive_remove_slots=int((eg[ix]>0).sum()),exact_negative_add_slots=int((eg[ix]<0).sum())))
            H=np.array(info['Gram']);coeff=np.array(info['coefficients']);combined2=float(coeff@H@coeff)
            count_share=float(coeff[0]*(H@coeff)[0]/combined2)
            record=dict(task=task['task'],dataset=task['dataset'],seed=task['seed'],update=info['update'],t=info['t'],
                count_exact_risk=float(er.sum()),count_sampled_risk=float(rc.sum(1).mean()),
                count_exact_margin_norm=norm(eg),count_LOO_margin_norm=norm(total),count_LOO_error_norm=norm(total-eg),
                count_LOO_cosine=cosine(total,eg),count_cross_relation_noise_norm=norm(total-on),
                count_cross_relation_noise_ratio=norm(total-on)/max(norm(total),1e-300),
                count_parameter_gradient_norm=info['individual_norms'][0],combined_gradient_norm=info['new_gradient_norm'],
                count_projection_share_of_combined=count_share,
                count_combined_parameter_cosine=float((H@coeff)[0]/np.sqrt(H[0,0]*combined2)))
            np.savez_compressed(OUT/f'{task["task"]}_count_gradients.npz',exact_margin_gradient=eg,
                LOO_margin_gradient=total,LOO_each_relation=lg,LOO_own_relation_only=on,ids=saved['ids'],exact_risks=er,sample_risks=rc)
            before=torch.load(task['stop_file'],map_location='cpu',weights_only=False)['state']
            after_wrap=torch.load(folder/'temporary_complete.pt',map_location='cpu',weights_only=False)
            assert after_wrap['parent_stop_sha256']==sha(task['stop_file']);after=after_wrap['state']
            assert after['updates']==before['updates']+1==info['update']
            o0,o1=before['optimizer'],after['optimizer'];assert o0['param_groups']==o1['param_groups']
            assert len(o0['param_groups'])==1
            opt=o1['param_groups'][0];assert not opt['amsgrad'] and not opt['maximize']
            names=list(before['ema']['shadow']);assert names==list(after['ema']['shadow'])
            assert names==[k for k in before['model'] if k!='prior']
            assert len(names)==len(opt['params']);np.testing.assert_array_equal(before['model']['prior'],after['model']['prior'])
            lr=opt['lr'];beta1,beta2=opt['betas'];eps=opt['eps'];wd=opt['weight_decay']
            sums={k:0. for k in ('g2','actual2','pred2','error2','historical2','current2','decay2','prevdenom2','history_current_dot','actual_current_dot','actual_history_dot','g_actual_dot','v_error2','v_new2')}
            total_n=0;dominant_v_n=0;maxerr=0.;state_count=0;unused=0;unchanged_steps=0
            recovered_flat=[];innovation_shares=[];steps=[]
            for name,pid in zip(names,opt['params']):
                theta0=before['model'][name].numpy().astype(float).ravel();theta1=after['model'][name].numpy().astype(float).ravel();delta=theta1-theta0
                if pid not in o0['state']:
                    assert pid not in o1['state'] and np.array_equal(theta0,theta1);unused+=1;continue
                a,z=o0['state'][pid],o1['state'][pid]
                assert tuple(a['exp_avg'].shape)==tuple(before['model'][name].shape)
                s0=int(a['step'].item());s1=int(z['step'].item());steps.append(s1)
                if s1==s0:
                    assert np.array_equal(theta0,theta1)
                    for field in ('exp_avg','exp_avg_sq'):assert torch.equal(a[field],z[field])
                    unchanged_steps+=1;continue
                assert s1==s0+1
                m0=a['exp_avg'].numpy().astype(float).ravel();m1=z['exp_avg'].numpy().astype(float).ravel()
                v0=a['exp_avg_sq'].numpy().astype(float).ravel();v1=z['exp_avg_sq'].numpy().astype(float).ravel()
                g=(m1-beta1*m0)/(1-beta1);v_formula=beta2*v0+(1-beta2)*g*g
                den=np.sqrt(v1/(1-beta2**s1))+eps;bc=1-beta1**s1
                hist=-lr*beta1*m0/bc/den;cur=-lr*(m1-beta1*m0)/bc/den;decay=-lr*wd*theta0
                predicted=hist+cur+decay;err=predicted-delta
                prevden=np.sqrt(v0/(1-beta2**s0))+eps
                prior_den_displacement=-lr*m1/bc/prevden+decay
                for key,value in [('g2',g),('actual2',delta),('pred2',predicted),('error2',err),('historical2',hist),('current2',cur),('decay2',decay),('prevdenom2',prior_den_displacement),('v_error2',v_formula-v1),('v_new2',v1)]:sums[key]+=float(value@value)
                for key,x,y in [('history_current_dot',hist,cur),('actual_current_dot',delta,cur),('actual_history_dot',delta,hist),('g_actual_dot',g,delta)]:sums[key]+=float(x@y)
                fraction=np.divide((1-beta2)*g*g,v1,out=np.zeros_like(v1),where=v1>0)
                total_n+=len(g);dominant_v_n+=int((fraction>.5).sum());maxerr=max(maxerr,float(abs(err).max()));state_count+=1
                parameter_rows.append(dict(task=task['task'],parameter=name,group=name.split('.')[0],elements=len(g),step_before=s0,step_after=s1,
                    inferred_gradient_norm=norm(g),actual_delta_norm=norm(delta),history_contribution_norm=norm(hist),current_contribution_norm=norm(cur),
                    actual_formula_error_norm=norm(err),actual_formula_max_error=float(abs(err).max()),new_second_moment_majority_fraction=float(np.mean(fraction>.5)),
                    old_denom_sensitivity_norm=norm(prior_den_displacement)))
            for k in ('g2','actual2','pred2','error2','historical2','current2','decay2','prevdenom2'):
                record[k[:-1]+'_norm']=sums[k]**.5
            record.update(optimizer= {k:opt[k] for k in ('lr','betas','eps','weight_decay','foreach','fused')},
                inferred_gradient_relative_norm_error=abs(sums['g2']**.5-info['new_gradient_norm'])/info['new_gradient_norm'],
                parameter_delta_relative_norm_error=abs(sums['actual2']**.5-info['actual_AdamW_norm'])/info['actual_AdamW_norm'],
                formula_relative_delta_error=(sums['error2']/sums['actual2'])**.5,formula_max_absolute_error=maxerr,
                second_moment_relative_formula_error=(sums['v_error2']/sums['v_new2'])**.5,
                current_projection_share=sums['actual_current_dot']/sums['actual2'],history_projection_share=sums['actual_history_dot']/sums['actual2'],
                history_current_cosine=sums['history_current_dot']/max((sums['historical2']*sums['current2'])**.5,1e-300),
                gradient_actual_descent_cosine=-sums['g_actual_dot']/max((sums['g2']*sums['actual2'])**.5,1e-300),
                coordinates_new_innovation_dominates_fraction=dominant_v_n/total_n,
                used_parameter_tensors=state_count,unused_parameter_tensors=unused,unchanged_optimizer_steps=unchanged_steps,
                coordinate_count=total_n,optimizer_step_range=[min(steps),max(steps)],
                moment_recovery_limitation='FP32 saved first-moment subtraction, not original gradient bitwise recovery',
                no_causal_counterfactual_claim=True)
            assert record['parameter_delta_relative_norm_error']<1e-10
            assert record['inferred_gradient_relative_norm_error']<1e-4
            assert record['formula_relative_delta_error']<1e-3
            assert record['second_moment_relative_formula_error']<1e-5
            summaries.append(record);dump(OUT/'summary.json',dict(tasks=summaries));table(OUT/'relations.csv',relation_rows);table(OUT/'parameters.csv',parameter_rows)
            print(json.dumps(dict(task=task['task'],seconds=time.time()-start,exact_count_norm=norm(eg),LOO_count_norm=norm(total),count_cosine=cosine(total,eg),formula_error=record['formula_relative_delta_error'])),flush=True)
            del before,after,after_wrap,o0,o1;gc.collect()
        for p,h in bindings.items():assert sha(p)==h
        receipt.update(status='complete',unchanged_bindings=len(bindings))
    except BaseException as exc:
        import traceback
        receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        table(OUT/'relations.csv',relation_rows);table(OUT/'parameters.csv',parameter_rows)
        receipt.update(seconds=time.time()-start,end_epoch=time.time(),models_completed=len(summaries),peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        dump(OUT/'receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
