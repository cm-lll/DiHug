"""Read saved finite trial only: independent kernel/cost audit and descriptive plots."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv, itertools, json, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/one_step_to_terminal_cpu_20261110'
START=time.monotonic()
a=np.load(OUT/'states_and_distributions.npz')
index=json.loads((OUT/'distribution_index.json').read_text())
cfg=json.loads((ROOT/'research_branches/terminal_structure_audit_20261109/protocol.json').read_text())
types=np.array(cfg['graph']['node_types']);edges=np.array(cfg['graph']['unordered_slots'])
bits=a['bits'];pi=np.array(cfg['graph']['family_priors']);alpha=np.array(cfg['noise']['cumulative_retention'])
families=np.array([0 if types[i]==types[j]==0 else 2 if types[i]==types[j]==1 else 1 for i,j in edges])
adj=np.zeros((len(bits),4,4));degree=np.zeros((len(bits),4,3));rel=np.zeros((len(bits),3))
for e,(i,j) in enumerate(edges):
    adj[:,i,j]=adj[:,j,i]=bits[:,e]
    degree[:,i,families[e]]+=bits[:,e];degree[:,j,families[e]]+=bits[:,e]
    rel[:,families[e]]+=bits[:,e]
tri=np.zeros((len(bits),2));comps=[]
for s,A in enumerate(adj):
    for nodes in itertools.combinations(range(4),3):
        if all(A[i,j] for i,j in itertools.combinations(nodes,2)):
            tri[s,types[list(nodes)].sum()-1]+=1
    reach=np.eye(4,dtype=bool)|(A>0)
    for k in range(4):reach|=reach[:,k,None]&reach[None,k,:]
    comps.append(len(set(tuple(r) for r in reach)))
comps=np.array(comps);ref=int(a['reference_id'])
joint=[]
for d in degree:
    groups=[]
    for typ in [0,1]:
        ix=np.flatnonzero(types==typ)
        groups.append(min(np.mean(abs(d[ix]-degree[ref,list(perm)]).sum(-1))/3
                          for perm in itertools.permutations(ix)))
    joint.append(np.mean(groups))
blocks=np.column_stack([(((rel-rel[ref])/np.maximum(1,rel[ref]))**2).mean(1),joint,
                        (((tri-tri[ref])/np.maximum(1,tri[ref]))**2).mean(1),((comps-comps[ref])/3)**2])
assert np.max(abs(blocks-a['blocks']))<1e-14
assert np.array_equal(comps,a['components']) and np.array_equal(tri.sum(1),a['triangles'])
checks=[];rows=[]
for n,item in enumerate(index):
    theta=a['parameters'][n];trans=[]
    for t in [1,2,3]:
        F=[];total=adj.sum(-1);cn=adj@adj
        for e,(i,j) in enumerate(edges):
            F.append(np.column_stack([np.tile(np.eye(3)[families[e]],(64,1)),bits[:,e],
                (total[:,i]+total[:,j])/6,((total[:,i]-total[:,j])/3)**2,
                cn[:,i,j]/2,np.full(64,t/3),bits.mean(1)]))
        p=1/(1+np.exp(-np.stack(F,axis=1)@theta))
        retention=alpha[t]/alpha[t-1]
        likelihood1=np.where(bits>0,retention+(1-retention)*pi[families],(1-retention)*(1-pi[families]))
        likelihood0=np.where(bits>0,(1-retention)*pi[families],retention+(1-retention)*(1-pi[families]))
        K=[]
        for b in [0,1]:
            v=alpha[t-1]*b+(1-alpha[t-1])*pi[families]
            K.append(v*likelihood1/(v*likelihood1+(1-v)*likelihood0))
        q=(1-p)*K[0]+p*K[1]
        P=np.prod(np.where(bits[None,:,:]>0,q[:,None,:],1-q[:,None,:]),axis=-1)
        err=float(np.max(abs(P-a['transitions'][n,t-1])))
        assert err<1e-12
        trans.append(P);checks.append(dict(**item,t=t,transition_error=err))
    prob=a['prior'].copy()
    for P in trans[::-1]:prob=prob@P
    assert np.max(abs(prob-a['distributions'][n]))<1e-12 and abs(prob.sum()-1)<1e-12
    for k in range(2):
        mean=float(prob@tri[:,k]);var=float(prob@(tri[:,k]-mean)**2)
        risk=float(prob@(tri[:,k]-tri[ref,k])**2)
        assert abs(risk-(mean-tri[ref,k])**2-var)<1e-12
        rows.append(dict(**item,triangle_type='001' if k==0 else '011',reference_count=float(tri[ref,k]),
                         mean=mean,variance=var,squared_mean_bias=(mean-tri[ref,k])**2,squared_error_expectation=risk))
with (OUT/'triangle_decomposition.csv').open('w') as f:
    w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'independent_audit.json').write_text(json.dumps(dict(checks=checks,passed=True,
    max_transition_error=max(c['transition_error'] for c in checks),cost_error=float(np.max(abs(blocks-a['blocks']))),
    audit_seconds=time.monotonic()-START,new_optimization_updates=0,GPU_hours=0),indent=2))
# Descriptive plot of previously computed records, no new optimization or model calls.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
curves=json.loads((OUT/'curves.json').read_text())
fig,axes=plt.subplots(2,3,figsize=(13,7))
keys=['forward_clean_risk','terminal_cost','typed_triangles','anonymous_joint_degree','connected_probability','isomorph_entropy']
titles=['Clean hard risk on forward inputs','Terminal hard risk','Typed triangle squared error','Joint relation-degree error','Connected probability','Type-preserving isomorph entropy']
for ax,key,title in zip(axes.ravel(),keys,titles):
    for si,seed in enumerate([2026110901,2026110902,2026110903]):
        for arm,style in [('reference_edge_CE','--'),('exact_forward_clean_structure','-')]:
            r=[r for r in curves if r['seed']==seed and r['arm']==arm]
            ax.plot([x['step'] for x in r],[x[key] for x in r],style,color=f'C{si}',
                    label=f'{str(seed)[-3:]} '+('CE' if style=='--' else 'Structure'))
    ax.set_title(title);ax.set_xlabel('Toy updates');ax.grid(alpha=.25)
axes[0,0].legend(fontsize=8,ncol=2)
fig.suptitle('Exact 64-state / 3-step toy diffusion; not real O/A or T1000',fontsize=12)
fig.tight_layout();fig.savefig(OUT/'curves.png',dpi=160);plt.close(fig)
print(json.dumps(dict(transition_checks=len(checks),max_error=max(c['transition_error'] for c in checks),seconds=time.monotonic()-START)))
