"""Independent pure-array verification of partitions and LP certificates."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,csv,itertools,time,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/toy_representability_20261110';start=time.time()
prot=json.loads((OUT/'protocol.json').read_text());parent=json.loads((ROOT/'results/mean_anchored_toy_20261110_attempt2/protocol.json').read_text());cfg=parent['graph'];ty=np.array(cfg['node_types']);edges=np.array(cfg['unordered_slots']);B=np.array(list(itertools.product([0.,1.],repeat=6)));A=np.zeros((64,4,4))
for e,(i,j) in enumerate(edges):A[:,i,j]=B[:,e];A[:,j,i]=B[:,e]
family=np.array([0 if ty[i]==ty[j]==0 else 2 if ty[i]==ty[j]==1 else 1 for i,j in edges]);cross=np.flatnonzero(family==1)
features=[]
for adj,bits in zip(A,B):
    deg=adj.sum(1);rows=[]
    for e,(i,j) in enumerate(edges):
        cn=sum(adj[i,k]*adj[j,k] for k in range(4))
        rows.append([*(np.arange(3)==family[e]).astype(float),bits[e],(deg[i]+deg[j])/6,((deg[i]-deg[j])/3)**2,cn/2,.01,bits.mean()])
    features.append(np.array(rows))
detail=json.loads((OUT/'state_details.json').read_text());lp=json.loads((OUT/'linear_separation.json').read_text());good=np.array(prot['good_state_ids']);counts={'witnesses':0,'identical_features_opposite_targets':0}
for x,row in enumerate(detail):
    F=features[x];groups=[np.array(g) for g in row['feature_edges']];auto=[set(g) for g in row['automorphism_edges']]
    for g in groups:assert np.array_equal(F[g],np.repeat(F[g[:1]],len(g),axis=0))
    for g,h in itertools.combinations(groups,2):assert not np.array_equal(F[g[0]],F[h[0]])
    if row['extra_feature_loss']:assert any(len({j for j,o in enumerate(auto) if set(g)&o})>1 for g in groups)
    p=np.array(row['attaining_probabilities']);v=sum(np.prod(np.where(B[y]>0,p,1-p)) for y in good);assert abs(v-row['feature_good_probability_cap'])<1e-14
    for result in [r for r in lp if r['state']==x]:
        y=2*B[result['target']]-1
        if result['feasible']:
            assert np.min(y*(F@np.array(result['theta'])))>=1-1e-7;counts['witnesses']+=1
        else:
            assert any(np.array_equal(F[i],F[j]) and y[i]!=y[j] for i,j in itertools.combinations(cross,2));counts['identical_features_opposite_targets']+=1
# Concrete extra-compression example: state2 has only edge (1,3).
x=2;i,j=2,3;assert np.array_equal(features[x][i],features[x][j]);assert family[i]==family[j]==1
example={'state':x,'existing_edges':edges[B[x]>0].tolist(),'indistinguishable_queries':edges[[i,j]].tolist(),'typed_endpoint_degrees':[[float(A[x].sum(1)[u]),float(A[x].sum(1)[v])] for u,v in edges[[i,j]]],'feature_vectors':features[x][[i,j]].tolist(),'target_labels':{str(y):B[y,[i,j]].astype(int).tolist() for y in good},'note':'These endpoints have different immutable types. Sum/squared-difference encoding erases which type has the larger degree.'}
(OUT/'compression_example.json').write_text(json.dumps(example,indent=2))
rr=list(csv.DictReader((OUT/'readout_bounds.csv').open()));aggregate=[];keys=['actual_good_probability','same_X1_feature_upper_bound','same_X1_automorphism_upper_bound','mass_on_feature_limited_states','mass_on_extra_feature_loss_states','mass_on_automorphism_limited_states']
for arm in ['initial','average','cagrad','same_norm_average']:
    for rd in ['raw','EMA']:
        rows=[r for r in rr if r['arm']==arm and r['readout']==rd]
        if rows:aggregate.append(dict(arm=arm,readout=rd,n=len(rows),**{k:float(np.mean([float(r[k]) for r in rows])) for k in keys}))
(OUT/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
for p,h in prot['bindings'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==json.loads((OUT/'freeze.json').read_text())['sha256']
result=dict(passed=True,states=64,linear_program_certificates=counts,independent_features_reconstructed=True,probability_bound_attainers_verified=True,bindings_unchanged=True,seconds=time.time()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,GPU_hours=0)
(OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));print(json.dumps(example))
