"""Replay only the predeclared CPU noise inputs; no replacement seeds or models."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/joint_degree_response_20261109';OLD=ROOT/'results/hard_structure_response_20261108T182154'
spec=importlib.util.spec_from_file_location('archived_helpers',ROOT/'results/output_boundary_replay_20261108T081654/worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
g,np,torch=w.g,w.np,w.torch;torch.set_num_threads(1)
asset=json.loads((ROOT/'results/joint_degree_response_plan_20261109/assets.json').read_text())[0]
s=torch.load(asset['checkpoint'],map_location='cpu',weights_only=False)['state'];cfg=s['binding']['config']
data=ROOT/g.SCI.relative_to(g.ROOT)/'package'/cfg['datasets']['PubMed']['file']
with np.load(data) as z:case=g.common.make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
old=torch.load(OLD/'PubMed_fixed_inputs.pt',map_location='cpu',weights_only=False)['inputs'];oldhashes={w.input_hash([v[2]]):v[0] for v in old}
rows=[];inputs={}
def edgekeys(q):return set((q[0]*192+q[1]).tolist())
reference=edgekeys(case['clean']) if isinstance(case['clean'],np.ndarray) and case['clean'].ndim==2 else None
for key,seed in asset['evaluation_noise_seeds'].items():
 t=int(key.split('_')[0][1:]);rng=np.random.default_rng(seed);initial=json.dumps(rng.bit_generator.state,sort_keys=True)
 v=w.pair_inputs(case,cfg,10000+t,rng,t)[0];h=w.input_hash([v]);ending=json.dumps(rng.bit_generator.state,sort_keys=True)
 matches=[r['key'] for r in rows if r['hash']==h];prior=[x for x in old if x[0]==t][0][2]
 exact={k:bool(np.array_equal(v[k],prior[k])) for k in ['t','ids','f','query','labels','edges','edge_families','ce_weight','round_id']}
 rows.append(dict(key=key,t=t,seed=seed,hash=h,matches_old_t=oldhashes.get(h),matches_predeclared=matches,old_fields_equal=exact,rng_consumed=initial!=ending,rng_initial_sha=hashlib.sha256(initial.encode()).hexdigest(),rng_final_sha=hashlib.sha256(ending.encode()).hexdigest(),edge_count=int(v['edges'].shape[1]),changed_edges_from_old=len(edgekeys(v['edges'])^edgekeys(prior['edges']))));inputs[key]=v
(OUT/'input_collision_audit.json').write_text(json.dumps(rows,indent=2));torch.save(dict(inputs=inputs,metadata=rows),OUT/'frozen_evaluation_inputs_cpu.pt')
print(json.dumps(rows,indent=2))
