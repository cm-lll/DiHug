"""CPU-only saved-output audit and nonlearned CN scale/mapping evidence."""
import pathlib,json,csv,hashlib,itertools,collections
import numpy as np
B=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,rows):
 with (B/name).open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
panel=read(B/'frozen_panel.json');receipt=read(B/'forward/receipt.json')
assert receipt['completed_full_queries']==receipt['attempted_full_queries']==8
assert receipt['optimizer_steps']==receipt['backwards']==receipt['sampled_graphs']==0
checks=collections.Counter();scale=[];compact=[]
validation=list(csv.DictReader((B/'fixed_validation.csv').open()))
for dataset in ['IMDB','PubMed']:
 for t in ['1','500','1000']:
  subset=[x for x in validation if x['dataset']==dataset and x['t']==t]
  assert len({x['input_sha256'] for x in subset})==1
  checks['validation_same_input_all_arms_checkpoints_readouts']+=1
source=B/'evidence/scientific_source'
source_files=['hsgd/common_neighbor.py','hsgd/model.py','hsgd/computational.py','hsgd/vendor/sample_edges_utils_typed.py','model_variants.py']
for item in panel['inputs']:
 x=dict(np.load(B/'inputs'/(item['name']+'.npz')));assert sha(B/'inputs'/(item['name']+'.npz'))==item['sha256']
 ids=x['ids'];assert np.array_equal(np.sort(ids),np.arange(18336));checks['input_full_coverage']+=1
 for arm in ['C','E']:
  file=B/'forward'/(arm+'_'+item['name']+'.npz');r=read(file.with_suffix('.json'));assert sha(file)==r['output_sha256']
  z=dict(np.load(file));assert np.array_equal(z['query'][:,ids],x['query']);assert np.array_equal(z['family'][ids],x['f']);assert np.array_equal(z['state'],x['state']);checks['input_mapping_and_output_hash']+=1
  for key in ['p0','q']:assert np.isfinite(z[key]).all() and ((z[key]>=0)&(z[key]<=1)).all()
  assert np.isfinite(z['logits']).all();checks['finite_probability_support']+=1
  s=z['state'].astype(bool);assert np.isclose(z['q'].sum()-s.sum(),z['q'][~s].sum()-(1-z['q'][s]).sum(),atol=1e-10,rtol=0);checks['net_identity']+=1
  if int(z['t'])==1:np.testing.assert_allclose(z['p0'],z['q'],atol=1e-12,rtol=0);checks['t1_q_equals_p0']+=1
  if r['historical_comparison']:assert r['historical_comparison']['ce_aggregate_pass'] and r['historical_comparison']['p0_aggregate_pass'];checks['historical_aggregate_reproduction']+=1
 z=dict(np.load(B/'forward'/('C_'+item['name']+'.npz')));u,v=z['query'];A=np.zeros((192,192),dtype=np.int64);s=z['state'].astype(bool);A[u[s],v[s]]=1;A[v[s],u[s]]=1
 cn=np.stack([(A[:,z['types']==typ]@A[z['types']==typ,:])[u,v] for typ in range(4)],1)
 # Independent set intersections verify endpoint correspondence; no model called.
 adj=[set(np.flatnonzero(a)) for a in A]
 for k,(a,b) in enumerate(zip(u,v)):
  common=adj[a]&adj[b];expected=np.bincount(z['types'][list(common)],minlength=4)
  assert np.array_equal(cn[k],expected)
 checks['typed_CN_slot_set_intersections']+=len(u)
 for typ in range(4):
  c=cn[:,typ];l=np.log1p(c)
  scale.append(dict(input=item['name'],middle_type=typ,slots=len(c),nonzero_slots=int((c>0).sum()),raw_max=int(c.max()),raw_mean=float(c.mean()),log1p_max=float(l.max()),log1p_mean=float(l.mean()),log1p_rms=float(np.sqrt(np.mean(l*l))),origin='CPU topology only; not learned vector'))
 assert np.array_equal(cn.sum(1),(A@A)[u,v]);checks['CN_total_consistency']+=1
rows=list(csv.DictReader((B/'cross_slot_strata.csv').open()))
for inp in [x['name'] for x in panel['inputs']]:
 for axis in ['cn_all','min_degree','min_relation_degree']:
  groups=collections.defaultdict(list)
  for x in rows:
   if x['input']==inp and x['axis']==axis:groups[(x['current_edge'],x['bin'])].append(x)
  for (edge,bin_),xs in groups.items():
   compact.append(dict(input=inp,axis=axis,current_edge=int(edge),bin=bin_,slots=sum(int(x['slots']) for x in xs),delta_p0_sum=sum(float(x['delta_p0_sum']) for x in xs),delta_q_sum=sum(float(x['delta_q_sum']) for x in xs)))
 for fam in range(10):
  for edge in [0,1]:
   for axis in sorted({x['axis'] for x in rows}):
    xs=[x for x in rows if x['input']==inp and int(x['family'])==fam and int(x['current_edge'])==edge and x['axis']==axis]
    z=dict(np.load(B/'forward'/('paired_'+inp+'.npz')));mask=(z['family']==fam)&(z['state']==edge)
    assert sum(int(x['slots']) for x in xs)==int(mask.sum())
    assert np.isclose(sum(float(x['delta_q_sum']) for x in xs),z['delta_q'][mask].sum(),atol=1e-12,rtol=0)
    checks['stratum_partition']+=1
for row in csv.DictReader((B/'closure_decomposition.csv').open()):
 if not row['reason']:
  assert abs(float(row['delta_closed'])-float(row['opportunity_component'])-float(row['rate_component']))<1e-9;checks['closure_algebra']+=1
write('cn_count_scales.csv',scale);write('strata_all_relations.csv',compact)
audit=dict(status='passed',checks=dict(checks),model_forward_calls=0,optimizer_steps=0,new_graphs=0,source_sha256={name:sha(source/name) for name in source_files},missing=['learned CN32 activations and joint-head block RMS were not captured during the already completed eight forwards; no extra forward permitted'],CN_dimensions=dict(phi_input=281,phi_hidden=32,phi_output=32,mean=32,typed_log_counts=4,existence=1,total=37,joint_input=776,C_joint_input=739,joint_hidden=128,joint_output=2))
(B/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
