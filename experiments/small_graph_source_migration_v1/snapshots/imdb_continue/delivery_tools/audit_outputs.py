"""CPU-only independent graph/aggregation audit after frozen evaluator finishes."""
import pathlib,json,csv,math,hashlib,sys,time
import numpy as np
import networkx as nx
OUT=pathlib.Path(__file__).parent
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 r=read(OUT/'evaluation.json');idx={};proof=[]
 reference=np.load(OUT/'reference/IMDB.npz',allow_pickle=False)
 for rec in r['records']:
  key=tuple(rec[k] for k in ('dataset','train_seed','arm','checkpoint','sample_seed','version'))
  assert key not in idx;idx[key]=rec
  f=OUT/'run'/f"IMDB_s{rec['train_seed']}_{rec['arm']}"/f"u{rec['checkpoint']}"/'ema_0.99'/f"sample_{rec['sample_seed']}"/'graph.npz'
  g=np.load(f,allow_pickle=False);edges=g['edges'];types=g['types'];schema=g['schema'];original=np.arange(len(types))
  np.testing.assert_array_equal(types,reference['types']);np.testing.assert_array_equal(schema,reference['schema'])
  paired_initial=None
  if rec['version']=='original' and rec['checkpoint'] in (6000,):
   binding=read(f.parent/'binding.json');parent=pathlib.Path('/data2/lyh/gnn_project/DiHug/results/count_aux_length_20261029T231348/run')/f"IMDB_s{rec['train_seed']}_{rec['arm']}"/'u3000/ema_0.99'/f"sample_{rec['sample_seed']}"/'binding.json'
   base=read(parent)
   assert binding['initial_state_sha256']==base['initial_state_sha256'] and binding['initial_rng']==base['initial_rng']
   steps=[json.loads(line) for line in (f.parent/'steps.jsonl').read_text().splitlines()]
   assert [row['time'] for row in steps]==list(range(1000,0,-1))
   assert all(row['blocks']==1 and row['covered_slots']==13120 and row['exact_coverage'] for row in steps)
   paired_initial=dict(initial_state_exact=True,initial_rng_exact=True,full1000steps=True,all13120slots_each_step=True,seconds=binding['seconds'])
  if rec['version']=='lcc':
   target=OUT/'run/derived_lcc_evaluation'/f"IMDB_s{rec['train_seed']}_{rec['arm']}_u{rec['checkpoint']}_ema_0.99_{rec['sample_seed']}"
   mp=read(target/'mapping.json');kg=np.load(target/'lcc.npz');new_to_old=kg['new_to_original'];orig_graph=nx.Graph();orig_graph.add_nodes_from(range(len(types)));orig_graph.add_edges_from(edges.T)
   expected=sorted([sorted(c) for c in nx.connected_components(orig_graph)],key=lambda c:(-len(c),c))[0]
   assert expected==new_to_old.tolist();mapped=new_to_old[kg['edges']]
   kept={tuple(e) for e in edges.T if e[0] in expected and e[1] in expected}
   assert set(map(tuple,mapped.T))==kept
   np.testing.assert_array_equal(kg['types'],types[new_to_old]);types=kg['types'];edges=kg['edges']
   assert sha(target/'lcc.npz')==mp['derived_sha256']
  graph=nx.Graph();graph.add_nodes_from(range(len(types)));graph.add_edges_from(edges.T)
  assert graph.number_of_edges()==edges.shape[1] and nx.number_of_selfloops(graph)==0
  raw=rec['raw'];assert raw['nodes']==len(types) and raw['edges']==edges.shape[1]
  assert raw['triangles']==sum(nx.triangles(graph).values())//3
  assert raw['components']==nx.number_connected_components(graph)
  assert raw['lcc']==len(max(nx.connected_components(graph),key=len))
  assert raw['isolates']==len(list(nx.isolates(graph)))
  assert raw['maximum_degree']==max(dict(graph.degree()).values())
  lookup={tuple(p):i for i,p in enumerate(schema)};families=np.array([lookup[tuple(sorted([int(types[u]),int(types[v])]))] for u,v in edges.T],int)
  for k in range(10):assert raw[f'family{k}_edges']==int((families==k).sum())
  for k,v in rec['absolute_error'].items():
   a,b=raw[k],rec['reference'].get(k)
   assert (v is None)==(a is None or b is None)
   if v is not None:assert math.isclose(v,abs(a-b),abs_tol=1e-12)
  d=rec['distribution_error'];a=d['main_incident4_W1'];b=d['all10_compat_W1']
  assert (a is None)==(b is None)
  if a is not None:assert math.isclose(b,.4*a,abs_tol=1e-12,rel_tol=1e-12)
  assert sum(x['path_denominator'] for x in rec['statistics']['typed_twohop_closure'].values())==raw['wedge_count']
  assert sum(x['path_numerator'] for x in rec['statistics']['typed_twohop_closure'].values())==3*raw['triangles']
  for u in (1,2,3):assert raw[f'MP{u}_total']==rec['statistics']['mp'][u-1]['total']
  proof.append(dict(identity=list(key),graph_sha256=sha(f),node_edge_triangle_component_checks=True,lcc_induced_mapping=rec['version']=='lcc',paired_sampling=paired_initial))
 for row in r['cell_means']:
  vals=[idx.get(('IMDB',row['train_seed'],row['arm'],row['checkpoint'],s,row['version']),{}).get(row['group'],{}).get(row['metric']) for s in (2026102821,2026102822)]
  expect=sum(vals)/2 if all(v is not None for v in vals) else None
  if expect is None:assert row['value'] is None
  else:assert math.isclose(row['value'],expect,abs_tol=1e-10,rel_tol=1e-12)
 for row in r['paired']:
  args=('IMDB',row['train_seed']);tail=(row['checkpoint'],row['sample_seed'],row['version'])
  a=idx.get(args+('CE_COUNT',)+tail,{}).get(row['group'],{}).get(row['metric']);b=idx.get(args+('CE',)+tail,{}).get(row['group'],{}).get(row['metric'])
  if a is None or b is None:assert row['difference'] is None
  else:assert math.isclose(row['difference'],a-b,abs_tol=1e-10,rel_tol=1e-12)
 checks=dict(status='passed',records=len(proof),per_graph=proof,mean_rows=len(r['cell_means']),paired_rows=len(r['paired']),
  new_chains=sum(i['status']=='complete' and i['job']['checkpoint'] in (6000,) for i in r['inventory']),
  checks=['independent graph counts','typed relation counts','reference absolute errors per graph','W1 0.4 relation','MP labels','closure numerator denominator','LCC exact induced mapping','means require both samples','paired differences by full identity'],optimizer_steps=0,model_forwards=0,epoch=time.time())
 (OUT/'independent_audit.json').write_text(json.dumps(checks,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in checks.items() if k!='per_graph'}))
if __name__=='__main__':main()

