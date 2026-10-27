"""Serialization-only: sorted original IDs, preserving complete exact induced graph."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,json,hashlib,csv,datetime
from pathlib import Path
R=Path(__file__).resolve().parent.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={}
for ds in ['IMDB','PubMed']:
 p=R/'samples'/ds;z=np.load(p/'graph.npz');old={k:z[k] for k in z.files};before=sha(p/'graph.npz');ids=np.sort(old['original_ids']);g=np.load(R/'inputs/reference'/f'{ds}.npz');inv=np.full(len(g['types']),-1,int);inv[ids]=np.arange(192);mask=(inv[g['edges']]>=0).all(1);new={'types':g['types'][ids],'edges':inv[g['edges'][mask]],'relations':g['relations'][mask],'schema':g['schema'],'original_ids':ids}
 es=lambda a:set((min(int(a['original_ids'][u]),int(a['original_ids'][v])),max(int(a['original_ids'][u]),int(a['original_ids'][v])),int(r)) for (u,v),r in zip(a['edges'],a['relations']))
 assert es(old)==es(new);assert np.all(new['edges'][:,0]<new['edges'][:,1]);assert np.array_equal(new['edges'],new['edges'][np.lexsort((new['edges'][:,1],new['edges'][:,0]))])
 np.savez_compressed(p/'graph.npz',**new);np.save(p/'original_node_ids.npy',ids)
 with (p/'node_map.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['node_id','original_node_id','type_id']);w.writerows((i,int(u),int(g['types'][u])) for i,u in enumerate(ids))
 md=json.loads((p/'metadata.json').read_text());md['graph_sha256']=sha(p/'graph.npz');md['original_ids']=ids.tolist();md['serialization']='Sorted original IDs, exact original ordered induced edges; u<v. No graph/metric changes.';baseline=np.load(p/'baseline_original_node_ids.npy');md['unique_nodes_replaced_vs_baseline']=len(set(map(int,ids))-set(map(int,baseline)));md['baseline_node_overlap']=len(set(map(int,ids))&set(map(int,baseline)));(p/'metadata.json').write_text(json.dumps(md,indent=2))
 receipt[ds]={'before_graph_sha256':before,'after_graph_sha256':sha(p/'graph.npz'),'same_original_labeled_edges':True,'original_ids_sorted':True,'canonical_edges_sorted':True,'unique_nodes_replaced_vs_baseline':md['unique_nodes_replaced_vs_baseline']}
(R/'serialization_receipt.json').write_text(json.dumps({'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'graphs':receipt},indent=2));print(json.dumps(receipt,indent=2))
