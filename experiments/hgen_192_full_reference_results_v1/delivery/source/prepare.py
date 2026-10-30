"""Only this192-node reference supplies paths and Hin2Vec training material."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,time,shutil,traceback
import numpy as np
import pandas as pd
import networkx as nx
import torch
from common import *
torch.set_num_threads(1)
def prepare():
 begun=time.time();rows=[]
 for d,expected in PROTOCOL['datasets'].items():
  check_time(900);folder=OUT/'data'/d;folder.mkdir(parents=True,exist_ok=False)
  ref=PROJECT/'results/count_aux_results_closeout_20261030_final/reference'/(d+'.npz');z=np.load(str(ref),allow_pickle=False);types=z['types'];edges=z['edges'];schema=z['schema']
  assert len(types)==expected['nodes'] and len(edges)==expected['edges'] and edges.shape[1]==2
  graph=nx.Graph();graph.add_nodes_from(range(len(types)));graph.add_edges_from(edges)
  assert sorted(map(len,nx.connected_components(graph)),reverse=True)==expected['components']
  assert not list(nx.isolates(graph));shutil.copy2(str(ref),str(folder/'reference.npz'))
  adj=[sorted(graph.neighbors(i)) for i in range(len(types))];banks=[[],[],[]]
  def visit(seq):
   if len(seq)>1:banks[len(seq)-2].append(seq.copy())
   if len(seq)==4:return
   for v in adj[seq[-1]]:
    if v not in seq:visit(seq+[v])
  for i in range(len(types)):visit([i])
  parts=[]
  for length,bank in enumerate(banks,2):
   arr=np.asarray(bank,dtype=np.int64);assert arr.shape[1]==length
   padded=np.full((len(arr),4),len(types),np.int64);padded[:,:length]=arr;parts.append(padded)
  paths=np.concatenate(parts);assert len(paths)>=24000;np.save(str(folder/'all_paths.npy'),paths)
  for s in PROTOCOL['train_seeds'][d]:
   ix=np.random.Generator(np.random.PCG64(s)).choice(len(paths),24000,replace=False);nodes=paths[ix];tt=np.full_like(nodes,4);valid=nodes<len(types);tt[valid]=types[nodes[valid]]
   np.savez_compressed(str(folder/('bank'+str(s)+'.npz')),nodes=nodes,types=tt,source_indices=ix)
  # Use the HGEN-linked native HIN2vec implementation, no synthetic embeddings.
  walker=module(ROOT/'vendor/hin2vec/walker.py','frozen_hin2vec_walker');model=module(ROOT/'vendor/hin2vec/model.py','frozen_hin2vec_model')
  seed(PROTOCOL['embedding_seeds'][d]);df=pd.DataFrame([dict(source_node=int(u),dest_node=int(v),source_class=int(types[u]),dest_class=int(types[v]),edge_class=str(types[u])+'-'+str(types[v]),weight=1.) for u,v in edges])
  hin=walker.load_a_HIN_from_pandas([df]);hin.window=5;examples=hin.sample(10,8);dataset=model.NSTrainSet(examples,hin.node_size,neg=8)
  assert set(hin.id2node.values())==set(range(192)) and hin.node_size==192
  np.savez_compressed(str(folder/'hin2vec_examples.npz'),x=dataset.x.numpy(),y=dataset.y.numpy())
  save(folder/'hin2vec_mapping.json',dict(node2id={int(k):int(v) for k,v in hin._node2id.items()},path_size=hin.path_size,seed=PROTOCOL['embedding_seeds'][d],reference_sha256=sha(ref)))
  rows.append(dict(dataset=d,nodes=len(types),edges=len(edges),components=expected['components'],reference_sha256=sha(ref),ordered_path_counts=[len(b) for b in banks],total_paths=len(paths),Hin2Vec_examples=len(dataset),Hin2Vec_batches_per_epoch=(len(dataset)+31)//32,window=5,train_only_this_reference=True))
 save(OUT/'input_gate.json',dict(status='passed',datasets=rows,seconds=time.time()-begun,no_LCC_crop=True,no_edge_split=True,embedding_fitted=False))
 print(rows)
if __name__=='__main__':
 try:prepare()
 except BaseException:save(OUT/'prepare_failure.json',dict(traceback=traceback.format_exc(),epoch=time.time()));raise
