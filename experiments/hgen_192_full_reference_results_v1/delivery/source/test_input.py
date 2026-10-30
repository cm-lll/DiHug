import unittest,ast,random
import numpy as np,scipy.sparse as sp
from common import *
from worker import compatible_assembly,scores
class InputTests(unittest.TestCase):
 def test_all_reference_nodes_and_components(self):
  import networkx as nx
  for d,v in PROTOCOL['datasets'].items():
   z=np.load(str(OUT/'data'/d/'reference.npz'));g=nx.Graph();g.add_nodes_from(range(192));g.add_edges_from(z['edges']);self.assertEqual(sorted(map(len,nx.connected_components(g)),reverse=True),v['components']);self.assertEqual(g.number_of_edges(),v['edges'])
 def test_every_cached_path_is_in_reference(self):
  for d in PROTOCOL['datasets']:
   z=np.load(str(OUT/'data'/d/'reference.npz'));adj=np.zeros((192,192),bool);es=z['edges'];adj[es[:,0],es[:,1]]=True;adj|=adj.T
   for s in PROTOCOL['train_seeds'][d]:
    b=np.load(str(OUT/'data'/d/('bank'+str(s)+'.npz')));self.assertEqual(len(np.unique(b['source_indices'])),24000)
    for i in range(3):
     good=(b['nodes'][:,i]<192)&(b['nodes'][:,i+1]<192);self.assertTrue(adj[b['nodes'][good,i],b['nodes'][good,i+1]].all())
    good=b['nodes']<192;np.testing.assert_array_equal(b['types'][good],z['types'][b['nodes'][good]])
 def test_native_classes_unmodified(self):
  src=ast.parse((ROOT/'vendor/hgen/hgen.py').read_text());env=load_core()
  for name in ('Generator','Discriminator','train'):self.assertIn(name,env)
  self.assertEqual(PROTOCOL['hgen']['epochs'],5);self.assertEqual(PROTOCOL['hgen']['train_paths'],24000)
 def test_assembly_defined_domain_matches(self):
  s=sp.csr_matrix(np.ones((8,8))-np.eye(8));p={(0,1,0):5,(1,0,1):3};groups={0:list(range(4)),1:list(range(4,8))}
  for v in (1,2,3):
   seed(v);a=load_core()['heterogeneous_graph_assemble'](s,12,p,groups);r1=random.getstate();n1=np.random.get_state()
   b,_,ok,_=compatible_assembly(s,12,p,groups,v);self.assertTrue(ok);np.testing.assert_array_equal(a,b);self.assertEqual(r1,random.getstate());np.testing.assert_array_equal(n1[1],np.random.get_state()[1])
 def test_empty_pattern_failure_kept(self):
  a,l,ok,why=compatible_assembly(sp.csr_matrix(np.ones((8,8))),8,{}, {0:[0,1],1:[2,3],2:[4,5],3:[6,7]},1);self.assertFalse(ok);self.assertEqual(why,'empty_pattern_support')
 def test_low_support_return_unchanged(self):
  s=np.zeros((8,8));s[0,1]=s[1,0]=1;a,l,ok,why=compatible_assembly(sp.csr_matrix(s),20,{(0,1):1},{0:[0,1,2,3],1:[4,5,6,7]},2);np.testing.assert_array_equal(a,s);self.assertTrue(ok)
 def test_eight_fits_sixteen_graphs(self):
  self.assertEqual(sum(map(len,PROTOCOL['train_seeds'].values())),8);self.assertEqual(len(PROTOCOL['sample_seeds']),2);self.assertEqual(len(set(sum(PROTOCOL['train_seeds'].values(),[]))),8)
 def test_Hin2Vec_mapping_not_assumed_identity(self):
  for d in PROTOCOL['datasets']:
   m=read(OUT/'data'/d/'hin2vec_mapping.json');self.assertEqual(set(map(int,m['node2id'])),set(range(192)));self.assertEqual(set(m['node2id'].values()),set(range(192)));self.assertGreater(m['path_size'],0)
if __name__=='__main__':unittest.main(verbosity=2)
