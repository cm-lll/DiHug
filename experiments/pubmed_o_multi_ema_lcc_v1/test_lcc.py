import itertools,pathlib,sys
import numpy as np
import pytest
import metrics as M
from analyze import mean,difference
ROOT=pathlib.Path(__file__).parent
F=M.frozen(ROOT/'vendor/frozen_metrics.py',ROOT/'vendor/aligned_evaluate.py',ROOT/'vendor/hgen_utils.py')
SCHEMA=np.array(list(itertools.combinations_with_replacement(range(4),2)))
NAMES=['Gene','Disease','Chemical','Species']
def graph(n,edges,types=None,ids=None):return M.make_graph(np.zeros(n,int) if types is None else types,np.array(edges,int).reshape(-1,2).T,SCHEMA,ids)
def stats(g):return M.stats(g,F,NAMES,lambda:None)

def test_true_lcc_not_only_isolate_removal_and_no_new_edge():
    g=graph(7,[(0,1),(1,2),(2,3),(4,5)]);k,m=M.lcc(g)
    assert k['original_ids'].tolist()==[0,1,2,3] and k['edges'].shape[1]==3
    assert sorted(x['size'] for x in m['excluded'])==[1,2]
    np.testing.assert_array_equal(k['degree'],g['degree'][m['selected_local_ids']])

def test_storage_id_tie_is_fixed_and_reported():
    g=graph(4,[(0,1),(2,3)],ids=[20,5,30,10]);k,m=M.lcc(g)
    assert sorted(k['original_ids'])==[5,20] and m['tie_count']==2 and len(m['tied_components'])==2

def test_tie_mapping_survives_storage_reindex_with_original_ids_preserved():
    g=graph(4,[(0,1),(2,3)],ids=[30,10,20,5]);k,m=M.lcc(g)
    assert sorted(k['original_ids'])==[5,20]

def test_mapping_types_relations_and_actual_capacity():
    g=graph(6,[(0,1),(1,2),(3,4)],types=[0,1,2,0,3,3]);k,m=M.lcc(g)
    np.testing.assert_array_equal(k['types'],[0,1,2]);np.testing.assert_array_equal(k['families'],g['families'][:2])
    cap=M.capacity(k);assert cap[0]==0 and cap[1]==1 and cap[-1]==0 and sum(cap)==3

def test_reference_connected_unchanged():
    g=graph(4,[(0,1),(1,2),(2,3)],types=[0,1,2,3]);k,m=M.lcc(g)
    np.testing.assert_array_equal(k['edges'],g['edges']);np.testing.assert_array_equal(k['types'],g['types'])

def test_paths_labels_and_counts_direct_summary():
    s=stats(graph(4,[(0,1),(1,2),(2,3)]))
    assert [x['total'] for x in s['mp']]==[3,2,1]
    assert s['typed_walks'][0]['total']==6 and s['typed_walks'][1]['total']==10

def test_square_is_not_legacy_hgen_clique():
    s=stats(graph(4,[(0,1),(1,2),(2,3),(0,3)]));assert s['four_cycles']==1 and s['HGEN_square_legacy_K4']==0
    s=stats(graph(4,list(itertools.combinations(range(4),2))));assert s['four_cycles']==3 and s['HGEN_square_legacy_K4']==1

def test_closure_zero_opportunities_stay_undefined():
    s=stats(graph(2,[(0,1)]));r=s['typed_twohop_closure'][str((0,0,0))]
    assert r['path_ratio'] is None and r['unique_pair_ratio'] is None and s['transitivity'] is None

def test_triangle_opportunities_closed_counts():
    s=stats(graph(3,[(0,1),(1,2),(0,2)],types=[0,1,2]));assert s['triangles']==1 and s['wedge_count']==3 and s['transitivity']==1
    assert sum(x['path_numerator'] for x in s['typed_twohop_closure'].values())==3

def test_missing_type_does_not_become_zero_degree_distribution():
    ref=graph(4,[(0,1),(1,2),(2,3)],types=[0,1,2,3]);g=graph(3,[(0,1),(1,2)],types=[0,1,2]);r=M.compare(g,stats(g),ref,stats(ref),F,NAMES)
    assert r['missing_types']==['Species'] and r['main_incident4_W1'] is None and r['main_joint_degree_TV'] is None
    assert r['typed_relation_W1']['Species']==[None]*10

def test_fixed_reference_weight_scale_all10_relation():
    ref=graph(5,[(0,1),(1,2),(2,3),(3,4)],types=[0,1,2,3,0]);g=graph(4,[(0,1),(1,2),(2,3)],types=[0,1,2,3]);r=M.compare(g,stats(g),ref,stats(ref),F,NAMES)
    assert r['all10_compat_W1']==pytest.approx(.4*r['main_incident4_W1'])
    assert r['main_incident4_W1']!=r['incident4_actual_typeweights']

def test_gini_legacy_shift_and_no_null_averaging():
    s=stats(graph(4,[(0,1),(1,2),(2,3)]));assert s['HGEN_gini_legacy_zero_based']==pytest.approx(s['gini_standard']-2/4)
    assert mean([1,None]) is None and difference(None,2) is None

def test_no_torch_or_model_imported():assert 'torch' not in sys.modules
