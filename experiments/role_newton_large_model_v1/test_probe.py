import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import large_common as l
import pytest
def test_source_configs_are_exact_same_existing_widths():
    for name in ('IMDB','PubMed'):
        c=l.read(l.EXP/f'joint_cn_dual_rounds_v1_1/package/configs/{name}.json')
        assert c['model']==l.c.settings()['model'] and (c['T'],c['rounds'])==(1000,20)
def test_full_scale_query_plan_has_no_holes():
    for name in ('IMDB','PubMed'):
        c,t,cfg,inv,p=l.dataset(name);plan=l.c.base.c.plan_for(c,[l.DRAW_SEED,20,61],20)
        q=l.c.base.c.query_inventory(c,plan)
        assert q['active_slots']==sum(q['query_sizes']) and q['rounds']==20
        assert len(c['space'].types)==inv['nodes']
def test_numerical_newton_gate_rejects_bad_row():
    p,lp,r=l.solver.solve(l.np.zeros((4,2)),l.np.array([1,3]));l.g.verify(p,lp,l.np.zeros((4,2)),l.np.array([1,3]),r)
    p[0,0]+=.1
    with pytest.raises(AssertionError):l.g.verify(p,lp,l.np.zeros((4,2)),l.np.array([1,3]),r)
def test_instrumented_step_same_as_original_Newton(tmp_path):
    from probe import Measurement
    types=l.np.array([0,0,1,1,2,2,2,3,3]);edges=l.np.array([[0,0,1,2,3,4,5],[2,4,5,4,6,7,8]])
    import itertools
    case=l.c.base.c.make_case(types,edges,tuple(itertools.combinations_with_replacement(range(4),2)));target=l.c.base.extract_target(case)
    cfg=l.read(l.EXP/'joint_cn_dual_rounds_v1_1/package/configs/IMDB.json');m=l.make_model(case,cfg);m.set_target(target)
    state=l.np.zeros(case['space'].total,l.np.uint8);f,loc=case['space'].encode(edges);state[case['space'].offsets[f]+loc]=1
    kernel=l.c.base.MarginalKernel(case['prior'],l.c.base.c.schedule(1000));original=l.cap.allocation_marginals
    l.cap.allocation_marginals=lambda L,c:l.solver.required(L,c,lambda:None)
    try:
        with l.torch.no_grad():a,aa=l.c.base.reverse_step(m,case,state,kernel,500,20,123,l.np.random.default_rng(123),'cpu',trace=True)
    finally:l.cap.allocation_marginals=original
    m.branch.tables.clear();mon=Measurement(m,tmp_path,lambda:None);mon.case=case;mon.install()
    try:
        with l.torch.no_grad():b,bb=l.c.base.reverse_step(m,case,state,kernel,500,20,123,l.np.random.default_rng(123),'cpu',trace=True)
        for key in ('p0','q','uniform','state','coverage'):l.np.testing.assert_array_equal(aa['trace'][key],bb['trace'][key])
        plan=l.c.base.c.plan_for(case,[123,500,61],20)
        assert len(mon.reads)==4 and bb['blocks']==sum(bool(len(plan.block(j)[0])) for j in range(20))
    finally:mon.close()
