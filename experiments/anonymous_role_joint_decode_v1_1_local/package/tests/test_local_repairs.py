import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from joint_decode.exact import draw
from joint_decode.frozen_input import Frozen,canonical_map
from joint_decode.contracts import Schema,EdgeLaw
from joint_decode.role_assignment import match
from joint_decode.structure import evaluate,differences
from audit_results import audit_fixed_graph
sys.path.insert(0,str(Path(__file__).parent))
from fixture_export import fixture,sha,write_json
ROOT=Path(__file__).resolve().parents[1]


def test_release_preserves_authenticated_nested_fixture_manifest():
    # Packaging regression: excluding every basename manifest.json removed the
    # fixture's trust-root file, even though ordinary source tests still passed.
    manifest=json.loads((ROOT/'manifest.json').read_text())
    path='evidence/SYNTHETIC_input/manifest.json'
    assert (ROOT/path).is_file() and path in manifest['files']
    pins=json.loads((ROOT/'evidence/SYNTHETIC_input/fixture_pins.json').read_text())
    assert manifest['files'][path]['sha256']==pins['manifest_sha256']==sha(ROOT/path)


def mutate(root,key,fn):
    m=json.loads((root/'manifest.json').read_text())
    if key=='manifest':fn(m)
    elif key=='payload':
        with np.load(root/'payload.npz') as z:a={k:z[k].copy() for k in z.files}
        fn(a);np.savez_compressed(root/'payload.npz',**a)
        m['files']['payload']['sha256']=sha(root/'payload.npz')
    else:
        p=root/m['files'][key]['path'];v=json.loads(p.read_text());fn(v);write_json(p,v);m['files'][key]['sha256']=sha(p)
    write_json(root/'manifest.json',m);return sha(root/'manifest.json')


def test_tail_zero_nextafter_regression():
    p=np.array([1/6]*6+[0,0]);x=draw(p,[np.nextafter(1.,0.)]);assert x.tolist()==[5]


@pytest.mark.parametrize('p',[[0,.25,0,.75,0],[0,0,1,0],[.5,0,.5],[1.]])
def test_draw_positive_support_and_boundaries(p):
    u=np.array([0.,.25,.5,.75,np.nextafter(1.,0.)]);x=draw(p,u)
    assert (np.asarray(p)[x]>0).all()
    if p==[.5,0,.5]:assert draw(p,[.5]).tolist()==[2]


@pytest.mark.parametrize('p',[[],[0,0],[-.1,1.1],[np.nan,1],[np.inf],[.2,.7],[[1.]]])
def test_draw_bad_probabilities(p):
    with pytest.raises((ValueError,FloatingPointError)):draw(p,[0.])


@pytest.mark.parametrize('u',[[-.1],[1.],[np.inf],[np.nan]])
def test_draw_bad_uniforms(u):
    with pytest.raises(ValueError):draw([1.],u)


@pytest.mark.parametrize('kind',['bits','code','score','assigned'])
def test_fixed_graph_two_representations_tamper_rejected(kind):
    a=dict(slots=np.array([[0,0,1],[1,2,2]]),families=np.zeros(3,int),types=np.zeros(3,int),
           fixed_assignment_bits=np.array([1,1,0]),assigned_joint_degrees=np.array([[2],[1],[1]]))
    target=Schema(a['types'],[(0,0)]).profile(a['fixed_assignment_bits']);p=np.array([.2,.3,.4])
    case=dict(fixed_assignment_graph_code=3,target=target,probabilities=p.tolist(),fixed_assignment_MAP=dict(log_q=float(np.log([.2,.3,.6]).sum())))
    audit_fixed_graph(case,a)
    if kind=='bits':a['fixed_assignment_bits'][0]^=1
    elif kind=='code':case['fixed_assignment_graph_code']^=1
    elif kind=='score':case['fixed_assignment_MAP']['log_q']+=1
    else:a['assigned_joint_degrees'][0,0]=0
    with pytest.raises(AssertionError):audit_fixed_graph(case,a)


def test_reader_correct_reorder_and_complete_joint_matching(tmp_path):
    p=fixture(tmp_path/'in');q=fixture(tmp_path/'reordered',reorder=True)
    a=Frozen(tmp_path/'in/manifest.json',p['manifest_sha256'],'A');b=Frozen(tmp_path/'reordered/manifest.json',q['manifest_sha256'],'A')
    assert np.array_equal(a.p0,b.p0) and np.array_equal(a.state,b.state)
    assigned,report=match(a.schema,a.target,a.p0)
    assert a.schema.profile(np.array([0]*a.schema.s))!=a.target
    assert report['cost']>=0 and assigned.shape==(6,6) and '合成' in a.source_limit
    from joint_decode.contracts import validate_assignment
    validate_assignment(a.schema,a.target,assigned)


@pytest.mark.parametrize('bad',['missing','duplicate','family','out_of_range','arm','time','guidance','semantic','support','saturation','q','query','seed','prior','target'])
def test_reader_rejects_bad_contract(tmp_path,bad):
    r=tmp_path/'in';pins=fixture(r);expected=pins['manifest_sha256']
    def array_bad(a):
        if bad=='missing':a['slots']=a['slots'][:,:-1]
        if bad=='duplicate':a['slots'][:,1]=a['slots'][:,0]
        if bad=='family':a['families'][0]+=1
        if bad=='out_of_range':a['slots'][1,0]=999
        if bad=='support':a['support'][0]=-1
        if bad=='saturation':a['p0'][np.flatnonzero(a['support']==-1)[0]]=1.
        if bad=='q':a['q'][np.flatnonzero(a['support']==-1)[0]]+=1e-10
        if bad=='query':a['query_ids'][0],a['query_ids'][1]=a['query_ids'][1],a['query_ids'][0]
        if bad=='prior':a['prior'][1]=.4
    if bad in ('arm','time','guidance','semantic','seed'):
        k,v=dict(arm=('arm','C'),time=('t',20),guidance=('unguided',False),semantic=('p0_semantics','absent_probability'),seed=('seed',99))[bad]
        expected=mutate(r,'manifest',lambda m:m.update({k:v}))
    elif bad=='target':expected=mutate(r,'target',lambda t:t.update(reference_sha256='9'*64))
    else:expected=mutate(r,'payload',array_bad)
    with pytest.raises((ValueError,KeyError)):Frozen(r/'manifest.json',expected,'A')


@pytest.mark.parametrize('file',['manifest','payload','receipt'])
def test_reader_expected_hash_chain_not_current_self_hash(tmp_path,file):
    r=tmp_path/'in';pins=fixture(r);p=r/(file+'.json' if file!='payload' else 'payload.npz')
    with p.open('ab') as f:f.write(b' ')
    with pytest.raises(ValueError,match='SHA256'):Frozen(r/'manifest.json',pins['manifest_sha256'],'A')


def test_explained_finite_saturation_remains_free_and_unexplained_is_rejected(tmp_path):
    r=tmp_path/'in';fixture(r)
    def change(a):
        p=a['p0'];free=a['support']==-1;logits=np.zeros_like(p);logits[free]=np.log(p[free])-np.log1p(-p[free]);logits[np.flatnonzero(free)[0]]=1000.
        a['clean_logits']=logits;p[free]=np.exp(-np.logaddexp(0.,-logits[free]));a['q']=p.copy()
    mutate(r,'payload',change);mutate(r,'source_receipt',lambda m:m.update(logits_semantics='finite_clean_present_minus_absent'))
    h=mutate(r,'manifest',lambda m:m.update(score_semantics='finite_clean_logits'))
    f=Frozen(r/'manifest.json',h,'A');i=np.flatnonzero(f.support==-1)[0]
    assert f.p0[i]==1 and f.law.support[i]==-1 and f.law.logits[i]==1000


def test_full_joint_matching_exact_bruteforce_and_not_lexicographic():
    s=Schema(np.array([0,0,1,1,2,2]),[(0,1),(0,2),(1,2)])
    reference=np.zeros(s.s,np.uint8);reference[[0,4,5,8]]=1;target=s.profile(reference)
    p=np.linspace(.1,.8,s.s);d,report=match(s,target,p)
    mu=np.zeros_like(d,dtype=float)
    for endpoint in s.slots:np.add.at(mu,(endpoint,s.families),p)
    for row in report['types']:
        c=row['type'];g=target['groups'][str(c)];roles=np.repeat(g['rows'],g['counts'],axis=0);nodes=np.flatnonzero(s.types==c)
        scale=np.array(row['scales']);valid=row['effective_relations'];k=len(valid)
        costs=[np.mean(np.sum(((mu[nodes][:,valid]-roles[list(perm)][:,valid])/scale[valid])**2,axis=1)/k) if k else 0 for perm in itertools.permutations(range(len(nodes)))]
        assert abs(row['cost']-min(costs))<1e-12


def test_single_dimension_matching_scales_duplicates_and_ties():
    s=Schema(np.array([0,0,1,1]),[(0,1)]);target=s.profile(np.array([1,1,0,0]))
    d,r=match(s,target,np.full(4,.5))
    assert any('sorting' in x['algorithm'] for x in r['types'])
    assert any('different-role tie present'==x['different_role_ties'] for x in r['types'])
    assert r['types'][0]['scales']==[1+np.sqrt(2)]


def test_structure_independent_small_graph_checks(tmp_path):
    r=tmp_path/'in';pins=fixture(r);f=Frozen(r/'manifest.json',pins['manifest_sha256'],'A');s=f.schema
    rng=np.random.default_rng(7)
    for bits in [np.zeros(s.s,np.uint8),np.ones(s.s,np.uint8),*rng.integers(0,2,size=(12,s.s),dtype=np.uint8)]:
        v=evaluate(s,bits,f.semantic_types);adj=np.zeros((6,6),int)
        for present,(u,w) in zip(bits,s.slots.T):adj[u,w]=adj[w,u]=present
        tri=sum(adj[a,b]*adj[a,c]*adj[b,c] for a,b,c in itertools.combinations(range(6),3))
        denom=num=0
        for d,a,m in itertools.product([0,1],[2,3],[4,5]):
            denom+=adj[d,m]*adj[m,a];num+=adj[d,m]*adj[m,a]*adj[d,a]
        assert v['triangles']==tri and v['director_movie_actor_closure']['numerator']==num and v['director_movie_actor_closure']['denominator']==denom
        assert sum(v['component_sizes'])==6 and v['total_degree']['mean']==adj.sum()/6
        assert differences(s,bits,bits)['added']==[0]*s.r


def cli(root,pins,out,**extra):
    args=[sys.executable,str(ROOT/'decode_frozen.py'),'--input',str(root if extra.get('root_input') else root/'manifest.json'),'--manifest-sha256',pins['manifest_sha256'],
          '--arm',extra.get('arm','A'),'--output',str(out),'--max-seconds',str(extra.get('seconds',30))]
    if extra.get('original'):
        args+=['--original-graph',str(root/'original.npz'),'--original-sha256',pins['original_sha256'],
               '--original-receipt',str(root/'original_receipt.json'),'--original-receipt-sha256',pins['original_receipt_sha256']]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run(args,cwd=ROOT,env=env,text=True,capture_output=True,timeout=40)


@pytest.mark.parametrize('arm',['A','B'])
def test_public_local_cli_real_parent_child_solver_end_to_end(tmp_path,arm):
    root=tmp_path/'input';pins=fixture(root,arm=arm);out=tmp_path/'output';res=cli(root,pins,out,original=True,arm=arm,root_input=arm=='B')
    assert res.returncode==0,(res.stdout,res.stderr,(out/'worker.log').read_text())
    summary=json.loads((out/'summary.json').read_text());receipt=json.loads((out/'controller.json').read_text())
    assert summary['status']=='complete' and summary['source_kind']=='synthetic_fixture'
    assert summary['original'] and all(summary['checks'].values()) and receipt['owned_group_gone']
    assert (out/'decoded_graph.npz').exists() and summary['solver']['no_anonymous_global_optimality_claim']


def test_no_original_graph_is_explicitly_missing_not_replaced(tmp_path):
    root=tmp_path/'input';pins=fixture(root,original=False);out=tmp_path/'output';res=cli(root,pins,out)
    assert res.returncode==0,(res.stdout,res.stderr)
    summary=json.loads((out/'summary.json').read_text())
    assert summary['original'] is None and summary['changes_by_family'] is None and summary['missing_comparisons']


def test_matching_assignment_is_not_engineering_lexicographic():
    from joint_decode.contracts import canonical_assignment
    s=Schema(np.array([0,0,1,1]),[(0,1)]);target=s.profile(np.array([1,1,0,0]))
    p=np.array([.9,.8,.2,.1]);d,r=match(s,target,p)
    assert not np.array_equal(d,canonical_assignment(s,target))
    assert np.array_equal(d[:,0],[2,0,1,1])


def test_output_protection_before_mkdir_and_input_hash_failure(tmp_path):
    root=tmp_path/'input';pins=fixture(root);before={p.relative_to(root):sha(p) for p in root.iterdir()}
    res=cli(root,pins,root/'BAD')
    assert res.returncode!=0 and not (root/'BAD').exists() and before=={p.relative_to(root):sha(p) for p in root.iterdir()}
    bad=pins.copy();bad['manifest_sha256']='0'*64;out=tmp_path/'failed';res=cli(root,bad,out)
    assert res.returncode!=0 and json.loads((out/'summary.json').read_text())['status']=='failed' and not (out/'decoded_graph.npz').exists()


@pytest.mark.parametrize('field,value',[('seed',7),('arm','B'),('weight_sha256','f'*64)])
def test_original_receipt_wrong_arm_seed_rejected(tmp_path,field,value):
    root=tmp_path/'in';pins=fixture(root);f=Frozen(root/'manifest.json',pins['manifest_sha256'],'A')
    r=json.loads((root/'original_receipt.json').read_text());r[field]=value;write_json(root/'original_receipt.json',r)
    with pytest.raises(ValueError,match='binding'):f.original(root/'original.npz',pins['original_sha256'],root/'original_receipt.json',sha(root/'original_receipt.json'))


def test_query_recipe_matches_local_old_implementation_without_importing_torch():
    # Trusted local baseline source is exercised directly only for this static
    # recipe equivalence test. It imports numpy/math, never a model/common.py.
    from joint_decode.frozen_input import query_plan
    import importlib.util
    old=ROOT/'format_reference/old_slots.py'
    spec=importlib.util.spec_from_file_location('old_local_slots_reference',old);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    s=Schema(np.array([0,0,1,1,2,2]),[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)])
    active=np.array([False,True,True,False,True,False]);seed=2026101907
    p=query_plan(s,active,seed);oldplan=mod.UniformQueryPlan(mod.TypedSlots(s.types,s.pairs),[seed,1,61],rounds=20,active_families=active)
    h=hashlib.sha256()
    for b in range(20):
        ids,f,q=oldplan.block(b);left,right=p['query_offsets'][b:b+2]
        assert np.array_equal(ids,p['query_ids'][left:right]) and np.array_equal(f,p['query_families'][left:right])
        assert np.array_equal(q,p['query_slots'][:,left:right])
        for a in (ids,f,q):h.update(a.tobytes())
    assert h.hexdigest()==p['query_sha256']


def test_reader_true_fixed_one_and_empty_families(tmp_path):
    from joint_decode.frozen_input import query_plan
    r=tmp_path/'input';fixture(r);s=Schema(np.array([0,1,2]),list(itertools.combinations_with_replacement(range(3),2)))
    bits=np.ones(s.s,np.uint8);target=s.profile(bits);counts=np.bincount(s.families,minlength=s.r)
    prior=(counts>0).astype(float);plan=query_plan(s,counts>0,2026101907)
    def replace(a):
        alpha=a['alpha'];a.clear();a.update(types=s.types,pairs=s.pairs,slots=s.slots,families=s.families,
            state=bits,p0=bits.astype(float),q=bits.astype(float),support=np.ones(s.s,np.int8),prior=prior,alpha=alpha,
            **{k:v for k,v in plan.items() if k!='query_sha256'})
    mutate(r,'payload',replace);mutate(r,'target',lambda a:a.update(profile=target))
    mutate(r,'source_receipt',lambda a:a.update(query_sha256=plan['query_sha256']))
    h=mutate(r,'manifest',lambda a:a.update(query_sha256=plan['query_sha256']))
    f=Frozen(r/'manifest.json',h,'A');assigned,report=match(s,target,f.p0)
    from joint_decode.decoder import decode
    decoded,solver=decode(s,target,assigned,f.law)
    assert solver['optimal'] and np.array_equal(decoded,bits) and np.all(f.support==1)
    assert f.law.log_prob(decoded)==0
