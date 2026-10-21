import os,sys,time,json,hashlib
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
HERE=Path(__file__).resolve().parent;EXP=HERE.parent
sys.path.insert(0,str(EXP/'role_newton_generation_v1'))
import generation_common as g
c=g.c;np=g.np;torch=g.torch;cap=g.old.cap;solver=g.newton
OUT=Path('/data2/lyh/gnn_project/DiHug/results/role_newton_large_model_20261021_v1')
read=g.read;save=g.save;sha=g.sha
SEED=2026102147;NOISE_SEED=2026102148;DRAW_SEED=2026102149
def check():
    if (OUT/'STOP').exists():raise InterruptedError('STOP')
    if time.time()>read(OUT/'activation.json')['deadline_epoch']-720:raise TimeoutError('60min ledger; reserve12min audit/publication')
def dataset(name):
    inv=read(EXP/'role_target_scalability_v1/evidence/large_graph_inventory.json')[name]
    path=EXP/'joint_cn_dual_rounds_v1_1/package/data'/f'{name}.npz';assert sha(path)==inv['sha256']
    cfg=read(EXP/f'joint_cn_dual_rounds_v1_1/package/configs/{name}.json');assert cfg['model']==c.settings()['model'] and cfg['T']==1000 and cfg['rounds']==20
    with np.load(path,allow_pickle=False) as z:types=z['types'];edges=z['edges'].T;pairs=tuple(map(tuple,z['schema']));fam=z['relations']-1
    case=c.base.c.make_case(types,edges,pairs);target=c.base.extract_target(case)
    np.testing.assert_array_equal(case['space'].encode(edges)[0],fam)
    assert len(types)==inv['nodes'] and edges.shape[1]==inv['physical_edges'] and np.bincount(types).tolist()==inv['type_counts']
    for typ,rows in enumerate(target.roles):assert [(list(v),n) for v,n in rows]==[(v['vector'],v['frequency']) for v in inv['types'][typ]['roles']]
    return case,target,cfg,inv,path
def make_model(case,cfg):
    c.base.c.seed_all(SEED)
    model=c.base.CapacityDenoiser(case['n_types'],case['n_families'],family_pairs=case['space'].pairs,prior=case['prior'],T=1000,**cfg['model']).eval()
    assert np.array_equal(model.prior.numpy(),case['prior']) and model.branch.T==1000
    return model
def state_digest(model):return c.base.sha([(k,hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()) for k,v in model.state_dict().items()])
def frozen():
    f=read(OUT/'freeze.json')
    for n,h in f['source'].items():assert sha(EXP/n)==h,n
    for x in f['inputs']:assert sha(OUT/x['path'])==x['sha256']
    assert sha(OUT/'activation.json')==f['activation_sha256']
    return f
