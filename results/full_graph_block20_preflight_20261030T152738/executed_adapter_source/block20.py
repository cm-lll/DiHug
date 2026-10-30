"""Full-graph CE, historical uniform 1-of-20 query protocol. COUNT disabled."""
import hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'package'))
import numpy as np
import torch
from hsgd.bw_preflight.common import make_case,train_input,plan_for,check_prob_device
from hsgd.computational import FrozenTypedContext
from hsgd.losses import supported_binary_ce
from model_variants import StructuralDenoiser
from early_cn import install

ROUNDS=20
DATA={'IMDB':(1653,4079,'10f1dbbbe55c6a069594b2dcb590d1b8295ca5436320ef4597cd30e6b7f73e42'),
      'PubMed':(1565,12284,'dacff5a392e23fbcfb695076a14a6db5317f4a4382b69af2a7596a1a217163f8')}
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def load_case(dataset):
    n,e,h=DATA[dataset];path=ROOT/'data'/f'{dataset}.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=h:raise ValueError('full original data binding')
    with np.load(path,allow_pickle=False) as z:
        if z['types'].shape!=(n,) or z['edges'].shape!=(e,2):raise ValueError('full graph dimensions')
        case=make_case(z['types'],z['edges'].T,tuple(map(tuple,z['schema'])))
        if not np.array_equal(case['space'].encode(z['edges'].T)[0],z['relations'].astype(np.int64)-1):raise ValueError('relation convention')
    cfg=json.loads((ROOT/'package/config.json').read_text())
    # Preserve model/optimizer/kernel values; replace only intake/query configuration.
    cfg.update(version='full_graph_block20_preflight_v1',rounds=20,formal_authorized=False,seed=2026103001 if dataset=='PubMed' else 2026103002)
    cfg['datasets']={dataset:dict(file=str(path),sha256=h,nodes=n,edges=e)}
    return case,cfg
def make_model(case,cfg,arm='A',basis=None):
    if arm in ('C','E') and basis is None:raise RuntimeError('full-graph dictionary unavailable; no small/synthetic fallback')
    model=StructuralDenoiser(case,cfg,arm,basis)
    if arm=='E':model=install(model)
    return model
def pair_inputs(case,cfg,position,rng,force_t=None):
    out=[]
    for j in range(2):
        pos=2*position+j
        inp=train_input(case,cfg['T'],ROUNDS,cfg['seed'],pos,rng,force_t=force_t)
        inp.update(plan_seed=[cfg['seed'],pos,21],rounds=ROUNDS)
        out.append(inp)
    return out
def validate_input(case,inp):
    if inp['rounds']!=20:raise ValueError('fixed 20 blocks required')
    ids,f,query=plan_for(case,inp['plan_seed'],20).block(inp['round_id'])
    for key,value in [('ids',ids),('f',f),('query',query)]:
        if not np.array_equal(inp[key],value):raise ValueError('query block contract '+key)
    if len(np.unique(ids))!=len(ids) or not np.all(case['active'][f]):raise ValueError('duplicate/inactive query slot')
    if not np.array_equal(inp['labels'],np.isin(ids,case['clean_ids']).astype(np.float32)):raise ValueError('clean labels')
    if inp['ce_weight']!=20/case['space'].total:raise ValueError('whole-schema CE scaling')
    return dict(slots=len(ids),round_id=int(inp['round_id']),query_ids_sha256=digest(ids),Xt_edges=inp['edges'].shape[1],Xt_sha256=digest(inp['edges']),t=inp['t'])
def normal2(model,opt,case,inputs,T,device,phase=None):
    if len(inputs)!=2:raise ValueError('exactly two independent Xt inputs')
    if getattr(model,'count_aux_weight',0.)!=0:raise RuntimeError('COUNT not adapted; nonlinear plug-in estimator forbidden')
    torch.cuda.synchronize(device);start=time.perf_counter();model.train();opt.zero_grad(set_to_none=True);rows=[];loss_sum=0.
    for j,inp in enumerate(inputs):
        row=validate_input(case,inp);token=('full20-normal2',j,id(inp));ctx=FrozenTypedContext(model,case['space'].types,inp['edges'],inp['edge_families'],token)
        try:
            if phase:phase('forward',j,row)
            torch.cuda.synchronize(device);stamp=time.perf_counter()
            graph,logits,p=ctx.predict(inp['query'],inp['f'],inp['t']/T,token,details=True)
            check_prob_device(p,inp['f'],case,model.prior)
            labels=torch.as_tensor(inp['labels'],device=device)
            loss=supported_binary_ce(logits,labels,model.prior[graph.query_families]).sum()*(20./case['space'].total)/2
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite half CE')
            torch.cuda.synchronize(device);row['forward_seconds']=time.perf_counter()-stamp
            row['half_CE']=float(loss.detach());row['computation_directed_edges']=int(graph.edges.shape[1])
            if phase:phase('backward',j,row)
            stamp=time.perf_counter();loss.backward();torch.cuda.synchronize(device);row['backward_seconds']=time.perf_counter()-stamp
            loss_sum+=row['half_CE'];rows.append(row)
        finally:ctx.close()
        del ctx,graph,logits,p,labels,loss
    live=[p.grad for p in model.parameters() if p.grad is not None]
    if not live or not all(torch.isfinite(x).all() for x in live):raise FloatingPointError('missing/nonfinite gradients')
    grad_norm=float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(x) for x in live])))
    if phase:phase('optimizer',None,{})
    stamp=time.perf_counter();opt.step();torch.cuda.synchronize(device)
    return dict(loss=loss_sum,gradient_norm=grad_norm,finite_gradients=True,inputs=rows,optimizer_seconds=time.perf_counter()-stamp,normal2_seconds=time.perf_counter()-start,
                supervised_slots=sum(x['slots'] for x in rows),CE_formula='sum over two inputs [0.5*20*sumCE/S]',schema_slots=case['space'].total,optimizer_steps=1)
