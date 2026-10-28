"""Immutable experiment inputs and fail-closed authorization checks."""
import json,pathlib,hashlib,time,copy
import numpy as np
from prepare import load_case,sha,ROOT
from role_dictionary import seed32

def read_dictionary(directory,dataset,expected=None):
    root=pathlib.Path(directory)/('dictionary_'+dataset);receipt=json.loads((root/'selection.json').read_text())
    if expected is not None:
        for name in ['final_basis.npz','selection.json','panel.json','frozen_splits.json']:
            if sha(root/name)!=expected['files'][name]:raise ValueError('approval dictionary hash mismatch: '+name)
        if receipt.get('K')!=expected['K'] or receipt.get('offsets')!=expected['offsets'] or receipt.get('offsets',[-1])[-1]!=expected['dimension']:raise ValueError('approval dictionary dimensions')
    if receipt['status']!='passed':raise RuntimeError(dataset+' dictionary gate failed: '+receipt['status'])
    for typ,row in receipt['types'].items():
        if len(row['candidates'])!=row['Kmax'] or not all(c['valid'] for c in row['candidates']):raise ValueError('incomplete K comparison')
        if not row['final_fit']['converged']:raise ValueError('final fit not converged')
    if sha(root/'final_basis.npz')!=receipt['basis_sha256'] or sha(root/'panel.json')!=receipt['panel_sha256'] or sha(root/'frozen_splits.json')!=receipt['splits_sha256']:raise ValueError('dictionary binding')
    z=np.load(root/'final_basis.npz',allow_pickle=False)
    if z['offsets'].tolist()!=receipt['offsets'] or z['K'].tolist()!=receipt['K']:raise ValueError('dictionary dimensions')
    return dict(basis=z['basis'],offsets=z['offsets']),receipt

def scientific_gate(ledger_path,approval_path,protocol_path,dictionary_dir,source_hash):
    ledger=json.loads(pathlib.Path(ledger_path).read_text());approval=json.loads(pathlib.Path(approval_path).read_text())
    if time.time()>=ledger['deadline_epoch']:raise TimeoutError('original cumulative budget expired')
    # File is only written after explicit reviewed-user continuation, never by this package.
    if approval.get('independent_review')!='passed' or approval.get('user_continue_confirmed') is not True:raise PermissionError('independent review AND explicit user continuation required')
    if approval.get('ledger_sha256')!=sha(ledger_path) or approval.get('protocol_sha256')!=sha(protocol_path) or approval.get('source_sha256')!=source_hash:raise ValueError('approval/source/ledger binding mismatch')
    if approval.get('updates') not in (500,1000,2000):raise ValueError('outside frozen update tiers')
    for dataset in ['IMDB','PubMed']:read_dictionary(dictionary_dir,dataset,approval['dictionaries'][dataset])
    return ledger,approval

def source_binding():
    files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix in ('.py','.json','.npz')}
    return files,hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def initialization(dataset,arm,seed,basis):
    import torch
    from model_variants import StructuralDenoiser,copy_common
    from hsgd.bw_preflight.common import cpu_tree
    case,cfg=load_case(dataset);torch.manual_seed(seed);anchor=StructuralDenoiser(case,cfg,'A')
    # Separate module creation stream; all shared tensors are subsequently copied.
    module_seed=seed32(seed,dataset=dataset,arm=arm,purpose='new_module_initialization')
    torch.manual_seed(module_seed);model=StructuralDenoiser(case,cfg,arm,basis);copied=copy_common(anchor,model)
    hashes={k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
    proof={}
    for name in copied:
        if name.endswith('[:,:shared]'):
            a=anchor.joint_head[0].weight[:,:model.joint_head[0].in_features];b=model.joint_head[0].weight
        else:a=anchor.state_dict()[name];b=model.state_dict()[name]
        proof[name]=a.detach().cpu().contiguous().numpy().tobytes()==b.detach().cpu().contiguous().numpy().tobytes()
    if not all(proof.values()):raise AssertionError('common initialization bytes differ')
    return model,case,cfg,dict(seed=seed,module_seed=module_seed,copied_exact=proof,tensor_sha256=hashes,parameters=sum(p.numel() for p in model.parameters()))

def recommend_tier(cost,remaining,parallel_devices):
    """Component-cost proposal using exactly the launcher's fixed LPT queues."""
    if not cost.get('complete_eight_cells') or parallel_devices<1:raise ValueError('no complete engineering cost gate')
    values=cost['cells'];labels=[d+'_'+a for d in ['IMDB','PubMed'] for a in 'OABC']
    if set(values)!=set(labels):raise ValueError('missing cells')
    result=[]
    for updates in [2000,1000,500]:
        from admission import assignment
        queues,loads=assignment(cost,updates,list(range(min(parallel_devices,8))))
        total=max(loads)*cost['concurrency_margin']+1200+900
        result.append(dict(updates=updates,seconds=total,fits=total<=remaining,unmeasured_concurrency_margin=cost['concurrency_margin']))
    return next((r for r in result if r['fits']),None),result

def reset_training_streams(seed,dataset,device):
    import random,torch,pickle
    from probe import rng_get
    streams={name:seed32(seed,dataset=dataset,purpose='paired_training_'+name) for name in ['python','global_numpy','torch_cpu','torch_cuda','generator']}
    random.seed(streams['python']);np.random.seed(streams['global_numpy']);torch.random.default_generator.manual_seed(streams['torch_cpu'])
    if device=='cuda':torch.cuda.manual_seed(streams['torch_cuda'])
    rng=np.random.default_rng(streams['generator']);state=rng_get(rng,device)
    hashes={name:hashlib.sha256(value.cpu().contiguous().numpy().tobytes() if torch.is_tensor(value) else pickle.dumps(value,protocol=4)).hexdigest() for name,value in state.items()}
    return rng,dict(seeds=streams,state_sha256=hashes,after_model_construction_and_copy=True)
