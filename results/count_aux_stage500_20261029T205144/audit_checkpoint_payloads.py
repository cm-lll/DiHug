"""CPU deserialization only: inspect final saved state, no model construction."""
import pathlib,json,gc
import torch
torch.set_num_threads(1)
ROOT=pathlib.Path(__file__).resolve().parent
def finite(tree):
    if torch.is_tensor(tree):return bool(torch.isfinite(tree).all())
    if isinstance(tree,dict):return all(finite(x) for x in tree.values())
    if isinstance(tree,(tuple,list)):return all(finite(x) for x in tree)
    return True
rows=[]
for folder in sorted((ROOT/'run').glob('*_s*_CE*')):
    if not folder.is_dir():continue
    path=folder/'checkpoint_u500.pt'
    if not path.exists():
        rows.append(dict(task=folder.name,passed=False,reason='missing final checkpoint'));continue
    p=torch.load(path,map_location='cpu')
    steps=[float(x['step']) for x in p['optimizer']['state'].values() if 'step' in x]
    row=dict(task=folder.name,updates=p['updates'],ema_updates=p['ema']['updates'],ema_decay=p['ema']['decay'],
        model_tensors=len(p['model']),optimizer_parameter_states=len(p['optimizer']['state']),
        optimizer_min_step=min(steps) if steps else None,optimizer_max_step=max(steps) if steps else None,
        rng_keys=sorted(p['rng']),rng_complete=set(p['rng'])=={'numpy','global_numpy','python','torch','device'},
        model_finite=finite(p['model']),optimizer_finite=finite(p['optimizer']),ema_finite=finite(p['ema']),
        lambda_count=p['binding']['lambda_count'],binding_target=p['binding']['updates'],
        source_sha256=p['binding']['source_sha256'],formula_sha256=p['binding']['formula_sha256'])
    row['passed']=p['updates']==p['ema']['updates']==500 and p['ema']['decay']==.99 and row['rng_complete'] and all(row[k] for k in ['model_finite','optimizer_finite','ema_finite']) and row['optimizer_max_step']==500
    rows.append(row);del p;gc.collect()
result=dict(passed=len(rows)==8 and all(r['passed'] for r in rows),rows=rows,method='CPU torch.load dictionary only; no model, forward, backward, optimizer.step, or CUDA call')
(ROOT/'analysis/checkpoint_payload_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(dict(passed=result['passed'],tasks=len(rows)),indent=2))
