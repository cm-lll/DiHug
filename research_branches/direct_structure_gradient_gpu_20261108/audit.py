"""Post-run read-only audit and tables; no model imports."""
import json,hashlib,csv,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'results/direct_structure_gradient_gpu_20261108T205133'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks={}
for path in [O/'source_binding.json',R/'results/direct_structure_cpu_20261108/source_binding.json',R/'results/joint_output_formal_projected_20261108T043000/source_binding.json']:
    values=read(path);checks[str(path)]=dict(files=len(values),changed=[p for p,h in values.items() if sha(p)!=h])
assets=read(O/'assets.json') if (O/'assets.json').exists() else []
for a in assets:
    checks[a['condition']]=dict(parent_unchanged=sha(a['parent'])==a['sha256'],CE_reference_unchanged=sha(a['reference'])==a['reference_sha256'])
rows=read(O/'results.json') if (O/'results.json').exists() else []
table=[]
for x in rows:
    for group,v in x['geometry'].items():table.append(dict(condition=x['condition'],seed=x['seed'],parent_updates=x['parent_updates'],training_t=x['training_t'],group=group,**v))
if table:
    with (O/'gradient_summary.csv').open('w') as f:w=csv.DictWriter(f,list(table[0]));w.writeheader();w.writerows(table)
for x in rows:assert sha(x['DIRECT_vs_MMD']['reference'])==x['DIRECT_vs_MMD']['sha256']
good=all(not v.get('changed',[]) and v.get('parent_unchanged',True) and v.get('CE_reference_unchanged',True) for v in checks.values())
audit=dict(at_epoch=time.time(),all_bindings_unchanged=good,checks=checks,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),local_tensors={str(p):sha(p) for p in O.glob('*.pt')},output_gradients={str(p):sha(p) for p in O.glob('*.npz')})
(O/'audit.json').write_text(json.dumps(audit,indent=2));assert good
print(json.dumps(dict(bindings_unchanged=good,conditions=len(rows),summary=[dict(condition=x['condition'],t=x['training_t'],geometry=x['geometry']['all'],DIRECT_vs_MMD=x['DIRECT_vs_MMD'],timing=x['timing']) for x in rows]),indent=2))
