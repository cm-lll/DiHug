"""Read saved CPU tensors only to summarize actual parameter displacement."""
import json,math
from pathlib import Path
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/reproducible_aggregation_20261108T113027'
tasks=json.loads((ROOT/'results/output_boundary_replay_20261108T081654/tasks.json').read_text());rows=[];cross=[]
def group(n):
    if n.startswith('common_neighbor.'):return 'CN_exclusive'
    if n.startswith('joint_head.'):return 'joint_head'
    if n.startswith('branch.'):return 'degree_branch'
    return 'shared_backbone_other'
for task in tasks:
    ds=task['dataset'];parent=torch.load(task['checkpoint'],map_location='cpu')['model']
    for kind in ['CE','PS']:
        ordinary=None
        for mode in ['ordinary','strict']:
            path=OUT/'attempt3'/f'{ds}_{kind}_{mode}_first.pt'
            if not path.exists():continue
            z=torch.load(path,map_location='cpu');names=list(z['gradient']);delta={n:z['state']['model'][n]-parent[n] for n in names}
            for grp in ['all','shared_backbone_other','CN_exclusive','joint_head','degree_branch']:
                ns=[n for n in names if grp=='all' or group(n)==grp]
                d2=sum(float(delta[n].double().square().sum()) for n in ns);g2=sum(float(z['gradient'][n].double().square().sum()) for n in ns if z['gradient'][n] is not None)
                dot=sum(float((delta[n].double()*z['gradient'][n].double()).sum()) for n in ns if z['gradient'][n] is not None)
                rows.append(dict(dataset=ds,kind=kind,mode=mode,group=grp,parameters=sum(delta[n].numel() for n in ns),delta_L2=math.sqrt(d2),gradient_L2=math.sqrt(g2),cos_negative_gradient=-dot/math.sqrt(d2*g2) if d2*g2 else None))
            if mode=='ordinary':ordinary=delta
            elif ordinary is not None:
                d2=sum(float((delta[n]-ordinary[n]).double().square().sum()) for n in names)
                a2=sum(float(delta[n].double().square().sum()) for n in names);b2=sum(float(ordinary[n].double().square().sum()) for n in names)
                dot=sum(float((delta[n].double()*ordinary[n].double()).sum()) for n in names)
                cross.append(dict(dataset=ds,kind=kind,cross_mode_delta_difference_L2=math.sqrt(d2),cosine=dot/math.sqrt(a2*b2) if a2*b2 else None))
            del z
    del parent
assert not torch.cuda.is_initialized()
(OUT/'parameter_displacement.json').write_text(json.dumps(dict(rows=rows,cross=cross,CUDA_initialized=False),indent=2)+'\n')
print(json.dumps(cross,indent=2))
