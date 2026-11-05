"""CPU-only inspection of saved repeatability tensors, without model calls."""
import argparse, json, math
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    torch.set_num_threads(1);d=json.loads((a.directory/'result.json').read_text());out=[]
    for arm in ('A','candidate'):
        base=torch.load(a.directory/(arm+'_roundtrip.pt'),map_location='cpu')
        for left,right in [('M1','M2'),('F1','F2')]:
            row=next(r for r in d['comparisons'] if r['arm']==arm and r['left']==left and r['right']==right)
            x=torch.load(a.directory/(arm+'_'+left+'.pt'),map_location='cpu')
            y=torch.load(a.directory/(arm+'_'+right+'.pt'),map_location='cpu')
            dot=aa=bb=dd=0.
            for n,v in x['delta'].items():
                u=y['delta'][n].double();v=v.double()
                assert torch.isfinite(u).all() and torch.isfinite(v).all()
                dot+=float((u*v).sum());aa+=float(v.square().sum());bb+=float(u.square().sum());dd+=float((v-u).square().sum())
            w=row['worst_coordinate'];name=w['name'];index=w['flat_index']
            # Parameter insertion order is the same unique named_parameters order
            # used to construct the optimizer; optimizer IDs are separately checked.
            names=list(x['parameters']);group=base['optimizer']['param_groups'][0]
            assert len(names)==len(group['params'])
            old=base['optimizer']['state'][group['params'][names.index(name)]]
            coord=dict(name=name,flat_index=index,old_m=float(old['exp_avg'].flatten()[index]),old_v=float(old['exp_avg_sq'].flatten()[index]),runs={})
            b1,b2=group['betas'];lr=group['lr'];eps=group['eps'];wd=group['weight_decay']
            for label,record in [(left,x),(right,y)]:
                s=record['moments'][name];step=float(s['step']);m=float(s['exp_avg'].flatten()[index]);v=float(s['exp_avg_sq'].flatten()[index]);before=float(base['model'][name].flatten()[index]);actual=float(record['delta'][name].flatten()[index])
                denominator=math.sqrt(v/(1-b2**step))+eps
                formula=-lr*(m/(1-b1**step))/denominator-lr*wd*before
                coord['runs'][label]=dict(gradient=float(record['gradients'][name].flatten()[index]),bias_corrected_denominator=denominator,
                    adam_formula_float64=formula,actual_fp32_delta=actual,rounding_difference=actual-formula)
            out.append(dict(arm=arm,left=left,right=right,delta_difference_l2=math.sqrt(dd),
                delta_cosine=dot/math.sqrt(aa*bb),worst_coordinate=coord,optimizer_group=group))
    (a.directory/'cpu_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
