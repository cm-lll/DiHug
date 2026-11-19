"""CPU analysis of already-computed split gradients; no model or optimizer calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,hashlib
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/feedback_gradient_split_20261113T110439'
def main():
    summaries=[];rows=[];rel=[];parameter_rows=[]
    for job in ['IMDB_2026110411_A_transport','PubMed_2026110422_A_transport']:
        if not (OUT/(job+'.json')).exists():continue
        obj=json.loads((OUT/(job+'.json')).read_text());pack=torch.load(OUT/(job+'_gradients.pt'),map_location='cpu',weights_only=False)
        data=np.load(OUT/(job+'_training_probabilities.npz'));family=data['family'];groups=obj['groups'];allg=groups[0]
        dot=lambda a,b:sum(float((pack[a][n]*pack[b][n]).sum()) for n in pack[a])
        nt=dot('total','total');contrib={k:dot(k,'total')/nt for k in ['label','prediction']} if nt else None
        bias=pack['total']['joint_head.2.bias'].numpy();expected=-obj['logit_summed']['total']['sum']
        bias_error=float(abs(bias[1]-expected));assert bias_error<1e-4
        stacked=np.concatenate([data[f'descent_{i}'] for i in range(32)])
        constants=[]
        for i in range(32):
            v=data[f'descent_{i}'];c=np.zeros_like(v)
            for r in np.unique(family):c[family==r]=v[family==r].mean()
            constants.append(c)
        common=np.concatenate(constants)
        common_energy=float(common@common/(stacked@stacked)) if stacked@stacked else None
        ds=obj['dataset'];parent=ROOT/'results/joint_transport_ablation_20261113T011928'/job/(ds+'_T_eager')/'initial_complete.pt'
        endpoint=ROOT/'results/robust_heat_update_check_20261113T095140'/(job+'_robust_temporary.pt')
        initial=torch.load(parent,map_location='cpu',weights_only=False);old=torch.load(endpoint,map_location='cpu',weights_only=False)
        gradflat=np.concatenate([pack['total'][n].numpy().ravel() for n in pack['total']])
        displacement=np.concatenate([(old['model'][n]-initial['model'][n]).numpy().ravel() for n in pack['total']]).astype(np.float64)
        cutoff=float(np.quantile(abs(gradflat),.9));top=abs(gradflat)>=cutoff
        optimizer=dict(initial_moments_empty=not initial['optimizer']['state'],top10_gradient_energy_fraction=float((gradflat[top]**2).sum()/(gradflat**2).sum()),
            top10_actual_displacement_energy_fraction=float((displacement[top]**2).sum()/(displacement**2).sum()),gradient_update_cosine=float(-gradflat@displacement/(np.linalg.norm(gradflat)*np.linalg.norm(displacement))),
            final_bias_actual_displacement=(old['model']['joint_head.2.bias']-initial['model']['joint_head.2.bias']).tolist())
        del initial,old,gradflat,displacement
        summaries.append(dict(job=job,parameter_groups=groups,logit=obj['logit_summed'],concatenated_logit_relation_constant_energy=common_energy,optimizer=optimizer,
            parameter_projection_contributions=contrib,joint_head_final_bias_gradient=bias.tolist(),joint_head_bias_logit_identity_error=bias_error,
            actual_gradient_recompute_relative_error=allg['recompute_relative_error']))
        for g in groups:rows.append(dict(job=job,**g))
        for r in obj['logit_summed_by_relation']:rel.append(dict(job=job,**r))
        for n,g in pack['total'].items():
            a=pack['label'][n];b=pack['prediction'][n]
            parameter_rows.append(dict(job=job,name=n,numel=g.numel(),total_norm=float(g.norm()),label_norm=float(a.norm()),prediction_norm=float(b.norm()),total_energy_fraction=float((g*g).sum())/nt if nt else None))
    for filename,rr in [('groups.csv',rows),('relations.csv',rel),('parameters.csv',parameter_rows)]:
        if rr:
            with (OUT/filename).open('w') as f:
                w=csv.DictWriter(f,list(rr[0]));w.writeheader();w.writerows(rr)
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2));print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
