"""CPU probability diagnostics. No sampled graphs and no surrogate reward claims."""
import json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/probability_shift_check_20261113T104256'
TIMES=[1,75,250,550,800,1000]
def cosine(a,b):
 den=float(np.linalg.norm(a)*np.linalg.norm(b))
 return float(a@b/den) if den else None
def change(a,b,f,label):
 d=b-a;ad=abs(d);l1=float(ad.sum());balanced=0.
 for r in np.unique(f):
  v=d[f==r];balanced+=float(abs(v).sum()-abs(v.sum()))
 # This is the probability under an independent shared-uniform coupling,
 # not total variation between complete graph distributions.
 anydiff=1. if (ad>=1).any() else float(-np.expm1(np.log1p(-ad).sum()))
 return dict(slots=len(d),sum_abs_probability_change=l1,mean_abs_probability_change=float(ad.mean()),max_abs_probability_change=float(ad.max()),p95_abs=float(np.quantile(ad,.95)),p99_abs=float(np.quantile(ad,.99)),net_expected_edge_change=float(d.sum()),increase_mass=float(d[d>0].sum()),decrease_mass=float(-d[d<0].sum()),balanced_within_relation_fraction=balanced/l1 if l1 else None,slots_change_above_1e_6=int((ad>1e-6).sum()),slots_change_above_1e_3=int((ad>1e-3).sum()),slots_change_above_1e_2=int((ad>1e-2).sum()),shared_uniform_probability_any_difference=anydiff,positive_missed_mass_change=float(-d[label==1].sum()),negative_false_mass_change=float(d[label==0].sum()))
def scalar_fit(full,partial):
 den=float(full@full);ss=float(partial@partial)
 if not den or not ss:return dict(fit_scale=None,residual_fraction=None,cosine=cosine(full,partial),same_direction_partial_L1_fraction=None)
 gamma=float((full@partial)/den);res=float(np.linalg.norm(partial-gamma*full)/np.sqrt(ss))
 return dict(fit_scale=gamma,residual_fraction=res,cosine=cosine(full,partial),one_scalar_explained_energy_fraction=1-res**2,same_direction_partial_L1_fraction=float(abs(partial)[full*partial>0].sum()/abs(partial).sum()))
def average(rows,key):
 a=[r[key] for r in rows if r.get(key) is not None];return float(np.mean(a)) if a else None
totals=[];relations=[];fits=[];nodes=[];summaries=[]
for file in sorted(OUT.glob('*_probabilities.npz')):
 job=file.name.removesuffix('_probabilities.npz');meta=json.loads((OUT/(job+'.json')).read_text());data=np.load(file)
 jobrows=[];jobfits=[]
 for j,t in enumerate(TIMES):
  f=data[f'family_{j}'];lab=data[f'label_{j}'];q=data[f'query_{j}'];types=data[f'types_{j}'];prior=data[f'prior_{j}'];active=(prior[f]>0)&(prior[f]<1)
  common=dict(dataset=meta['dataset'],seed=meta['seed'],t=t,active_slots=int(active.sum()))
  ff=data[f'full_p_{j}']-data[f'baseline_p_{j}'];pp=data[f'partial_p_{j}']-data[f'baseline_p_{j}']
  fit=dict(**common,**scalar_fit(ff,pp));fits.append(fit);jobfits.append(fit)
  for a,b in [('baseline','full'),('baseline','partial'),('full','partial')]:
   contrast=a+'_to_'+b
   for space in ['p','q']:
    aa=data[f'{a}_{space}_{j}'];bb=data[f'{b}_{space}_{j}'];row=dict(**common,contrast=contrast,space='clean' if space=='p' else 'reverse',**change(aa,bb,f,lab));totals.append(row);jobrows.append(row)
    if space=='p':
     dl=data[f'{b}_logit_{j}']-data[f'{a}_logit_{j}'];res=dl.copy()
     for r in np.unique(f[active]):res[f==r]-=dl[f==r].mean()
     den=float(dl[active]@dl[active]);row['family_constant_logit_explained_energy']=float(1-res[active]@res[active]/den) if den else None
     for r in np.unique(f):
      ix=f==r;relations.append(dict(**common,contrast=contrast,family=int(r),positive_n=int((lab[ix]==1).sum()),negative_n=int((lab[ix]==0).sum()),**change(aa[ix],bb[ix],f[ix],lab[ix])))
     ref=np.zeros(len(types));da=np.zeros(len(types));db=np.zeros(len(types))
     for endpoint in q:
      np.add.at(ref,endpoint,lab);np.add.at(da,endpoint,aa);np.add.at(db,endpoint,bb)
     for typ in np.unique(types):
      ix=types==typ
      for group,sel in [('all',ix),('low',ix&(ref<=1)),('hub',ix&(ref>=np.quantile(ref[ix],.9)))]:
       nodes.append(dict(**common,contrast=contrast,type=int(typ),group=group,n=int(sel.sum()),expected_degree_shift=float((db-da)[sel].mean()) if sel.any() else None,degree_AE_change=float((abs(db-ref)-abs(da-ref))[sel].mean()) if sel.any() else None))
  # Store observed-slot labels and query identities in NPZ; never sample them.
 summary=dict(dataset=meta['dataset'],seed=meta['seed'],recomputed_previous_MAE_max_abs=max(abs(r['recompute_difference']) for r in meta['rows']),contrasts={},mean_fit={k:average(jobfits,k) for k in ['fit_scale','residual_fraction','cosine','one_scalar_explained_energy_fraction','same_direction_partial_L1_fraction']})
 for contrast in ['baseline_to_full','baseline_to_partial','full_to_partial']:
  summary['contrasts'][contrast]={}
  for space in ['clean','reverse']:
   rr=[r for r in jobrows if r['contrast']==contrast and r['space']==space]
   summary['contrasts'][contrast][space]={k:average(rr,k) for k in ['sum_abs_probability_change','mean_abs_probability_change','max_abs_probability_change','net_expected_edge_change','balanced_within_relation_fraction','slots_change_above_1e_3','positive_missed_mass_change','negative_false_mass_change','family_constant_logit_explained_energy']}
 summary['pooled_fit']=scalar_fit(np.concatenate([data[f'full_p_{j}']-data[f'baseline_p_{j}'] for j in range(len(TIMES))]),np.concatenate([data[f'partial_p_{j}']-data[f'baseline_p_{j}'] for j in range(len(TIMES))]))
 for contrast in summary['contrasts']:
  for space in ['clean','reverse']:
   rr=[r for r in jobrows if r['contrast']==contrast and r['space']==space]
   total=sum(r['sum_abs_probability_change'] for r in rr)
   summary['contrasts'][contrast][space]['mass_weighted_balanced_within_relation_fraction']=sum(r['sum_abs_probability_change']*(r['balanced_within_relation_fraction'] or 0.) for r in rr)/total if total else None
 summaries.append(summary)
for name,rows in [('per_time_probability_change.csv',totals),('per_relation_probability_change.csv',relations),('full_vs_partial_fit.csv',fits),('per_type_node_changes.csv',nodes)]:
 if rows:
  fields=list(dict.fromkeys(k for r in rows for k in r))
  with (OUT/name).open('w') as h:
   writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader();writer.writerows(rows)
(OUT/'summary.json').write_text(json.dumps(summaries,indent=2));print(json.dumps(summaries,indent=2))
