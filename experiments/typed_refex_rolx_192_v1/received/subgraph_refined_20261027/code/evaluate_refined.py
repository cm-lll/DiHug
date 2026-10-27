"""Exact before/after full evaluator, CPU only; no optimization."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLCONFIGDIR']='/tmp/refine-mpl'
import json,numpy as np,csv,shutil
from pathlib import Path
import legacy_evaluator as ev
ROOT=Path(__file__).resolve().parent.parent;SRC=Path('/workspace/shared/subgraph_preview_20261027');OUT=ROOT/'evaluation';OUT.mkdir(exist_ok=True)
import refine
NAMES={'IMDB':['Director','Actor','Movie','Genre'],'PubMed':['Gene','Disease','Chemical','Species']}
METRICS={};NORMAL={};rows=[];GRAPHS={};ARRAYS={}
for ds,names in NAMES.items():
 g=ev.load(ROOT/'inputs/reference'/f'{ds}.npz');m,a=ev.calc(g,names);obj=refine.Objective(g,refine.fast_metrics(g));ARR={};rec={'original':m,'stages':{}}
 np.savez_compressed(OUT/f'{ds}_original_pernode.npz',original_ids=np.arange(len(g['types'])),types=g['types'],**a)
 for stage,filename in [('before','baseline_graph.npz'),('after','graph.npz')]:
  sg=ev.load(ROOT/'samples'/ds/filename);ids=sg['original_ids'];sm,sa=ev.calc(sg,names);score,terms=obj.evaluate(sg['types'],refine.fast_metrics(sg));ret=np.divide(sa['total_degree'],a['total_degree'][ids],out=np.zeros(len(ids),float),where=a['total_degree'][ids]>0)
  inv=np.full(len(g['types']),-1,int);inv[ids]=np.arange(len(ids));mask=(inv[g['edges']]>=0).all(1)
  audit={'exact_induced':bool(np.array_equal(inv[g['edges'][mask]],sg['edges']) and np.array_equal(g['relations'][mask],sg['relations'])),'unique_original_ids':len(set(ids))==192,'nodes_192':len(ids)==192,'no_isolates':sm['isolates']==0,'types_and_schema':bool(np.array_equal(g['types'][ids],sg['types']) and np.array_equal(g['schema'],sg['schema']))}
  top={}
  for typ,name in enumerate(names):
   members=np.flatnonzero(g['types']==typ);rank=sorted(members,key=lambda u:(-a['total_degree'][u],int(u)));h=set(rank[:max(1,int(np.ceil(len(rank)*.1)))]);hit=np.array([i for i,u in enumerate(ids) if u in h],dtype=int)
   top[name]=dict(original_top10_count=len(h),selected_count=len(hit),selected_original_ids=ids[hit].tolist(),mean_retained_degree_fraction=float(ret[hit].mean()) if len(hit) else None)
  r=dict(final_induced=sm,objective_score=score,objective_terms=terms,audit=audit,selected_nodes_in_original=ev.subset_scope(g['types'],a,ids,names),hub_retention=top,degree_sum_retention=float(sa['total_degree'].sum()/a['total_degree'][ids].sum()),retained_degree_fraction=dict(mean=float(ret.mean()),median=float(np.median(ret)),min=float(ret.min()),max=float(ret.max())),mp_composition_TV_vs_original=[ev.tv(x['proportions'],y['proportions']) for x,y in zip(sm['mp'],m['mp'])],mp_type_coverage=[dict(observed=len(sm['mp'][k]['counts']),reference=len(m['mp'][k]['counts']),missing=sorted(set(m['mp'][k]['counts'])-set(sm['mp'][k]['counts']))) for k in range(3)])
  rec['stages'][stage]=r;ARR[stage]=sa;GRAPHS[(ds,stage)]=sg
  np.savez_compressed(OUT/f'{ds}_{stage}_pernode.npz',original_ids=ids,types=sg['types'],original_degree=a['degree'][ids],induced_degree=sa['degree'],original_triangles=a['triangle'][ids],induced_triangles=sa['triangle'],original_clustering=a['clustering'][ids],induced_clustering=sa['clustering'],retained_degree_fraction=ret)
  rows.append(dict(dataset=ds,stage=stage,nodes=sm['nodes'],edges=sm['edges'],triangles=sm['triangles'],clustering=sm['average_clustering'],lcc=sm['lcc'],components=len(sm['component_sizes']),isolates=sm['isolates'],mean_degree=sm['degree']['mean'],degree_max=sm['degree']['max'],objective=score,**terms,MP2_total=sm['mp'][1]['total'],MP3_total=sm['mp'][2]['total'],MP3_TV=r['mp_composition_TV_vs_original'][2],MP2_types=len(sm['mp'][1]['counts']),MP3_types=len(sm['mp'][2]['counts']),degree_sum_retention=r['degree_sum_retention']))
 METRICS[ds]=rec;NORMAL[ds]=rec['stages']['after']['final_induced'];ARRAYS[ds]={'original':a,**ARR};print(ds,json.dumps({k:{'objective':v['objective_score'],'mpTV':v['mp_composition_TV_vs_original'],'mp_coverage':v['mp_type_coverage'],'edges':v['final_induced']['edges'],'triangles':v['final_induced']['triangles']} for k,v in rec['stages'].items()}),flush=True)
ev.dump(OUT/'metrics.json',dict(metric_source='code/legacy_evaluator.py + evaluation/audited_paths_excerpt.py',mp_convention='Exact simple undirected paths; reversal identified. Families zero-based in metrics, one-based in NPZ.',datasets=METRICS));ev.dump(OUT/'final_induced_metrics.json',NORMAL)
with (OUT/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Raw-degree empirical CDFs, separate selected-node original scope, and fixed raw axes.
colors={'original':'#222222','before':'#E69F00','after':'#0072B2'}
for ds,names in NAMES.items():
 fig,axes=ev.plt.subplots(1,3,figsize=(15,4.4))
 for stage in ['original','before','after']:
  vals=ARRAYS[ds][stage]['total_degree'];z=np.sort(vals);axes[0].step(z,np.arange(1,len(z)+1)/len(z),where='post',label=stage,color=colors[stage])
 for stage in ['before','after']:
  z=np.sort(ARRAYS[ds]['original']['total_degree'][GRAPHS[(ds,stage)]['original_ids']]);axes[0].step(z,np.arange(1,len(z)+1)/len(z),where='post',label=f'{stage} nodes in original',ls=':',alpha=.7,color=colors[stage])
 axes[0].set(xscale='symlog',xlabel='Actual degree (symlog)',ylabel='Empirical CDF',title='Raw total degree; no rescaling');axes[0].legend(fontsize=7)
 x=np.arange(10);width=.25
 for j,stage in enumerate(['original','before','after']):
  mm=METRICS[ds]['original'] if stage=='original' else METRICS[ds]['stages'][stage]['final_induced'];axes[1].bar(x+(j-1)*width,mm['family_proportions'],width=width,label=stage,color=colors[stage])
 axes[1].set(xticks=x,xticklabels=[f'{a}-{b}' for a,b in GRAPHS[(ds,'after')]['schema']],xlabel='Relation family (type IDs)',ylabel='Edge proportion',title='Relation composition');axes[1].tick_params(axis='x',rotation=65)
 keys=['relation_TV','typed_logdegree_CDF','clustering_abs_error','typed_triangle_TV','MP2_TV'];labels=['Relation TV','Typed degree\nCDF-L1','Clustering\nabs. error','Triangle TV','MP2 TV'];x=np.arange(5)
 for j,stage in enumerate(['before','after']):axes[2].bar(x+(j-.5)*.35,[METRICS[ds]['stages'][stage]['objective_terms'][k] for k in keys],width=.35,label=stage,color=colors[stage])
 axes[2].set(xticks=x,xticklabels=labels,ylabel='Distance (lower is better)',title='Frozen objective components');axes[2].tick_params(axis='x',labelsize=8)
 for ax in axes:ax.grid(alpha=.15,axis='y')
 fig.suptitle(f'{ds}: before / refined / full reference');fig.tight_layout();fig.savefig(OUT/f'{ds}_distribution_comparison.png',dpi=170);ev.plt.close(fig)
 fig,axes=ev.plt.subplots(1,4,figsize=(15,3.8))
 for typ,name in enumerate(names):
  ax=axes[typ]
  for stage in ['original','before','after']:
   t=ev.load(ROOT/'inputs/reference'/f'{ds}.npz')['types'] if stage=='original' else GRAPHS[(ds,stage)]['types'];v=np.sort(ARRAYS[ds][stage]['total_degree'][t==typ]);ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=stage,color=colors[stage])
  ax.set(xscale='symlog',title=name,xlabel='Actual total degree',ylabel='CDF');ax.grid(alpha=.15)
 axes[0].legend(fontsize=8);fig.suptitle(f'{ds}: raw degree distributions by node type (no rescaling)');fig.tight_layout();fig.savefig(OUT/f'{ds}_typed_degree_CDF.png',dpi=170);ev.plt.close(fig)
fig,axes=ev.plt.subplots(2,2,figsize=(12,11));palette=['#0072B2','#E69F00','#009E73','#CC79A7']
for ax,(ds,stage) in zip(axes.flat,[(d,s) for d in NAMES for s in ['before','after']]):
 sg=GRAPHS[(ds,stage)];xy=ev.packed_layout(sg,7301);ax.add_collection(ev.LineCollection(xy[sg['edges']],colors='#808080',linewidths=.4,alpha=.28))
 for typ,name in enumerate(NAMES[ds]):sel=sg['types']==typ;ax.scatter(*xy[sel].T,s=20,c=palette[typ],label=f'{name} ({sum(sel)})',edgecolors='white',linewidths=.2)
 mm=METRICS[ds]['stages'][stage]['final_induced'];ax.autoscale();ax.axis('off');ax.set_title(f'{ds} {stage}: N=192, E={mm["edges"]}, triangles={mm["triangles"]}\nLCC={mm["lcc"]}; mean clustering={mm["average_clustering"]:.4f}');ax.legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.01));ax.set_aspect('equal')
fig.suptitle('Exact induced subgraphs; layout seed 7301, color=node type, positions not comparable');fig.tight_layout();fig.savefig(OUT/'before_after_graphs.png',dpi=170);ev.plt.close(fig)
