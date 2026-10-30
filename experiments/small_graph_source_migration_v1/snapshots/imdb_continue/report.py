"""Complete frozen evaluation, original/LCC, all fields and missing future stages.

Adapters affect only jobs and result locations in this isolated CPU process.
No metric, sampling or main-run file is modified.
"""
import sys,shutil,csv,math,datetime,collections,traceback
from common import *
sys.path.insert(0,str(SOURCE))
import summarize as S
import cross_batch as C
ROOT=pathlib.Path(__file__).parent
STAGES=(3000,6000)
def local_jobs():return dict(training=[dict(dataset='IMDB',train_seed=s,arm=a,id=identity(s,a),slot=i,model='E',fusion='early') for i,(s,a) in enumerate(CELLS)],sampling=jobs(STAGES))
def configure_evaluation():
 S.jobs=local_jobs;S.generation_checkpoints=lambda d:list(STAGES)
 C.jobs=local_jobs;C.generation_checkpoints=lambda d:list(STAGES)
 C.PROTOCOL=dict(train_seeds={'IMDB':[2026102911,2026102931]},arms=['CE','CE_COUNT'],sample_seeds=[2026102821,2026102822])
 # S.main uses a local import of the old immutable protocol only to enumerate brief output keys.
 # Metric formulas and aggregate() are unchanged; the new complete CSVs cover every STAGE.
def import_finished():
 rows=[]
 for j in jobs((3000,)):
  src=PARENT/j['directory'];dst=OUT/'run'/j['directory']
  if not (src/'summary.json').exists():rows.append(dict(job=j,status='pending',reason='no completed summary'));continue
  try:
   ss=read(src/'summary.json')
   if ss['status']!='complete':rows.append(dict(job=j,status='pending',reason='main chain not complete'));continue
   b=read(src/'binding.json');g=src/'graph.npz'
   if b['job']!=j or b['source_sha256']!=SOURCE_HASH or sha(g)!=b['graph_sha256']:raise ValueError('main graph identity/source/hash')
   # Main status can change only from pending to complete; never read/copy a partial graph.
   dst.mkdir(parents=True,exist_ok=False)
   for name in ('graph.npz','binding.json','summary.json','metrics.json'):
    shutil.copy2(src/name,dst/name)
   rows.append(dict(job=j,status='imported_readonly',source=str(src),graph_sha256=b['graph_sha256']))
  except BaseException as exc:rows.append(dict(job=j,status='failed_import',reason=repr(exc)))
 save(OUT/'external_imports.json',rows)
def leaves(value,path=''):
 if isinstance(value,dict):
  for k,v in value.items():yield from leaves(v,path+'.'+str(k) if path else str(k))
 elif isinstance(value,list):
  for i,v in enumerate(value):yield from leaves(v,path+f'[{i}]')
 else:yield path,value
def csvwrite(path,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with pathlib.Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def definition(k,group):
 if group=='absolute_error':return 'abs(generated metric - fixed reference); computed separately for each graph','same as raw','lower'
 if group=='distribution_error':
  if 'W1' in k:return 'Wasserstein-1 of discrete degree histograms; incident4 uses fixed reference type weights, all10=0.4*incident4','degree','lower'
  if 'TV' in k:return 'total variation:one half L1 of normalized pattern/degree-vector frequencies; frozen empty-pattern convention','0..1','lower'
  if 'MMD' in k:return 'biased Gaussian-kernel MMD squared, alpha4, frozen implementation; raw/nonnegative explicitly separated','kernel distance','lower'
 if k.startswith('closure_'):return 'typed two-hop endpoints/middle pattern; path counts count each distinct middle; unique_pair counts endpoints once; ratio=numerator/denominator, null at zero denominator','ratio' if k.endswith('ratio') else 'instances','descriptive; closeness to reference'
 if k.startswith('family'):
  return 'schema family index; edges=count, capacity=n_a*n_b or n_a*(n_a-1)/2, density=edges/capacity using actual retained nodes','fraction' if k.endswith('density') else 'count','closeness to reference'
 if k.startswith('triangle_'):return 'unordered simple triangle by sorted three coarse node types; zero reference is legal','triangles','closeness to reference'
 if k.startswith('MP'):return 'undirected simple path of stated edge length; all nodes distinct; reversal canonicalized, each path counted once; per_node divides actual nodes','paths/node' if k.endswith('per_node') else 'paths','closeness to reference'
 if k.startswith('walk'):return 'typed walk of stated edge length; repeated vertices/backtracking allowed; direction instances retained although reversed type patterns share key','walk instances','closeness to reference'
 if k.startswith('type_'):
  return 'named coarse-type subset; relationN uses schema column N; degree includes all incident edges. maximum/second=ordered degrees; p90/p99=NumPy quantile; zero/low_le1 counts; fractions divide actual type count; share divides graph nodes; fixed_ref_P99 uses ceil reference p99 threshold; Gini and top5 preserve frozen definitions','fraction' if any(s in k for s in ('fraction','share','gini')) else 'degree or count as field suffix','closeness to reference; larger not inherently better'
 definitions={
 'nodes':'number of retained nodes','edges':'number of unique undirected edges','components':'number of connected components over all relations','lcc':'largest connected component node count','second_component':'second largest component size,0 if absent','isolates':'zero-total-degree node count',
 'triangles':'unordered simple triangle count,all types','four_cycles':'true undirected simple4-cycle count,rotations/reversal deduplicated','HGEN_square_legacy_K4':'legacy HGEN square actually counts4-cliques; distinct from true4-cycles',
 'average_clustering':'NetworkX mean local clustering,including isolates as0','transitivity':'3*triangles/wedges, null for0 wedges','HGEN_clustering_legacy_3T_over_claws':'legacy3*triangles/claws, null at0 claws; not standard clustering',
 'wedge_count':'sum d*(d-1)/2','claw_count':'sum d*(d-1)*(d-2)/6',
 'cpl_finite_ordered_pairs':'mean finite shortest path distance,ordered distinct reachable pairs only','cpl_pair_count':'number of ordered reachable distinct pairs','reachable_pair_fraction':'reachable ordered pairs / (n*(n-1))','cpl_lcc':'mean shortest path distance among ordered distinct LCC pairs','diameter_global':'finite only when entire graph connected; otherwise null','diameter_lcc':'maximum distance within LCC','diameter_max_component_finite':'maximum finite distance across all components',
 'HGEN_native_powerlaw':'unavailable native dependency; null,no installation or substitute','powerlaw_fixed_xmin1_alpha':'project estimate fixed xmin1; not native HGEN powerlaw','powerlaw_fixed_xmin1_KS':'project fixed xmin1 fit KS distance','assortativity':'untyped degree Pearson correlation on edge endpoints; null when variance0',
 'HGEN_gini_legacy_zero_based':'frozen HGEN zero-based-index Gini; distinct from standard Gini','HGEN_entropy_legacy_eps1e_4':'frozen HGEN degree entropy including epsilon1e-4','gini_standard':'standard sorted-degree Gini; null at zero total degree',
 'edge_overlap_count':'reference-aligned node-ID edge intersection size','EO_reference_denominator':'edge intersection / reference edges','EO_generated_denominator':'edge intersection / generated edges','EO_jaccard':'edge intersection / union','edge_novelty_generated_denominator':'generated-only edges / generated edges',
 'maximum_degree':'maximum total node degree','minimum_degree':'minimum total node degree','mean_degree':'2*edges/nodes'}
 return definitions.get(k,'frozen metric field; detailed expression in bound evaluation_fields.py and lcc_eval/metrics.py'),'see field/count/degree/ratio','closeness to reference; descriptive raw value'
def exports(result):
 tables=OUT/'tables';tables.mkdir(exist_ok=True);scalars=[];nested=[];catalog={};reference=result['records'][0]['reference'] if result['records'] else {}
 for r in result['records']:
  key={k:r[k] for k in ('dataset','train_seed','arm','checkpoint','sample_seed','version')}
  for group in ('raw','reference','absolute_error','distribution_error'):
   for metric,v in r[group].items():
    scalars.append(dict(**key,group=group,metric=metric,value=v,reason=r.get('undefined',{}).get(group,{}).get(metric)))
    if group!='reference':
     de,unit,direction=definition(metric,group);catalog[group,metric]=dict(group=group,field=metric,definition=de,unit=unit,reference=reference.get(metric) if group!='distribution_error' else 0,direction=direction,source=str(S.SOURCE if hasattr(S,'SOURCE') else SOURCE/'evaluation_fields.py'))
  for group in ('statistics','comparison'):
   for path,v in leaves(r[group]):nested.append(dict(**key,group=group,field=path,value=v,reason='undefined_or_unavailable' if v is None else None))
 csvwrite(tables/'per_graph_all_scalar_metrics.csv',scalars);csvwrite(tables/'per_graph_all_detail_fields.csv',nested)
 csvwrite(tables/'metric_catalog.csv',list(catalog.values()))
 # Every scalar field has an explicit reference,error and direction; all variable pattern/histogram arrays remain in JSON+long CSV.
 for name in ('cell_means','paired','paired_means','length_paired','length_means','lcc_minus_original'):
  csvwrite(tables/(name+'.csv'),result.get(name,[]))
 save(tables/'sample_diversity.json',result['diversity']);save(tables/'missing_identities.json',result['missing_identities'])
 for name in ('METRICS.md','metric_definition_reference.md'):
  shutil.copy2(SOURCE/name,OUT/name)
 for name in ('evaluation_fields.py','cross_batch.py','summarize.py','lcc_eval/metrics.py'):
  dest=OUT/'frozen_evaluation_source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/name,dest)
 shutil.copytree(SOURCE/'lcc_eval/vendor',OUT/'frozen_evaluation_source/lcc_eval/vendor',dirs_exist_ok=True)
def plots(result):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 metrics=[('absolute_error','edges'),('distribution_error','main_incident4_W1'),('distribution_error','main_joint_degree_TV'),('absolute_error','triangles'),('distribution_error','triangle_composition_TV'),('raw','type_Genre_low_le1_fraction'),('raw','type_Genre_maximum'),('raw','lcc'),('distribution_error','MP2_composition_TV'),('distribution_error','MP3_composition_TV')]
 for version in ('original','lcc'):
  fig,axs=plt.subplots(5,2,figsize=(12,18));index={(r['train_seed'],r['arm'],r['checkpoint'],r['group'],r['metric']):r['value'] for r in result['cell_means'] if r['version']==version}
  for ax,(group,metric) in zip(axs.flat,metrics):
   for s,style in [(2026102911,'-'),(2026102931,'--')]:
    for arm,color in [('CE','tab:blue'),('CE_COUNT','tab:orange')]:
     values=[index.get((s,arm,u,group,metric)) for u in STAGES]
     ax.plot(STAGES,[float('nan') if v is None else v for v in values],style,color=color,marker='o',label=f'{s} {arm}')
   ax.set_title(metric);ax.set_xlabel('normal2 updates');ax.set_xticks(STAGES);ax.grid(alpha=.2)
  axs.flat[0].legend(fontsize=7);fig.suptitle(f'IMDB actual generated graphs ({version}); 2 samples/seed; missing6000 stays blank')
  fig.tight_layout();fig.savefig(OUT/f'trends_{version}.png',dpi=140);fig.savefig(OUT/f'trends_{version}.pdf');plt.close(fig)
def main():
 check(False);configure_evaluation();import_finished()
 sys.argv=['summarize','--run',str(OUT/'run'),'--output',str(OUT/'evaluation.json'),'--deadline',str(DEADLINE-70)]
 S.main() # an incomplete future6000 panel is expected, retained as null
 result=read(OUT/'evaluation.json');exports(result)
 try:plots(result)
 except BaseException as exc:save(OUT/'plot_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
 save(OUT/'report_completion.json',dict(epoch=time.time(),new_complete_graphs=sum(x['status']=='complete' for x in result['inventory'] if x['job']['checkpoint']==6000),
  graph_counts={u:sum(x['status']=='complete' and x['job']['checkpoint']==u for x in result['inventory']) for u in STAGES},errors=result['errors'],adapter=own_binding()))
if __name__=='__main__':main()

