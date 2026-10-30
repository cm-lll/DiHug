"""Descriptive comparison, not paired GAN/diffusion training trajectories."""
import os,pathlib,csv,collections
ROOT=pathlib.Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/matplotlib');os.environ['OMP_NUM_THREADS']='1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=list(csv.DictReader((ROOT/'tables/cross_method_training_seed_means.csv').open()))
metrics=[('absolute_error','edges','Absolute edge count error'),('distribution_error','main_incident4_W1','Incident4 degree W1'),('distribution_error','main_joint_degree_TV','Joint degree TV'),('absolute_error','triangles','Absolute triangle error'),('distribution_error','triangle_composition_TV','Triangle composition TV'),('distribution_error','MP2_composition_TV','MP2 composition TV'),('raw','lcc','LCC nodes (descriptive)'),('raw','isolates','Isolates (descriptive)')]
order=['E-earlyCN_CE@3000','E-earlyCN_CE_COUNT@3000','E-earlyCN_CE@6000','E-earlyCN_CE_COUNT@6000','HGEN@D3750_G750']
labels=['CE 3k','COUNT 3k','CE 6k','COUNT 6k','HGEN']
colors=['#7C93BA','#64B5A5','#486DA1','#218C74','#D07849']
for d in ('IMDB','PubMed'):
 fig,axes=plt.subplots(2,4,figsize=(16,8));fig.suptitle(d+' | full-reference baseline; original graphs | each dot = one training seed, mean of two graphs')
 for ax,(group,metric,title) in zip(axes.ravel(),metrics):
  for i,key in enumerate(order):
   rr=[r for r in rows if r['dataset']==d and r['version']=='original' and r['group']==group and r['metric']==metric and r['method']+'@'+r['checkpoint']==key]
   vv=[float(r['value']) for r in rr if r.get('value') not in ('',None)]
   if vv:
    offsets=[(j-(len(vv)-1)/2)*.07 for j in range(len(vv))]
    ax.scatter([i+v for v in offsets],vv,color=colors[i],s=36,alpha=.85)
    avg=sum(vv)/len(vv);ax.plot([i-.18,i+.18],[avg,avg],color=colors[i],lw=3)
  ax.set_title(title,fontsize=10);ax.set_xticks(range(5));ax.set_xticklabels(labels,rotation=35,ha='right',fontsize=8);ax.grid(axis='y',alpha=.18)
 fig.text(.01,.01,'HGEN uses the native reference total-edge budget and shares one Hin2Vec fit per dataset. GAN steps != diffusion normal2 steps. No significance claim.',fontsize=9)
 fig.tight_layout(rect=[0,.045,1,.93]);fig.savefig(ROOT/(d+'_baseline_comparison.png'),dpi=180);fig.savefig(ROOT/(d+'_baseline_comparison.pdf'));plt.close(fig)
print('two descriptive comparison figures saved')
