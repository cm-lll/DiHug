"""CPU-only complete CE tables; fixed t inputs are not unseen-graph validation."""
from .formal_runtime import *
import pathlib
import csv,json,statistics as st
def write(name,rows):
 save(OUT/'analysis'/(name+'.json'),rows)
 if rows:
  with (OUT/'analysis'/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
 (OUT/'analysis').mkdir(exist_ok=True);raw=[];bands=[];val=[];rel=[];checks=[]
 os.environ['MPLCONFIGDIR']=str(OUT/'temporary/matplotlib');os.environ['TMPDIR']=str(OUT/'temporary')
 for task in TASKS:
  ds,seed,_=task.split('_',2)
  fixed={}
  for arm,base in [('INPUT',CONTROL),('OUTPUT',OUT)]:
   folder=base/'run'/task;path=folder/'training.jsonl'
   if not path.exists():continue
   records=[]
   for line in path.read_text().splitlines():
    try:records.append(json.loads(line))
    except json.JSONDecodeError:pass # concurrent incomplete trailing line only
   for r in records:
    raw.append(dict(dataset=ds,train_seed=int(seed),arm=arm,update=r['update'],t=r['t'],CE=r.get('unweighted_CE',r['loss']),gradient_L2=r['gradient_norm']))
   for end in range(500,3001,500):
    for band,lo,hi in [('low',1,333),('middle',334,666),('high',667,1000)]:
     rr=[r for r in records if end-500<r['update']<=end and lo<=r['t']<=hi];vv=[r.get('unweighted_CE',r['loss']) for r in rr]
     bands.append(dict(dataset=ds,train_seed=int(seed),arm=arm,start_update=end-499,end_update=end,band=band,t_min=lo,t_max=hi,n=len(vv),mean=st.mean(vv) if vv else None,std=st.stdev(vv) if len(vv)>1 else None,minimum=min(vv) if vv else None,maximum=max(vv) if vv else None))
   vp=folder/'validation.json'
   if not vp.exists():continue
   for v in read(vp):
    for mode in ['raw','ema_0_99']:
     for r in v[mode]['rows']:
      key=(v['updates'],mode,r['t']);h=r['input_sha256']
      if arm=='INPUT':fixed[key]=h
      elif key not in fixed or fixed[key]!=h:raise ValueError('fixed validation input mismatch')
      row=dict(dataset=ds,train_seed=int(seed),arm=arm,update=v['updates'],readout=mode,t=r['t'],input_sha256=h,CE=r['ce_sum_over_S'])
      for label in ['positive','negative']:
       n=sum(f[label+'_count'] for f in r['family']);row[label+'_count']=n;row[label+'_CE']=sum(f[label+'_count']*f[label+'_ce'] for f in r['family'] if f[label+'_ce'] is not None)/n if n else None
      val.append(row)
      for f in r['family']:rel.append(dict(row,**f))
   if arm=='OUTPUT':
    checks.append(dict(task=task,recorded_updates=len(records),paired_inputs=all(r['paired_input_exact'] for r in records),formula='normal1 CE + .01 typed degree/triangle projected quantiles'))
 for name,rows in [('training_raw',raw),('training_500step_bands',bands),('validation_total',val),('validation_relations',rel),('time_window_checks',checks)]:write(name,rows)
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,axes=plt.subplots(2,2,figsize=(13,8))
  for ax,task in zip(axes.flat,TASKS):
   ds,seed,_=task.split('_',2)
   for arm,color in [('INPUT','tab:blue'),('OUTPUT','tab:orange')]:
    v=[r for r in raw if r['dataset']==ds and r['train_seed']==int(seed) and r['arm']==arm]
    ax.scatter([r['update'] for r in v],[r['CE'] for r in v],s=1,alpha=.12,color=color)
    sm=[(v[k+49]['update'],st.mean(r['CE'] for r in v[k:k+50])) for k in range(0,len(v)-49,50)]
    ax.plot([x for x,y in sm],[y for x,y in sm],label=arm,color=color)
   ax.set_title(ds+' '+seed);ax.set_xlabel('optimizer updates');ax.set_ylabel('unweighted CE (raw + disjoint50 mean)');ax.legend()
  fig.tight_layout();fig.savefig(OUT/'analysis/training_CE.png',dpi=160);plt.close(fig)
 except ImportError as ex:save(OUT/'analysis/plot_missing.json',dict(error=repr(ex)))
 print('CE tables',len(raw),len(val),len(rel))
if __name__=='__main__':main()
