"""Re-score only previously viewed terminal graphs; no held-out or model call."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json,hashlib,time
from pathlib import Path
import numpy as np
from feedback import count_terms
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/gdpo_feedback_repair_20261113'
PANEL=ROOT/'results/paper_ablation_20261112T190000'

def main():
    start=time.time();p=json.loads((PANEL/'protocol.json').read_text());refs=json.loads((PANEL/'reference_metrics.json').read_text());rows=[];bindings={}
    for job in p['jobs']:
        if job['architecture']!='A' or job['reward']!='typed':continue
        ds=job['dataset'];data=np.load(p['parents'][job['parent']]['data']);types=data['types'];pairs=data['schema']
        ref=np.array(refs[ds]['family_edges']);counts=np.bincount(types)
        S=np.array([counts[a]*(counts[a]-1)//2 if a==b else counts[a]*counts[b] for a,b in pairs])
        for u in range(1,5):
            directory=PANEL/job['id']/f'{ds}_T_eager'
            if job['cached_complete']:
                directory=ROOT/(f'results/three_arm_learning_pilot_20261111T211641/{ds}_T_eager' if u<=2 else f'results/terminal_continuation_20261112T003000/{ds}_T_eager')
            batch=json.loads((directory/f'round{u}_batch.json').read_text());old=[];new=[];h=[]
            for k,plan in enumerate(job['terminal_plan'][str(u)]):
                path=directory/f'train_round{u}_seed{plan["seed"]}/graph.npz'
                bindings[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                z=np.load(path);m=np.bincount(z['families'],minlength=len(ref))
                old.append(float(count_terms(m,ref,S,'legacy').sum()))
                new.append(float(count_terms(m,ref,S,'bounded_variance').sum()))
                h.append(batch['costs'][k]-old[-1])
            dh=h[0]-h[1];dn=old[0]-old[1];dn_new=new[0]-new[1]
            rows.append(dict(job=job['id'],dataset=ds,update=u,H_gap=dh,old_N_gap=dn,new_N_gap=dn_new,
                old_N_dominates=abs(dn)>abs(dh),new_N_dominates=abs(dn_new)>abs(dh),
                old_total_opposes_H=(dh+dn)*dh<0,new_total_opposes_H=(dh+dn_new)*dh<0,
                old_total_gap=dh+dn,new_total_gap=dh+dn_new))
    with (OUT/'development_rescore.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for ds in ('IMDB','PubMed'):
        r=[v for v in rows if v['dataset']==ds]
        summary[ds]={key:sum(v[key] for v in r) for key in ('old_N_dominates','new_N_dominates','old_total_opposes_H','new_total_opposes_H')};summary[ds]['batches']=len(r)
    out=dict(status='complete',seconds=time.time()-start,batches=len(rows),summary=summary,
        bindings=bindings,model_calls=0,GPU_hours=0,scope='Viewed development graphs only; alternative rankings are not evidence of better generation',
        candidate='H unchanged + mean_r(1-exp(-0.5*(m-mref)^2/[mref*(1-mref/S)])); exact fixed supports checked')
    (OUT/'cpu_review.json').write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
