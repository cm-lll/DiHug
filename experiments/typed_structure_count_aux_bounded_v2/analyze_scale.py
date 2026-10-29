"""Use only sealed aggregate records; no weights, model calls or GPU."""
import csv,json,math,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent;LAM=.001;S=18336
def write(name,rows):
    with (ROOT/'evidence'/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    inp=ROOT/'evidence/source_relation_bounds.csv'
    old=list(csv.DictReader(inp.open()));out=[]
    for r in old:
        K,m,mu,R=int(r['K']),float(r['m']),float(r['mu']),int(r['R'])
        delta=mu-m;s=math.sqrt(max(m,1));u=delta/s;h=math.hypot(1,u)
        rho=(u/(h+1))*u;slope=u/h/s
        q=dict(dataset=r['dataset'],seed=int(r['seed']),updates=int(r['updates']),readout=r['readout'],t=int(r['t']),relation=r['relation'],K=K,m=m,mu=mu,delta=delta,scale=s,u=u,transition_abs_delta=s,included=bool(K),weighted_contribution=LAM*rho/R if K else 0.,old_weighted_contribution=float(r['weighted_aux_contribution']),weighted_count_derivative=LAM*slope/R if K else 0.,weighted_count_slope_cap=LAM/(R*s) if K else 0.,new_slot_ratio_bound=LAM*S*abs(slope)/R if K else 0.,old_slot_ratio_bound=float(r['gradient_ratio_upper_bound']),actual_slot_or_parameter_gradients=None,positive_CE_alignment=r['positive_CE_alignment'],negative_CE_alignment=r['negative_CE_alignment'],input_sha256=r['input_sha256'])
        # Optional reference direction; NOT a learned module or parameter.
        q['common_shift_ratio_bound']=(LAM*S*abs(slope)*mu*(1-mu/K)/(R*abs(delta))) if K and delta else None
        out.append(q)
    assert len(out)==960
    write('relation_scales.csv',out)
    panels=[]
    for key in sorted({tuple(r[k] for k in ['dataset','seed','updates','readout','t']) for r in out}):
        rows=[r for r in out if tuple(r[k] for k in ['dataset','seed','updates','readout','t'])==key]
        total=sum(r['weighted_contribution'] for r in rows)
        maxrow=max(rows,key=lambda r:r['weighted_contribution'])
        p=dict(zip(['dataset','seed','updates','readout','t'],key));p.update(weighted_aux=total,old_weighted_aux=sum(r['old_weighted_contribution'] for r in rows),largest_contributor=maxrow['relation'],largest_contribution_fraction=maxrow['weighted_contribution']/total if total else None,maximum_slot_ratio_bound=max(r['new_slot_ratio_bound'] for r in rows))
        panels.append(p)
    assert len(panels)==96
    write('panel_scales.csv',panels)
    focus=[r for r in out if (r['dataset']=='IMDB' and r['relation']=='AA') or (r['dataset']=='PubMed' and r['relation'] in ['GD','DC','SS'])]
    assert len(focus)==192
    write('predefined_focus.csv',focus)
    examples=[]
    for name,m,K,delta,prob,y in [
        ('AA_initial_assumed_slot',3,1540,883.7692162394524,.5,0),
        ('GD_moderate_under_assumed_slot',137,2960,-40,.03,0),
        ('GD_wrong_saturated_positive',137,2960,-40,1e-9,1),
        ('zero_target_assumed_slot',0,100,25,.25,0),
        ('correct_count_wrong_topology',3,6,0,.95,0),
    ]:
        R=5 if name.startswith('AA') else 10
        s=math.sqrt(max(m,1));u=delta/s;a=LAM*u/math.hypot(1,u)/(R*s)
        gc=(prob-y)/S;ga=a*prob*(1-prob)
        examples.append(dict(name=name,kind='hypothetical scalar slot, not observed model output',m=m,K=K,delta=delta,R=R,p=prob,y=y,ce_logit_gradient=gc,aux_logit_gradient=ga,absolute_ratio=abs(ga/gc) if gc else None))
    write('hypothetical_slots.csv',examples)
    manifest=dict(source_csv_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),rows=960,panels=96,focus_rows=192,formula_count=1,lambda_values=[LAM],model_loads=0,model_forwards=0,optimizer_steps=0,gpu_used=False,raw_slot_arrays_used=False,missing='E-earlyCN slot p0/logits/labels and parameter gradients unavailable; no backfill',original_reference_rows_preserved=True,finished=datetime.datetime.now().astimezone().isoformat())
    (ROOT/'evidence/analysis_receipt.json').write_text(json.dumps(manifest,indent=2))
    for r in focus:
        if r['updates'] in [0,3000] and r['readout']=='ema_0.99' and r['t']==500:
            print({k:r[k] for k in ['dataset','seed','updates','relation','delta','weighted_contribution','new_slot_ratio_bound','old_slot_ratio_bound']})
if __name__=='__main__':main()
