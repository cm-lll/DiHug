"""Posthoc first-order feasibility from saved Gram/actual slopes; no model calls.

Project a recorded displacement d onto {x: G x <= 0}. With H=G G^T,
x=d-G^T lambda, lambda>=0, H lambda-b>=0, lambda*(H lambda-b)=0,
b=G d. Only norm and slopes can be reconstructed from the saved sufficient
statistics. This does NOT reconstruct/commit x or evaluate its finite-step loss.
"""
import itertools,json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/exact_common_updates_20261110'
def main():
    rows=[]
    for folder in sorted(OUT.glob('*_2026*')):
        if not (folder/'updates.json').exists():continue
        for record in json.loads((folder/'updates.json').read_text()):
            z=np.load(folder/f'train_common_u{record["update"]}.npz');H=z['Gram'];b=z['actual_directional_derivatives'];scale=max(float(abs(H).max()),1e-30);hs=H/scale;bs=b/scale;tol=1e-9*max(1.,float(abs(bs).max()));candidates=[]
            for n in range(6):
                for active in itertools.combinations(range(5),n):
                    lam=np.zeros(5)
                    if active:
                        ix=list(active);v=np.linalg.lstsq(hs[np.ix_(ix,ix)],bs[ix],rcond=1e-12)[0]
                        if min(v)<-tol:continue
                        lam[ix]=np.maximum(v,0)
                    slack=hs@lam-bs;res=max(max(0.,float(-slack.min())),float(abs(slack[lam>0]).max()) if np.any(lam>0) else 0.)
                    if res<=tol:candidates.append((float(lam@H@lam),float(lam@lam),lam,res))
            if not candidates:raise RuntimeError('unverified saved-state displacement projection; do not fabricate zero')
            corr_sq,_,lam,res=min(candidates,key=lambda x:(x[0],x[1]));original_sq=record['displacement_norm']**2
            retained_sq=original_sq-2*float(lam@b)+corr_sq
            assert retained_sq>=-1e-8*max(original_sq,1e-30)
            assert corr_sq<=original_sq+1e-8*max(original_sq,1e-30)
            ds,seed=folder.name.rsplit('_',1)
            rows.append(dict(dataset=ds,seed=int(seed),update=record['update'],t=record['t'],original_displacement_norm=record['displacement_norm'],hypothetical_correction_norm=float(np.sqrt(max(0.,corr_sq))),hypothetical_retained_norm_fraction=float(np.sqrt(max(0.,retained_sq)/original_sq)) if original_sq else None,active_constraints=int((lam>0).sum()),normalized_KKT_residual=res,maximum_remaining_slope=float((b-H@lam).max()),interpretation='saved-state first-order geometry only; no parameter write or finite-loss evaluation'))
    with (OUT/'displacement_feasibility.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'displacement_feasibility.json').write_text(json.dumps(dict(rows=rows,model_calls=0,GPU_hours=0,parameter_updates=0,warning='This is a posthoc CPU feasibility calculation, not an executed repair or an independent efficacy test.'),indent=2))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
