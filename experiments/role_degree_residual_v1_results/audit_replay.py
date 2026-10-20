"""Independent saved-array checks; no model calls or generated samples."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--replay",type=Path,required=True)
    a=ap.parse_args();start=time.monotonic()
    receipt=json.loads((a.replay/"receipt.json").read_text())
    assert receipt["status"]=="completed" and receipt["checked_steps"]==384
    paths=sorted(a.replay.glob("seed*_t*.npz"));assert len(paths)==42
    max_degree=max_mu=max_net=0.;records=[];hashes={}
    for p in paths:
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p,allow_pickle=False) as z:
            types=z["types"];active=z["active"];ee=z["endpoints"];ff=z["families"]
            n,R=z["current_degree"].shape
            observed=np.zeros((n,R));mu=np.zeros((n,R));next_mu=np.zeros((n,R))
            for i in np.flatnonzero(active):
                u,v=ee[:,i];r=ff[i];assert 0<=u<v<n
                for node in (u,v):
                    observed[node,r]+=z["Xt"][i]
                    mu[node,r]+=z["p0"][i]
                    next_mu[node,r]+=z["q"][i]
            assert np.array_equal(observed,z["current_degree"])
            max_mu=max(max_mu,float(abs(mu-z["clean_expected_degree"]).max()))
            max_net=max(max_net,float(abs(next_mu-observed-z["expected_births"]+z["expected_deaths"]).max()))
            assert np.array_equal(z["next_state"][active],(z["uniform"][active]<z["q"][active]).astype(np.uint8))
            t=int(p.stem.rsplit("t",1)[1])
            if t==1:np.testing.assert_allclose(z["q"][active],z["p0"][active],atol=2e-12,rtol=0)
            for typ in range(4):
                attention=z[f"attention_type{typ}"]
                np.testing.assert_allclose(attention.sum(1),1,atol=2e-10,rtol=0)
                np.testing.assert_allclose(attention.sum(0),np.rint(attention.sum(0)),atol=2e-9,rtol=0)
            if t==1:
                ix=types==3;soft=z["soft_target_degree"][ix,8];pred=mu[ix,8]
                nonhub=soft<3  # descriptive role band, not quality selection
                records.append(dict(file=p.name,Genre_soft=soft.tolist(),Genre_p0=pred.tolist(),
                    Genre_Xt=observed[ix,8].tolist(),Genre_q=next_mu[ix,8].tolist(),
                    mean_nonhub_p0_minus_soft=float((pred-soft)[nonhub].mean()) if nonhub.any() else None))
    assert max_mu<2e-12 and max_net<2e-12
    result=dict(status="passed",files=42,hashes=hashes,degree_exact=True,
        mean_degree_max_error=max_mu,birth_death_identity_max_error=max_net,
        threshold_decisions_exact=True,t1_q_p0_passed=True,final=records,
        wall_seconds=time.monotonic()-start)
    (a.replay/"independent_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("hashes","final")}))

if __name__=="__main__":main()
