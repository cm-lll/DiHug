"""Independent stored-array audit; no model invocation or graph generation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"role_error_recovery_v1"))
from diagnostic import load_panel,c,slots,metrics,MarginalKernel,fingerprint

def put(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:close(a[k],b[k])
    elif isinstance(a,list): np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)
    elif a is None or isinstance(a,str):assert a==b
    else:np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--panel",type=Path,required=True);a=ap.parse_args()
    start=time.monotonic()
    receipt=json.loads((a.run/"receipt.json").read_text());assert receipt["status"]=="completed"
    records,targets=load_panel(a.panel);cases={r["id"]:r["case"] for r in records}
    rows=json.loads((a.run/"records.json").read_text());counts=defaultdict(int)
    for r in rows:
        cc=cases[r["graph"]];stem=f"{r['graph']}_t{r['t']}"
        inp=np.load(a.run/(stem+"_input.npz"));out=np.load(a.run/f"{r['seed']}_{stem}_{r['variant']}.npz")
        clean=inp["clean"];x=inp["ordinary"] if r["variant"]=="ordinary" else inp[r["variant"]]
        mask=np.zeros(len(x),bool) if r["variant"]=="ordinary" else inp[r["variant"]+"_edit"]
        # Independently derive canonical endpoints/families and clean slot mask.
        ee,ff=slots(cc);assert np.array_equal(ee,inp["endpoints"]) and np.array_equal(ff,inp["families"])
        assert np.array_equal(np.flatnonzero(clean),np.sort(cc["clean_ids"]))
        coverage=np.zeros(len(x),int);h=hashlib.sha256()
        for b in range(4):
            ids,fam,query=(inp[f"{name}{b}"] for name in ("ids","families","query"))
            assert np.array_equal(ff[ids],fam) and np.array_equal(ee[:,ids],query)
            coverage[ids]+=1
            for ar in (ids,fam,query):h.update(np.ascontiguousarray(ar).tobytes())
        assert h.hexdigest()==r["query_sha256"]
        assert np.array_equal(coverage,np.repeat(cc["active"].astype(int),cc["space"].counts))
        kernel=MarginalKernel(cc["prior"],c.schedule(64))
        np.testing.assert_allclose(out["q"],kernel.reverse(out["p0"],x,ff,r["t"]-1,r["t"]),atol=1e-12,rtol=0)
        pi=cc["prior"][ff]
        assert (out["p0"][pi==0]==0).all() and (out["p0"][pi==1]==1).all()
        close(r["metrics"],metrics(cc,clean,x,mask,r["selected_node"],out["p0"],out["q"]))
        # Independent flow and two-endpoint degree checks, not only shared summary code.
        assert abs(out["q"].sum()-x.sum()-r["metrics"]["flows"]["global"]["net"])<1e-10
        actual=np.zeros((34,10))
        for i,(u,v) in enumerate(ee.T):
            actual[u,ff[i]]+=out["p0"][i];actual[v,ff[i]]+=out["p0"][i]
        np.testing.assert_allclose(actual,r["metrics"]["role"]["p0"]["expected_degree"],atol=1e-12)
        if r["variant"]!="ordinary":
            assert int(mask.sum())==r["edits"]==np.count_nonzero(x!=inp["ordinary"])
            assert ((pi[mask]>0)&(pi[mask]<1)&(ff[mask]==8)).all()
            if r["condition"]=="extreme":assert (clean[mask]==0).all() and (x[mask]==1).all() and (inp["ordinary"][mask]==0).all()
            else:assert (clean[mask]==1).all() and (x[mask]==0).all() and (inp["ordinary"][mask]==1).all()
            base=np.load(a.run/f"{r['seed']}_{stem}_ordinary.npz")
            close(r["paired_ordinary"],metrics(cc,clean,inp["ordinary"],mask,r["selected_node"],base["p0"],base["q"]))
        counts[r["seed"]]+=1
    panel=json.loads((a.run/"panel.json").read_text())
    expected=12+sum(x["status"]=="ready" for x in panel)
    assert dict(counts)=={2026102011:expected,2026102012:expected,2026102013:expected}
    sources=json.loads((a.run/"sources.json").read_text())
    assert all(fingerprint(p)==h for p,h in sources.items())
    grouped=defaultdict(list);ordinary=defaultdict(list)
    for r in rows:
        m=r["metrics"]
        if r["variant"]=="ordinary":
            ordinary[r["seed"],r["condition"],r["t"]].append({
                "true_p0":m["categories"]["remaining_true_MG"]["p0"],"true_q":m["categories"]["remaining_true_MG"]["q"],
                "nonedge_p0":m["categories"]["untouched_nonedge_MG"]["p0"],"nonedge_q":m["categories"]["untouched_nonedge_MG"]["q"],
                "MG_p0_edges":m["by_family"]["expected_p0"][8],"MG_q_edges":m["by_family"]["expected_q"][8],
                "G_expected_W1":m["role"]["p0"]["G_expected_W1"]})
            continue
        b=r["paired_ordinary"];k="added_false" if r["condition"]=="extreme" else "deleted_true"
        data={"edits":r["edits"]}
        for field in ("p0","q"):
            for label,cat in [("edited",k),("true","remaining_true_MG"),("nonedge","untouched_nonedge_MG")]:
                data[f"{label}_{field}"]=m["categories"][cat][field]
                data[f"{label}_{field}_ordinary"]=b["categories"][cat][field]
                data[f"{label}_{field}_delta"]=data[f"{label}_{field}"]-data[f"{label}_{field}_ordinary"]
            data[f"G_expected_W1_{field}"]=m["role"][field]["G_expected_W1"]
            data[f"G_expected_W1_{field}_delta"]=m["role"][field]["G_expected_W1"]-b["role"][field]["G_expected_W1"]
        for scope in ("global","selected_node","edited_endpoints","MG"):
            for key in ("births","deaths","net","expected_next_edges"):
                data[f"{scope}_{key}"]=m["flows"][scope][key]
                data[f"{scope}_{key}_delta"]=m["flows"][scope][key]-b["flows"][scope][key]
        grouped[r["seed"],r["condition"],r["t"],r["variant"]].append(data)
    def aggregate(groups,names):
        return [dict(zip(names,k),graphs=len(v),**{field:float(np.mean([x[field] for x in v])) for field in v[0]}) for k,v in sorted(groups.items())]
    seed_rows=aggregate(grouped,["seed","condition","t","variant"])
    base_rows=aggregate(ordinary,["seed","condition","t"])
    macro=defaultdict(list)
    for r in seed_rows:macro[r["condition"],r["t"],r["variant"]].append({k:v for k,v in r.items() if k not in ("seed","condition","t","variant","graphs")})
    result=dict(per_seed=seed_rows,macro=aggregate(macro,["condition","t","variant"]),ordinary_per_seed=base_rows,
                semantics="Average two source graphs within seed, then equal weight three training seeds; not edge-level CI.")
    put(a.run/"summary.json",result)
    put(a.run/"audit.json",dict(status="passed",records=len(rows),per_seed=dict(counts),infeasible=receipt["infeasible"],
         checks=["slot identity","query bytes and coverage","exact edit masks/counts/support","posterior",
                 "metric recomputation","independent endpoint degrees/net flow","source rehash"],
         model_calls=0,wall_seconds=time.monotonic()-start))
    print(json.dumps(dict(audit="passed",records=len(rows),wall=time.monotonic()-start)))
if __name__=="__main__":main()
