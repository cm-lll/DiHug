"""New trained weights on archived old Xt; feature masking is not a baseline."""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/"role_degree_residual_v1"),str(HERE.parent/"role_specialization_v1")]
from panel import load_panel,c,fingerprint,save
from residual_model import DegreeResidualDenoiser
from hsgd.computational import FrozenTypedContext

def main():
    ap=argparse.ArgumentParser()
    for k in ("run","panel","replay","output"):ap.add_argument("--"+k,type=Path,required=True)
    a=ap.parse_args();started=time.monotonic()
    a.output.mkdir(exist_ok=False)
    ledger=json.loads((a.run/"activation.json").read_text())
    cfg=json.loads((a.run/"resolved_config.json").read_text())
    hashes=json.loads((a.run/"weight_hashes.json").read_text())
    records,targets=load_panel(a.panel);case=records[0]["case"]
    torch.set_num_threads(1);torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False)
    rows=[]
    for seed in cfg["train_seeds"]:
        path=a.run/f"seed{seed}/degree_C_final_1000.pt"
        assert fingerprint(path)==hashes[str(path.relative_to(a.run))]
        model=DegreeResidualDenoiser(case["n_types"],case["n_families"],
            family_pairs=case["space"].pairs,prior=case["prior"],T=64,**cfg["model"])
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
        model.eval();saved=model.degree_fusion.weight.detach().clone()
        for condition in ("balanced","extreme"):
            assert time.time()<ledger["deadline_epoch"] and not (a.run/"STOP").exists()
            src=a.replay/f"seed{seed}_{condition}_t01.npz"
            with np.load(src,allow_pickle=False) as z:
                state=z["Xt"].copy();old=z["p0"].copy();ep=z["endpoints"].copy()
                fam=z["families"].copy();active=z["active"].copy();soft=z["soft_target_degree"].copy()
            model.set_target(targets[condition]);ee,ff=c.state_edges(case,state)
            outputs=[]
            for mask in (False,True,False):
                with torch.no_grad():
                    model.degree_fusion.weight.copy_(torch.zeros_like(saved) if mask else saved)
                    ctx=FrozenTypedContext(model,case["space"].types,ee,ff,1)
                    p0=np.zeros(case["space"].total)
                    try:
                        plan=c.plan_for(case,[cfg["sampling_seeds"][0],1,61],4)
                        for b in range(4):
                            ids,f,q=plan.block(b)
                            _,ll,p=ctx.predict(q,f,1/64,1,details=True)
                            p0[ids]=c.check_prob(p,f,case)
                    finally:ctx.close()
                outputs.append(p0)
            assert np.array_equal(outputs[0],outputs[2])
            def genre_degree(p):
                ans=np.zeros(len(case["space"].types))
                for i in np.flatnonzero(active & (fam==8)):
                    ans[ep[0,i]]+=p[i];ans[ep[1,i]]+=p[i]
                return ans[case["space"].types==3]
            rows.append(dict(seed=seed,condition=condition,input_sha256=fingerprint(src),
                weight_sha256=fingerprint(path),old_G_p0=genre_degree(old).tolist(),
                new_G_p0=genre_degree(outputs[0]).tolist(),masked_new_G_p0=genre_degree(outputs[1]).tolist(),
                soft_G_target=soft[case["space"].types==3,8].tolist(),
                feature_mask_max_p0_difference=float(abs(outputs[0]-outputs[1]).max()),
                same_condition_repeat_max_difference=0.,
                added_weight_norm=float(saved.norm())))
            np.savez_compressed(a.output/f"seed{seed}_{condition}.npz",
                original=old,new=outputs[0],new_feature_masked=outputs[1])
    save(a.output/"records.json",rows)
    save(a.output/"receipt.json",dict(status="completed",fixed_states=6,full_block_predictions=72,
        meaning="trained-model feature masking, not independently trained ablation",
        wall_seconds=time.monotonic()-started,cumulative_seconds=time.time()-ledger["activation_epoch"]))

if __name__=="__main__":main()
