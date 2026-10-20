"""Read-only aggregation of existing diagnosis and archived large-graph evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--evidence",type=Path,required=True)
    ap.add_argument("--archives",type=Path,required=True);a=ap.parse_args()
    out=a.evidence/"summary.json"
    if out.exists():raise FileExistsError(out)
    rows=read(a.evidence/"traces/records.json");params=read(a.evidence/"traces/parameter_changes.json")
    seeds=[]
    for seed in sorted({r["seed"] for r in rows}):
        rr=[r for r in rows if r["seed"]==seed]
        pp=next(p for p in params if p["seed"]==seed)
        seeds.append(dict(seed=seed,states=len(rr),
            maxima={key:max(r["target_difference"][key]["max_abs"] for r in rr)
                    for key in ["role_context","role_projection","MG_logit_gap","MG_p0"]},
            repeat_noise_maxima={key:max(r["repeat_noise"][key]["max_abs"] for r in rr)
                                for key in rr[0]["repeat_noise"]},
            genre_encoded_weighted_mean_max=max(r["by_type"]["3"]["encoded_weighted_mean"]["max_abs"] for r in rr),
            genre_attention_max_deviation=max(r["by_type"]["3"]["attention"]["extreme"]["deviation_from_frequency_max"] for r in rr),
            genre_attention_entropy_range=[
                min(r["by_type"]["3"]["attention"]["extreme"]["entropy_mean"] for r in rr),
                max(r["by_type"]["3"]["attention"]["extreme"]["entropy_mean"] for r in rr)],
            initial_validation_abs_error=pp["initial_validation_abs_error"],
            branch_parameter_changes=[p for p in pp["parameters"] if p["name"].startswith("branch.")]))
    train=read(a.archives/"training_analysis.json");graphs=read(a.archives/"per_graph_comparison.json")
    curves={}
    for ds,v in train.items():
        table=[]
        for u in [3000,6000,9000,10000,12000]:
            val=[r for r in v["validation"] if r["update"]==u]
            low=next(r for r in val if r["t"]==1)
            g=next(r for r in graphs if r["dataset"]==ds and r["method"]=="joint_CN" and r["update"]==u)
            table.append(dict(update=u,mean_fixed_time_ce=statistics.mean(r["ce"] for r in val),
                low_t_ce=low["ce"],low_t_positive_ce=low["positive_ce"],
                low_t_expected_edges=low["expected_edges"],generated_edges=g["edges"],
                generated_max_degree=g["maximum_degree"],generated_lcc=g["lcc"]))
        curves[ds]=table
    measurements=read(a.evidence/"imdb_gpu_probe/measurements.json");gpu=[]
    for label in ["original_C","late_C"]:
        rr=[r for r in measurements if r["model"]==label]
        gpu.append(dict(model=label,steady_n=3,
            median_update_seconds=statistics.median(r["wall_seconds"] for r in rr if r.get("timing_class")=="steady"),
            complete_20_block_seconds=rr[-1]["wall_seconds"],
            max_allocated_mib=max(r["peak_allocated_mib"] for r in rr),
            max_reserved_mib=max(r["peak_reserved_mib"] for r in rr),
            max_observed_device_total_used_mib=max(r["device_total_used_mib"] for r in rr),
            device_memory_sampling="after each synchronized update/step; NOT a continuous peak sampler"))
    result=dict(base_commit="c1163a885322c9dffc2919c8968c6a98adf375c7",frozen_C_diagnosis=seeds,
        large_graph_archive_source={name:sha(a.archives/name) for name in ["training_analysis.json","per_graph_comparison.json"]},
        large_graph_curves=curves,gpu_resource_gate=gpu,
        claim_boundaries=["No new learning-panel or full science chain",
          "10 full-IMDB temporary engineering optimizer updates only; no trained weights retained",
          "Existing trained C CPU diagnostics are not new training",
          "Late-fusion response witnesses are not learned generation efficacy",
          "Fixed-validation vs chain discrepancy does not identify the generating-input distribution shift by itself"])
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
if __name__=="__main__":main()
