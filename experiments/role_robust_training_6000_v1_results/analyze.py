"""Read-only aggregation of the frozen paired panel; no model calls."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def average(rows):
    return sum(rows) / len(rows) if rows else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert read(a.run / "audit.json")["status"] == "passed"
    receipt = read(a.run / "worker_receipt.json")
    assert receipt["status"] == "completed"
    rows = read(a.run / "paired_results.json")["per_seed"]
    generations = read(a.run / "generation.json")
    assert len(rows) == 36
    a.output.mkdir(parents=True, exist_ok=False)
    metrics = ["G_W1", "G_joint_TV", "G_max", "G_high_ge12", "G_multiple_hubs",
               "G_low_le1", "G_zero", "MG_signed_bias", "MG_absolute_error", "edges",
               "connected", "lcc", "isolates", "triangles", "closure_path", "closure_pair",
               "topology_classes", "inverse_simpson"]
    macro = []
    for point in (1000, 3000, 6000):
        for cond in ("balanced", "extreme"):
            for arm in ("normal2", "robust"):
                group = [r for r in rows if (r["checkpoint_update"], r["condition"], r["arm"]) == (point, cond, arm)]
                assert len(group) == 3
                value = dict(checkpoint_update=point, condition=cond, arm=arm, training_seeds=3, graphs=24)
                for k in metrics:
                    vals = [r[k] for r in group]
                    value[k] = average(vals) if all(v is not None for v in vals) else None
                value["family_signed_bias"] = [average([r["family_signed_bias"][j] for r in group]) for j in range(10)]
                raw = [r for r in generations if (r["checkpoint_update"],r["condition"],r["arm"]) == (point,cond,arm)]
                value["joint_TV_by_type"]={typ:average([r["comparison"]["joint_relation_degree_TV"][typ] for r in raw]) for typ in ("Director","Actor","Movie","Genre")}
                value["degree_W1_by_type_relation"]={typ:[average([r["comparison"]["typed_relation_degree_W1"][typ][j] for r in raw]) for j in range(10)] for typ in ("Director","Actor","Movie","Genre")}
                macro.append(value)
    with (a.output / "generation_macro.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint_update", "condition", "arm"] + metrics)
        w.writeheader()
        for r in macro:
            w.writerow({k: r[k] for k in w.fieldnames})
    validation = []
    for path in sorted(a.run.glob("seedjobs/seed*/seed*/*_validation_*.json")):
        seed = int(path.parent.name[4:])
        arm, u = path.stem.split("_validation_")
        v = read(path)
        for condition in ("balanced", "extreme"):
            for t in (1, 32, 64):
                group = [r for r in v["rows"] if r["condition"] == condition and r["t"] == t]
                assert group
                losses = [sum(r["ce_sum_by_family_label"][8][j] for r in group) for j in range(2)]
                counts = [sum(r["counts_by_family_label"][8][j] for r in group) for j in range(2)]
                validation.append(dict(seed=seed, arm=arm, update=int(u), condition=condition, t=t,
                    ce=average([r["ce"] for r in group]), MG_positive_ce=losses[1]/counts[1] if counts[1] else None,
                    MG_negative_ce=losses[0]/counts[0] if counts[0] else None,
                    MG_positive_count=counts[1], MG_negative_count=counts[0],
                    MG_expected_bias=average([r["expected_edge_bias"][8] for r in group])))
    assert len(validation) == 504
    with (a.output / "validation_curves.csv").open("w") as f:
        w=csv.DictWriter(f, fieldnames=list(validation[0]));w.writeheader();w.writerows(validation)
    updates = [json.loads(s) for s in (a.run / "updates.jsonl").read_text().splitlines()]
    train = []
    for seed in (2026102011, 2026102012, 2026102013):
        for arm in ("normal2", "robust"):
            group = [r for r in updates if r["seed"] == seed and r["arm"] == arm]
            assert len(group) == 6000
            for end in range(100, 6001, 100):
                chunk = group[end-100:end]
                train.append(dict(seed=seed, arm=arm, end_update=end,
                    mixed_objective_loss=average([r["loss"] for r in chunk]),
                    seconds=average([r["seconds"] for r in chunk]),
                    gradient_norm=average([r["gradient_norm"] for r in chunk])))
    with (a.output / "training_curves.csv").open("w") as f:
        w=csv.DictWriter(f, fieldnames=list(train[0]));w.writeheader();w.writerows(train)
    pressure = []
    for path in sorted(a.run.glob("pressure_seed*.json")):
        pressure += read(path)
    pressure_macro=[]
    for arm in ("normal2", "robust"):
        for cond in ("balanced", "extreme"):
            for t in (1, 4, 8):
                for variant in ("ordinary", "add1", "add5", "delete1", "delete3"):
                    g=[r for r in pressure if (r["arm"], r["condition"], r["t"], r["variant"])==(arm,cond,t,variant)]
                    assert len(g)==6
                    v=dict(arm=arm, condition=cond, t=t, variant=variant)
                    for cat in ("edited", "remaining_true", "untouched_nonedge"):
                        vals=[r["categories"][cat]["q"] for r in g]
                        v[cat+"_q"]=average(vals) if all(x is not None for x in vals) else None
                        vals=[r["categories"][cat]["ordinary_q_same_positions"] for r in g]
                        v[cat+"_ordinary_q"]=average(vals) if all(x is not None for x in vals) else None
                    pressure_macro.append(v)
    resources=read(a.run / "seed_resources.json")
    timing=[]
    for arm in ("normal2", "robust"):
        x=sorted(r["seconds"] for r in updates if r["arm"]==arm and r["update"]>100)
        timing.append(dict(arm=arm, stable_updates=len(x), median_seconds=statistics.median(x),
                           p90_seconds=(lambda k:x[int(k)]+(k-int(k))*(x[min(int(k)+1,len(x)-1)]-x[int(k)]))(.9*(len(x)-1)),
                           total_update_seconds=sum(r["seconds"] for r in updates if r["arm"]==arm)))
    validation_seconds=sum(read(path)["seconds"] for path in a.run.glob("seedjobs/seed*/seed*/*_validation_*.json"))
    sampling_seconds=sum(r["seconds"] for r in generations)
    for path in a.run.glob("seedjobs/seed*/input_bindings.json"):
        import hashlib
        for name,h in read(path).items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,(name,"historical input changed")
    runtime=dict(training_update_sum_seconds=sum(r["seconds"] for r in updates),sampling_chain_sum_seconds=sampling_seconds,
                 fixed_validation_sum_seconds=validation_seconds,final_pressure_forward_sum_seconds=sum(r["seconds"] for r in pressure),
                 scientific_worker_CPU_seconds=sum(r["CPU_seconds"] for r in resources),
                 historical_inputs_postverified=True,quantiles="linear interpolation; stable updates exclude first100 per arm and seed")
    result=dict(macro=macro, per_seed=rows, validation=validation, pressure_macro=pressure_macro,
                resource_receipts=resources, update_timing=timing, main_receipt=receipt,runtime=runtime,
                scope="3 independent paired train seeds; 8 paired draws per condition and checkpoint; descriptive development panel")
    (a.output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status="completed", validation_rows=len(validation), generation_macro_rows=len(macro))))

if __name__ == "__main__":
    main()
