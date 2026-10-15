"""Fail-closed, read-only evidence audit. Never runs a model or sampler.

Audit old runs with --source-package pointing to their unchanged source package.
The audit interpreter/backend is not compared with the recorded run GPU.
"""
import argparse
import hashlib
import json
import numbers
from pathlib import Path
import sys
import numpy as np
import torch

STAGES = ("A", "B", "C-small", "C-imdb")
COMMON_PASSED = ("summary.json", "run.json", "events.jsonl", "progress.json", "launcher.json")
REQUIRED_FILES = {
    "A": {"passed": COMMON_PASSED + ("smoke_step1.pt", "smoke_final.pt"),
          "incomplete": ("summary.json",), "failed": ("summary.json",)},
    "B": {"passed": COMMON_PASSED + ("preflight_last.pt", "baseline_checks.json"),
          "incomplete": ("summary.json",), "failed": ("summary.json",)},
    "C-small": {"passed": COMMON_PASSED + ("chain_last.pt", "engineering_final_graph.npz"),
                "incomplete": ("summary.json",), "failed": ("summary.json",)},
    "C-imdb": {"passed": COMMON_PASSED + ("chain_last.pt", "engineering_final_graph.npz", "sweep_cost.json"),
               "incomplete": ("summary.json",), "failed": ("summary.json",)},
}
PACKAGE = Path(__file__).resolve().parents[2]


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read_json(path):
    with Path(path).open() as f:
        value = json.load(f)
    require(isinstance(value, dict), "JSON object required: " + str(path))
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(value, name, low=0):
    require(isinstance(value, numbers.Integral) and not isinstance(value, bool), name + " must be an integer")
    require(value >= low, name + " below allowed range")
    return int(value)


def finite(value, name, low=0):
    require(isinstance(value, numbers.Real) and not isinstance(value, bool), name + " must be numeric")
    require(np.isfinite(value) and value >= low, name + " nonfinite/out of range")
    return float(value)


def needed(directory, names):
    missing = [n for n in names if not (directory/n).is_file() or (directory/n).stat().st_size == 0]
    require(not missing, "missing/empty required evidence: " + ", ".join(missing))


def source_info(package):
    package = Path(package).resolve()
    cfg_path = package/"hsgd/bw_preflight/config.json"
    cfg = read_json(cfg_path)
    files = sorted((package/"hsgd").rglob("*.py")) + [cfg_path]
    require(len(files) > 1, "source package has no Python source")
    hashes = {str(p.relative_to(package)): digest(p) for p in files}
    manifest = next((package/n for n in ("manifest_learning_v9.json", "manifest_learning_v7.json", "manifest_runtime_v6.json", "manifest_perf_v5.json", "manifest_v4.json", "manifest_v3.json", "manifest_v2.json") if (package/n).exists()), None)
    if manifest is not None:
        for name, record in read_json(manifest)["files"].items():
            require(digest(package/name) == record["sha256"], "source manifest mismatch: " + name)
    return dict(package=package, config=cfg, code=hashes)


def common_check(directory, summary, source):
    run = read_json(directory/"run.json")
    require(run["stage"] == summary["stage"], "summary/run stage mismatch")
    binding = run["initial_binding"]
    require(binding["code"] == source["code"], "source code binding mismatch; select the run's unchanged --source-package")
    require(binding["config"] == source["config"] == summary["config"], "configuration binding mismatch")
    require(run["environment"] == summary["environment"], "summary/run environment mismatch")
    keys = ("packages", "device_name", "hip", "cuda")
    schema = source["config"].get("runtime_binding_schema", 1)
    require(schema in (1, 2), "unknown runtime binding schema")
    if schema == 2:
        keys += ("python_version", "torch_runtime_version")
    expected = {k: summary["environment"][k] for k in keys}
    if "deterministic_algorithms" in binding["backend"]:
        expected.update({k:summary["environment"][k] for k in ("deterministic_algorithms","deterministic_warn_only")})
    require(binding["backend"] == expected, "backend binding mismatch")
    if (source["package"]/"hsgd/runtime.py").exists():
        require(binding.get("graph_backend") in ("upstream","legacy"), "missing actual graph backend")
        require(summary.get("graph_backend") == binding["graph_backend"], "graph backend mismatch")
    require(summary["code_unchanged"] is True, "run did not certify unchanged source")
    if "checks" in summary:
        require(isinstance(summary["checks"], list) and len(summary["checks"]) > 0, "empty summary checks")
    progress = read_json(directory/"progress.json")
    require(progress.get("phase") == "stage_finished" and progress.get("status") == "passed", "missing successful terminal progress")
    events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines() if line.strip()]
    require(events and events[-1].get("phase") == "stage_finished" and events[-1].get("status") == "passed", "missing successful terminal event")
    launcher = read_json(directory/"launcher.json")
    require(integer(launcher["child_exit_code"], "child exit") == 0, "launcher child did not succeed")
    return binding, ["required files nonempty", "stage consistency", "source/config/backend binding", "terminal ledger and launcher"]


def rng_check(rng):
    require(isinstance(rng,dict) and all(k in rng for k in ("generator","python","numpy","torch","accelerator")), "missing RNG state")
    require(isinstance(rng["generator"],dict) and rng["generator"], "empty NumPy generator state")
    for key in ("python","numpy"):
        require(isinstance(rng[key],(list,tuple)) and len(rng[key]) > 0, "empty global RNG state: " + key)
    require(torch.is_tensor(rng["torch"]) and rng["torch"].numel() > 0, "empty Torch RNG state")
    if rng["accelerator"] is not None:
        require(isinstance(rng["accelerator"],(list,tuple)) and rng["accelerator"], "empty accelerator RNG state")
        require(all(torch.is_tensor(x) and x.numel() > 0 for x in rng["accelerator"]), "invalid accelerator RNG state")


def training_checkpoint(path, expected_binding, count):
    # User-owned engineering artifacts only; loading is CPU-only.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(ck, dict), "checkpoint must be a dictionary")
    require(ck["binding"] == expected_binding, "checkpoint source/config/data binding mismatch: " + path.name)
    require(integer(ck["completed_updates"], "checkpoint updates") == count, "checkpoint completed updates mismatch")
    model = ck["model"]
    require(isinstance(model, dict) and model, "empty model checkpoint")
    require(all(torch.is_tensor(v) and bool(torch.isfinite(v).all()) for v in model.values()), "invalid/nonfinite model state")
    optimizer = ck["optimizer"]
    require(optimizer.get("state") and optimizer.get("param_groups"), "empty optimizer checkpoint")
    for state in optimizer["state"].values():
        step = state["step"]
        if torch.is_tensor(step):
            require(step.numel() == 1, "Adam step must be scalar")
            step = step.item()
        require(finite(step, "Adam step") == count, "Adam/completed update count mismatch")
        for value in state.values():
            if torch.is_tensor(value):
                require(bool(torch.isfinite(value).all()), "nonfinite optimizer state")
    for group in optimizer["param_groups"]:
        require(group["params"], "empty optimizer parameter group")
        for key in ("lr", "weight_decay"):
            require(group[key] == expected_binding["config"]["optimizer"][key], "optimizer setting mismatch: " + key)
    history = ck["history"]
    require(isinstance(history,list) and len(history) == count, "checkpoint history/update count mismatch")
    for record in history:
        if "graph_backend" in expected_binding:
            require(record.get("backend") == record.get("actual_dispatch") == expected_binding["graph_backend"], "checkpoint history dispatch mismatch")
        finite(record["loss"], "loss")
        finite(record["gradient_norm"], "gradient norm")
        require(finite(record["edge_head_max_update"], "parameter update") > 0, "no recorded parameter change")
        require(record["optimizer_update"] is True, "history contains a non-update")
    rng_check(ck["rng"])
    return ck


def case_arrays(stage, source):
    if stage in ("A","C-small"):
        return (np.array([0,0,0,1,1,1],dtype=np.int64),
                np.array([[0,1],[1,2],[2,3],[3,4],[4,5]],dtype=np.int64),
                "deterministic six-node engineering fixture")
    cfg, package = source["config"], source["package"]
    path = package/"data/IMDB_movie_reference.npz"
    require(digest(path) == cfg["imdb_sha256"], "reference data hash mismatch")
    with np.load(path,allow_pickle=False) as z:
        types,edges,relations = z["types"].copy(),z["edges"].copy(),z["relations"].copy()
    require(len(types) == cfg["imdb_nodes"] and len(edges) == cfg["imdb_edges"], "reference size mismatch")
    require(np.bincount(types).tolist() == cfg["imdb_type_counts"], "reference type inventory mismatch")
    pairs = [(a,b) for a in range(int(types.max())+1) for b in range(a,int(types.max())+1)]
    lookup = {p:i for i,p in enumerate(pairs)}
    fam = [lookup[tuple(sorted((int(types[u]),int(types[v]))))] for u,v in edges]
    require(np.array_equal(np.array(fam),relations-1), "reference relation ordering mismatch")
    require(np.bincount(fam,minlength=len(pairs)).tolist() == cfg["imdb_family_counts"], "reference family count mismatch")
    return types,edges,cfg["imdb_sha256"]


def expected_training_binding(initial, stage, source):
    _,_,data = case_arrays(stage,source)
    small = stage in ("A","C-small")
    return dict(initial,data=data,T=8 if small else source["config"]["T"],
                rounds=3 if small else source["config"]["rounds"])


def validate_a(directory, summary, source, initial):
    require(integer(summary["completed_updates"],"A updates") == 2, "A must complete two unique updates")
    require(integer(summary["unique_updates"],"A unique updates") == 2 and integer(summary["optimizer_calls"],"A calls") == 3, "A recovery accounting mismatch")
    binding = expected_training_binding(initial,"A",source)
    training_checkpoint(directory/"smoke_step1.pt",binding,1)
    training_checkpoint(directory/"smoke_final.pt",binding,2)
    recovery = summary["recovery"]
    require(recovery["atol"] == source["config"]["recovery_atol"] and recovery["rtol"] == 0, "recovery tolerance changed")
    for key in ("output_error","parameter_error","optimizer_error"):
        require(finite(recovery[key],key) <= recovery["atol"], "recovery error exceeds tolerance")
    require(recovery["rng_exact"] is True, "RNG recovery not recorded as exact")
    for key in ("tensor_inputs_checked","symmetric_backbone_outputs_checked"):
        require(integer(summary["device_checks"][key],key) > 0, "empty device/symmetry checks")
    return ["A checkpoints and Adam counts 1/2", "recovery tolerance/RNG receipts", "device/symmetry checks nonempty"]


def validate_b(directory, summary, source, initial):
    cfg = source["config"]
    require(integer(cfg["updates"],"configured B updates") == 100, "B protocol requires 100 updates")
    require(integer(summary["completed_updates"],"B updates") == 100, "B incomplete update count")
    ck = training_checkpoint(directory/"preflight_last.pt",expected_training_binding(initial,"B",source),100)
    require([r["position"] for r in ck["history"]] == list(range(1,101)), "B history positions are not 1..100")
    require(summary["records"] == ck["history"], "summary/checkpoint history mismatch")
    for record in ck["history"]:
        require(1 <= integer(record["time"],"training time",1) <= cfg["T"], "training time outside schedule")
        integer(record["query_slots"],"query count",1)
    baseline = read_json(directory/"baseline_checks.json")
    require(baseline["device_checks"] == summary["device_checks"], "device-check receipts mismatch")
    require([r["time"] for r in baseline["device_checks"]] == cfg["device_check_times"], "missing noise-stratified device checks")
    require(baseline["validation_before"] == summary["validation_before"], "baseline validation mismatch")
    for key in ("validation_before","validation_after"):
        require([r["time"] for r in summary[key]] == cfg["validation_times"], "missing validation time: " + key)
        for record in summary[key]:
            finite(record["weighted_ce"],key+" CE")
            integer(record["slots"],key+" slots",1)
    require(summary["data_hash_end"] == cfg["imdb_sha256"], "end-of-run input hash mismatch")
    return ["B source data and 100 committed/Adam updates", "full history matches summary", "noise checks and validation receipts"]


def verify_graph(directory, summary, source, chain):
    expected_types,reference_edges,_ = case_arrays(summary["stage"],source)
    path = directory/"engineering_final_graph.npz"
    with np.load(path,allow_pickle=False) as z:
        types,edges,family,state,T = z["types"],z["edges"],z["families"],z["state"],z["T"]
    require(np.array_equal(types,expected_types), "final graph node order/type inventory mismatch")
    require(T.shape == () and integer(T.item(),"NPZ T",1) == chain["binding"]["T"], "NPZ schedule mismatch")
    require(edges.ndim == 2 and edges.shape[1] == 2 and family.shape == (len(edges),), "final graph array shape mismatch")
    require(state.ndim == 1 and ((state == 0)|(state == 1)).all(), "invalid final binary state")
    require(np.array_equal(chain["state"],state), "final graph/chain checkpoint state mismatch")
    physical = set(map(tuple,edges.tolist()))
    require(len(physical) == len(edges) and all(0 <= u < v < len(types) for u,v in physical), "non-simple/noncanonical final graph")
    pairs = [(a,b) for a in range(int(types.max())+1) for b in range(a,int(types.max())+1)]
    require(all(0 <= f < len(pairs) for f in family), "illegal final relation ID")
    require(all(tuple(sorted((int(types[u]),int(types[v])))) == pairs[int(f)] for (u,v),f in zip(edges,family)), "final schema mismatch")
    nodes = [np.flatnonzero(types == t).tolist() for t in range(int(types.max())+1)]
    codes,offset,ranges = {},0,[]
    for a,b in pairs:
        begin = offset
        for i,u in enumerate(nodes[a]):
            for v in (nodes[b][i+1:] if a == b else nodes[b]):
                codes[tuple(sorted((u,v)))] = offset
                offset += 1
        ranges.append((begin,offset))
    require(offset == len(state) and {codes[e] for e in physical} == set(np.flatnonzero(state).tolist()), "graph/full-slot coding mismatch")
    ref_set = set(map(tuple,reference_edges.tolist()))
    for a,b in ranges:
        ref_count = sum(a <= codes[e] < b for e in ref_set)
        if ref_count == 0:
            require(not state[a:b].any(), "exact-none support violated")
        if ref_count == b-a and b > a:
            require(state[a:b].all(), "exact-one support violated")
    degree = [0]*len(types)
    relation_degree = np.zeros((len(types),len(pairs)),dtype=np.int64)
    adjacency = [set() for _ in types]
    for (u,v),f in zip(edges,family):
        degree[u] += 1
        degree[v] += 1
        relation_degree[u,f] += 1
        relation_degree[v,f] += 1
        adjacency[u].add(int(v))
        adjacency[v].add(int(u))
    remaining,components = set(range(len(types))),[]
    while remaining:
        root = remaining.pop()
        members,todo = {root},[root]
        while todo:
            for v in adjacency[todo.pop()]:
                if v in remaining:
                    remaining.remove(v)
                    members.add(v)
                    todo.append(v)
        components.append(len(members))
    expected = dict(nodes=len(types),type_counts=np.bincount(types).tolist(),edges=len(edges),
                    family_edges=np.bincount(family,minlength=len(pairs)).tolist(),components=len(components),
                    component_sizes=sorted(components,reverse=True),connected=len(components) == 1,
                    degree=degree,degree_histogram=np.bincount(degree).tolist(),relation_degree=relation_degree.tolist(),
                    schema_simple_undirected=True)
    require(summary["final_graph_metrics"] == expected, "independent graph statistics mismatch")
    require(digest(path) == summary["graph_file_sha256"], "final graph hash mismatch")
    return ["final graph/types/T match chain", "independent slot coding and support", "independent degree/component/relation statistics", "final graph hash"]


def validate_c(directory, summary, source, initial):
    stage,cfg = summary["stage"],source["config"]
    T = 8 if stage == "C-small" else cfg["T"]
    require(integer(summary["completed_reverse_steps"],"reverse steps") == T, "summary reverse steps do not complete T")
    require(integer(summary["remaining_time"],"remaining time") == 0, "passed chain has remaining time")
    require(integer(summary["graph_T"],"graph T") == T and integer(summary["engineering_chain_outputs"],"outputs") == 1, "final chain accounting mismatch")
    parent_stage = "A" if stage == "C-small" else "B"
    parent = directory.parent/parent_stage
    filename = "smoke_final.pt" if parent_stage == "A" else "preflight_last.pt"
    needed(parent,REQUIRED_FILES[parent_stage]["passed"])
    parent_summary = read_json(parent/"summary.json")
    require(parent_summary["stage"] == parent_stage and parent_summary["status"] == "passed", "source weight prerequisite not passed")
    parent_initial,_ = common_check(parent,parent_summary,source)
    require(parent_initial == initial, "source weight prerequisite binding differs")
    parent_checks = stage_check(parent,parent_summary,source,parent_initial)
    require(parent_checks, "empty prerequisite checks")
    if stage == "C-imdb":
        small = directory.parent/"C-small"
        needed(small,REQUIRED_FILES["C-small"]["passed"])
        small_summary = read_json(small/"summary.json")
        require(small_summary["stage"] == "C-small" and small_summary["status"] == "passed", "small-chain prerequisite not passed")
        small_initial,_ = common_check(small,small_summary,source)
        require(small_initial == initial, "small-chain prerequisite binding differs")
        require(stage_check(small,small_summary,source,small_initial), "empty small-chain prerequisite checks")
    expected = dict(expected_training_binding(initial,stage,source),weight_sha256=digest(parent/filename))
    chain = torch.load(directory/"chain_last.pt",map_location="cpu",weights_only=False)
    require(chain["binding"] == expected == summary["sampling_binding"], "chain/summary/source-weight binding mismatch")
    require(integer(chain["completed_reverse_steps"],"checkpoint reverse steps") == T and integer(chain["remaining_time"],"checkpoint time") == 0, "checkpoint reverse steps do not complete T")
    rng_check(chain["rng"])
    if stage == "C-imdb":
        require(T == 1000, "C-imdb requires T1000")
        cost = read_json(directory/"sweep_cost.json")
        for key in ("measured_sweep_seconds","estimated_T1000_seconds"):
            require(finite(cost[key],key) > 0 and cost[key] == summary[key], "sweep cost receipt mismatch")
    return ["complete T/remaining-time counts", "source weight hash and binding", "chain checkpoint and RNG"] + verify_graph(directory,summary,source,chain)


def stage_check(directory, summary, source, initial):
    if summary["stage"] == "A":
        return validate_a(directory,summary,source,initial)
    if summary["stage"] == "B":
        return validate_b(directory,summary,source,initial)
    return validate_c(directory,summary,source,initial)


def audit(directory, source_package=PACKAGE, expected_stage=None):
    directory = Path(directory).resolve()
    summary = read_json(directory/"summary.json")
    stage,status = summary["stage"],summary["status"]
    require(stage in STAGES and status in ("passed","incomplete","failed"), "unknown stage/status")
    inferred = directory.name if directory.name in STAGES else None
    require((expected_stage or inferred or stage) == stage, "requested/directory stage mismatch")
    needed(directory,REQUIRED_FILES[stage][status])
    evidence = dict(stage=stage,run_status=status,checks=[],required_files=list(REQUIRED_FILES[stage][status]))
    if status != "passed":
        # Partial startup can legitimately lack a run/checkpoint. Never accept it.
        if "completed_reverse_steps" in summary or "remaining_time" in summary:
            needed(directory,("chain_last.pt",))
            ck = torch.load(directory/"chain_last.pt",map_location="cpu",weights_only=False)
            done = integer(ck["completed_reverse_steps"],"partial reverse steps")
            left = integer(ck["remaining_time"],"partial time")
            require(done+left == integer(ck["binding"]["T"],"partial T",1), "partial chain count mismatch")
            require(summary["completed_reverse_steps"] == done and summary["remaining_time"] == left, "partial summary/checkpoint count mismatch")
            evidence["checks"].append("partial checkpoint counts; not accepted as complete")
        if "completed_updates" in summary:
            filename = "preflight_last.pt" if stage == "B" else "smoke_step1.pt"
            needed(directory,(filename,))
            ck = torch.load(directory/filename,map_location="cpu",weights_only=False)
            require(summary["completed_updates"] == ck["completed_updates"], "partial update count mismatch")
            evidence["checks"].append("partial update count; not accepted as complete")
        evidence.update(audit_status="not_complete",exit_code=75 if status == "incomplete" else 1)
        return evidence
    source = source_info(source_package)
    initial,common_checks = common_check(directory,summary,source)
    specific = stage_check(directory,summary,source,initial)
    require(isinstance(specific,list) and len(specific) > 0, "empty stage-specific checks: cannot accept passed")
    evidence.update(checks=common_checks+specific,audit_status="verified",exit_code=0,
                    runtime_version_binding_verified=source["config"].get("runtime_binding_schema",1) == 2,
                    source_package=str(source["package"]),limits="consistency/read-only audit; no GPU recovery replay or trajectory regeneration")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--source-package",type=Path,default=PACKAGE)
    parser.add_argument("--stage",choices=STAGES)
    args = parser.parse_args(argv)
    try:
        evidence = audit(args.directory,args.source_package,args.stage)
    except Exception as exc:
        evidence = dict(audit_status="rejected",exit_code=1,error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(evidence,indent=2))
    return evidence["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
