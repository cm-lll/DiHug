"""Shared bounded-run accounting, same backend-independent probability contract."""
import copy
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time

import numpy as np
import torch

from hsgd.kernel import MarginalKernel, full_sweep
from hsgd.model import TypedSparseDenoiser
from hsgd.noising import global_noisy_edges
from hsgd.slots import TypedSlots, UniformQueryPlan

PACKAGE = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("config.json")
EXPERIMENT_ARM = "cn"


def set_arm(arm):
    global EXPERIMENT_ARM
    if arm not in ("baseline","cn"):raise ValueError("unknown frozen experiment arm")
    EXPERIMENT_ARM=arm


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def torch_save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def trusted_load(path):
    # Only our own hashed local engineering artifacts, never untrusted uploads.
    return torch.load(path, map_location="cpu", weights_only=False)


def config():
    cfg=json.loads(CONFIG.read_text())
    cfg["version"]="bw_cn_v2"
    cfg["experiment_arm"]=EXPERIMENT_ARM
    cfg["model"]["cn_hidden"]=32 if EXPERIMENT_ARM=="cn" else 0
    return cfg


def code_hashes():
    files = sorted((PACKAGE / "hsgd").rglob("*.py")) + [CONFIG]
    return {str(p.relative_to(PACKAGE)): sha(p) for p in files}


def verify_manifest():
    path = PACKAGE/"manifest_cn_v2.json"
    if not path.is_file():
        raise RuntimeError("CN package manifest missing")
    if path is not None:
        manifest = json.loads(path.read_text())
        for name, expected in manifest["files"].items():
            if sha(PACKAGE / name) != expected["sha256"]:
                raise RuntimeError("package input changed: " + name)


def environment(device):
    packages = {}
    for name in ["torch", "torch-geometric", "numpy", "scipy", "pytest", "networkx"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    r = dict(python=sys.version, python_version=platform.python_version(), executable=sys.executable, system=platform.platform(),
             machine=platform.machine(), packages=packages, hip=torch.version.hip,
             cuda=torch.version.cuda, requested_device=device,
             torch_runtime_version=torch.__version__,
             float32_matmul_precision=torch.get_float32_matmul_precision(),
             matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
             deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
             deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
             torch_threads=torch.get_num_threads(), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("accelerator unavailable; no CPU fallback")
        p = torch.cuda.get_device_properties(0)
        r.update(device_name=torch.cuda.get_device_name(0), visible_devices=torch.cuda.device_count(),
                 device_memory_bytes=p.total_memory, device_type="HIP" if torch.version.hip else "CUDA")
    else:
        r.update(device_name="CPU engineering test", device_type="CPU")
    return r


def runtime_binding(env):
    """Use actual Python and Torch builds, not only package metadata versions."""
    keys = ("packages", "device_name", "hip", "cuda", "python_version", "torch_runtime_version")
    result={key: env[key] for key in keys}
    result.update({key:env.get(key) for key in ("deterministic_algorithms","deterministic_warn_only")})
    return result


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def memory(device):
    r = dict(process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    if device == "cuda":
        r.update(peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                 peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    return r


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_save(rng, device):
    return dict(generator=copy.deepcopy(rng.bit_generator.state), python=random.getstate(),
                numpy=np.random.get_state(), torch=torch.get_rng_state(),
                accelerator=torch.cuda.get_rng_state_all() if device == "cuda" else None)


def rng_restore(saved, rng, device):
    rng.bit_generator.state = copy.deepcopy(saved["generator"])
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(saved["accelerator"])


def schedule(T, offset=.008):
    t = np.arange(T + 1, dtype=np.float64) / T
    alpha = np.cos((t + offset) / (1 + offset) * np.pi / 2)**2
    alpha /= np.cos(offset / (1 + offset) * np.pi / 2)**2
    alpha[0], alpha[-1] = 1., 0.
    return alpha


def make_case(types, edges, pairs):
    space = TypedSlots(types, pairs)
    edges = np.asarray(edges, dtype=np.int64).reshape(2, -1)
    f, loc = space.encode(edges)
    ids = space.offsets[f] + loc
    if (edges[0] >= edges[1]).any() or len(np.unique(ids)) != len(ids):
        raise ValueError("noncanonical/duplicate physical reference edge")
    counts = np.bincount(f, minlength=len(pairs))
    prior = np.divide(counts, space.counts, out=np.zeros(len(pairs), dtype=np.float64), where=space.counts > 0)
    return dict(space=space, clean=edges, clean_ids=ids, prior=prior, counts=counts,
                active=counts > 0, n_types=int(np.max(types)) + 1, n_families=len(pairs))


def toy_case():
    return make_case(np.array([0, 0, 0, 1, 1, 1]),
                     np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]),
                     ((0, 0), (0, 1), (1, 1)))


def imdb_case(path, cfg):
    if sha(path) != cfg["imdb_sha256"]:
        raise RuntimeError("IMDB data hash mismatch; no alternate/cropped data")
    with np.load(path, allow_pickle=False) as z:
        types, edges, rel = z["types"], z["edges"].T, z["relations"] - 1
    pairs = tuple(itertools.combinations_with_replacement(range(4), 2))
    c = make_case(types, edges, pairs)
    assert np.array_equal(c["space"].encode(edges)[0], rel)
    assert len(types) == cfg["imdb_nodes"] and edges.shape[1] == cfg["imdb_edges"]
    assert np.bincount(types, minlength=4).tolist() == cfg["imdb_type_counts"]
    assert c["counts"].tolist() == cfg["imdb_family_counts"]
    return c


def plan_for(case, seed, rounds):
    return UniformQueryPlan(case["space"], seed, rounds=rounds, active_families=case["active"])


def query_inventory(case, plan):
    seen = np.zeros(case["space"].total, dtype=np.uint8)
    family_by_round = []
    for r in range(plan.rounds):
        ids, f, _ = plan.block(r)
        assert not seen[ids].any()
        seen[ids] += 1
        family_by_round.append(np.bincount(f, minlength=case["n_families"]).tolist())
    for f, enabled in enumerate(case["active"]):
        values = seen[case["space"].offsets[f]:case["space"].offsets[f + 1]]
        assert (values == int(enabled)).all()
        if not enabled:
            assert case["prior"][f] == 0
    return dict(all_slots=case["space"].total, active_slots=plan.query_total,
                deterministic_none_slots=case["space"].total - plan.query_total,
                family_slot_counts=case["space"].counts.tolist(), prior=case["prior"].tolist(),
                reference_family_edges=case["counts"].tolist(), active_mask=case["active"].tolist(),
                rounds=plan.rounds, query_sizes=plan.sizes.tolist(), family_by_round=family_by_round,
                active_inclusion_probability=1 / plan.rounds, full_schema_ce_weight=plan.training_ce_weight(),
                coverage="each active slot exactly once; exact-none inactive states checked and preserved")


def make_model(case, cfg, device):
    m = TypedSparseDenoiser(case["n_types"], case["n_families"],
                           family_pairs=case["space"].pairs, prior=case["prior"], **cfg["model"]).to(device)
    assert all(p.device.type == device and p.dtype == torch.float32 for p in m.parameters())
    assert m.prior.dtype == torch.float64 and m.prior.device.type == device
    for layer in m.backbone.tf_layers:
        assert layer.linX1.out_features == cfg["model"]["ff_node"]
        assert layer.linE1.out_features == cfg["model"]["ff_edge"]
        if layer.last_layer:
            assert layer.lin_y1.out_features == cfg["model"]["ff_global"]
    return m


def dimensions(model):
    linear = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear[name] = {"in": module.in_features, "out": module.out_features}
        elif module.__class__.__name__ == "Linear" and hasattr(module, "in_channels") and hasattr(module, "out_channels"):
            linear[name] = {"in": module.in_channels, "out": module.out_channels}
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                common_neighbor_parameters=sum(p.numel() for p in model.common_neighbor.parameters()) if model.common_neighbor is not None else 0,
                linear_layers=linear,
                layer_norms={n: {"shape": list(v.normalized_shape), "eps": v.eps}
                             for n, v in model.named_modules() if isinstance(v, torch.nn.LayerNorm)},
                graph_output_head=False, mutable_node_charge_dims=0,
                dropout_config=model.backbone.dropout, dropout_active=False,
                features="v9 inputs + optional Xt-only shared first-layer contextual common-neighbor sum; no ID/spectrum/hard closure")


def train_input(case, T, rounds, seed, position, rng, force_t=None):
    start = time.perf_counter()
    t = int(rng.integers(1, T + 1)) if force_t is None else int(force_t)
    p = plan_for(case, [seed, position, 21], rounds)
    r = int(rng.integers(rounds))
    ids, f, qe = p.block(r)
    labels = np.isin(ids, case["clean_ids"]).astype(np.float32)
    preparation = time.perf_counter() - start
    start = time.perf_counter()
    ei, ef, _ = global_noisy_edges(case["space"], case["clean"], case["prior"], schedule(T)[t], rng)
    return dict(t=t, ids=ids, f=f, query=qe, labels=labels, edges=ei, edge_families=ef,
                ce_weight=p.training_ce_weight(), preparation_seconds=preparation,
                noise_seconds=time.perf_counter() - start, round_id=r)


def check_prob_device(p, family, case, device_prior=None):
    """Same checks without unused full probability D2H in training."""
    if not torch.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise RuntimeError("nonfinite/out-of-range probability")
    if device_prior is None:
        pi = torch.as_tensor(case["prior"][family], device=p.device)
    else:
        pi = device_prior[torch.as_tensor(family, dtype=torch.long, device=p.device)]
    if not (p[pi == 0] == 0).all() or not (p[pi == 1] == 1).all():
        raise RuntimeError("probability outside exact support")


def check_prob(p, family, case):
    # Sampling still needs the full CPU probabilities: preserve the v4 path.
    if not torch.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise RuntimeError("nonfinite/out-of-range probability")
    q = p.detach().cpu().numpy().astype(np.float64)
    pi = case["prior"][family]
    assert (q[pi == 0] == 0).all() and (q[pi == 1] == 1).all()
    return q


def update(model, optimizer, case, inp, T, device, do_step=True, backend="upstream", diagnostics=None, require_parameter_change=True):
    from hsgd.runtime import predict, backend_name
    backend_name(backend)
    sync(device)
    wall = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    start = time.perf_counter()
    label = torch.as_tensor(inp["labels"], device=device)
    sync(device)
    transfer = time.perf_counter() - start
    start = time.perf_counter()
    timings = {} if diagnostics is not None else None
    result = predict(model, case["space"].types, inp["edges"], inp["edge_families"],
                     inp["query"], inp["f"], inp["t"] / T, backend=backend,
                     times=timings, details=diagnostics is not None)
    if diagnostics is not None:
        logits,p = result
        diagnostics.update(logits=cpu_tree(logits),p0=cpu_tree(p))
    else:
        p = result
    sync(device)
    forward = time.perf_counter() - start
    check_prob_device(p, inp["f"], case, device_prior=model.prior)
    loss = torch.nn.functional.binary_cross_entropy(p, label, reduction="sum") * inp["ce_weight"]
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite loss")
    start = time.perf_counter()
    loss.backward()
    sync(device)
    backward = time.perf_counter() - start
    named = [(n, v) for n, v in model.named_parameters() if v.grad is not None]
    if not named or not all(torch.isfinite(v.grad).all() for _, v in named):
        raise RuntimeError("absent/all-nonfinite participating gradient")
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(v.grad) for _, v in named]))
    if not torch.isfinite(norm):
        raise RuntimeError("nonfinite aggregate gradient norm")
    if diagnostics is not None:
        diagnostics.update(loss=cpu_tree(loss),gradients={n:cpu_tree(v.grad) for n,v in model.named_parameters()})
    before = model.backbone.lin_out_E.weight.detach().clone()
    start = time.perf_counter()
    if do_step:
        optimizer.step()
    sync(device)
    optimizer_time = time.perf_counter() - start
    delta = float((model.backbone.lin_out_E.weight.detach() - before).abs().max())
    if do_step and require_parameter_change and not delta > 0:
        raise RuntimeError("expected trainable edge head did not change")
    with torch.no_grad():
        # Logging only; original BCE(sum)*(R/S) and its backward are unchanged.
        item=torch.nn.functional.binary_cross_entropy(p.detach(),label,reduction="none")
        split={}
        for label_value,label_name in ((0,"negative"),(1,"positive")):
            mask=label==label_value;count=int(mask.sum())
            split[label_name+"_count"]=count
            split[label_name+"_ce"]=float(item[mask].mean()) if count else None
    r = dict(backend=backend, actual_dispatch=backend, forward_timings=timings or {},
             loss=float(loss.detach()), time=inp["t"], query_slots=len(inp["ids"]),
             query_round=inp["round_id"], noisy_edges=inp["edges"].shape[1], gradient_norm=float(norm),
             participating_parameters=len(named), no_gradient_parameters=[n for n, v in model.named_parameters() if v.grad is None],
             no_gradient_indices=[i for i,(n,v) in enumerate(model.named_parameters()) if v.grad is None],
             edge_head_max_update=delta, preparation_seconds=inp["preparation_seconds"], noise_seconds=inp["noise_seconds"],
             label_transfer_seconds=transfer, forward_seconds=forward, backward_seconds=backward,
             optimizer_seconds=optimizer_time, synchronized_update_seconds=time.perf_counter() - wall,
             optimizer_update=do_step, **split, **memory(device))
    return r


def checkpoint(model, optimizer, rng, device, completed, binding, history):
    return dict(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                optimizer_parameter_names=[n for n,p in model.named_parameters()],
                completed_updates=completed, rng=cpu_tree(rng_save(rng, device)), binding=binding,
                history=history, purpose="engineering preflight, NOT a formal trained model")


def restore(ck, model, optimizer, rng, device, binding):
    if ck["binding"] != binding:
        raise RuntimeError("checkpoint code/data/config/device binding mismatch")
    model.load_state_dict(ck["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(copy.deepcopy(ck["optimizer"]))
    rng_restore(ck["rng"], rng, device)
    return ck["completed_updates"]


def tree_error(a, b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape == b.shape
        return float((a.double() - b.double()).abs().max()) if a.numel() else 0.
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max([tree_error(a[k], b[k]) for k in a] or [0.])
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return max([tree_error(x, y) for x, y in zip(a, b)] or [0.])
    assert a == b
    return 0.


def state_edges(case, state):
    ei, ef = [], []
    space = case["space"]
    for f in range(case["n_families"]):
        ids = np.flatnonzero(state[space.offsets[f]:space.offsets[f + 1]])
        ei.append(space.decode(f, ids))
        ef.append(np.full(len(ids), f, dtype=np.int64))
    return np.concatenate(ei, axis=1), np.concatenate(ef)


def graph_stats(case, state):
    import networkx as nx
    assert state.shape == (case["space"].total,) and ((state == 0) | (state == 1)).all()
    for f in np.flatnonzero(~case["active"]):
        assert not state[case["space"].offsets[f]:case["space"].offsets[f+1]].any()
    edges, fam = state_edges(case, state)
    space = case["space"]
    ff, ids = space.encode(edges)
    assert np.array_equal(ff, fam)
    assert len(np.unique(space.offsets[ff] + ids)) == len(ids)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(space.types)))
    graph.add_edges_from(edges.T.tolist())
    degree = np.bincount(edges.ravel(), minlength=len(space.types))
    components = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    relation_degree = np.zeros((len(space.types), case["n_families"]), dtype=np.int64)
    for f in range(case["n_families"]):
        relation_degree[:, f] = np.bincount(edges[:, fam == f].ravel(), minlength=len(space.types))
    return dict(nodes=len(space.types), type_counts=np.bincount(space.types).tolist(),
                edges=edges.shape[1], family_edges=np.bincount(fam, minlength=case["n_families"]).tolist(),
                components=len(components), component_sizes=components, connected=len(components) == 1,
                degree=degree.tolist(), degree_histogram=np.bincount(degree).tolist(),
                relation_degree=relation_degree.tolist(), schema_simple_undirected=True)
