"""Deterministic equivalence checks. Tolerances frozen before execution.

Integer graph inputs: exact; no sampling or trajectory equality inferred from
similar edge counts. Forward hooks only obtain the old path's actual logits.
"""
import copy
import pickle
import numpy as np
import torch
from hsgd.computational import FrozenTypedContext, legacy_arrays, legacy_transfer
from hsgd.bw_preflight.common import rng_save

TOLERANCES = {
    "logits": {"atol": 5e-6, "rtol": 1e-5},
    "p0": {"atol": 5e-7, "rtol": 1e-5},
    "reverse": {"atol": 2e-6, "rtol": 1e-5},
    "loss": {"atol": 2e-6, "rtol": 1e-5},
    "gradient": {"atol": 2e-5, "rtol": 1e-4},
    "updated_parameter": {"atol": 5e-6, "rtol": 0.0},
    "renumber_p0": {"atol": 2e-6, "rtol": 1e-5},
}


def compare_float(old, new, kind):
    a, b = torch.as_tensor(old).detach().cpu(), torch.as_tensor(new).detach().cpu()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError(kind + " shape/nonfinite")
    error = float((a.double()-b.double()).abs().max()) if a.numel() else 0.
    if not torch.allclose(a, b, **TOLERANCES[kind]):
        raise AssertionError(f"{kind}: error {error}, frozen tolerance {TOLERANCES[kind]}")
    return error


def exact_graph(old, new):
    for name in ("edges", "states", "families", "query_mask", "query_index", "query_families", "attrs"):
        if not torch.equal(getattr(old, name), getattr(new, name)):
            raise AssertionError("computational graph mismatch: " + name)
    return True


def old_details(model, types, edges, families, query, qf, time, graph):
    captured = []
    hook = model.backbone.register_forward_hook(lambda module, args, result: captured.append(result.edge_attr))
    try:
        p0 = model(types, edges, families, query, qf, time)
    finally:
        hook.remove()
    if not captured:
        return torch.empty((0,2), device=p0.device), p0
    pred = captured[0]
    logits = torch.stack([pred[graph.query_index,0], pred[graph.query_index,graph.query_families+1]], dim=-1)
    return logits, p0


def compare_forward(model, types, edges, families, query, qf, time):
    token = ("fixed-engineering-context", 0)
    ctx = FrozenTypedContext(model, types, edges, families, token)
    old = legacy_transfer(ctx, legacy_arrays(ctx, query, qf, token))
    new = ctx.assemble(ctx.transfer_query(ctx.prepare_query(query, qf, token), token), token)
    exact_graph(old, new)
    old_logits, old_p = old_details(model, types, edges, families, query, qf, time, old)
    logits, p = ctx.run_network(new, time, token)
    record = dict(graph_exact=True, logits_error=compare_float(old_logits, logits, "logits"),
                  p0_error=compare_float(old_p, p, "p0"))
    ctx.close()
    return record


def compare_update(model, case, inp, T, optimizer_cfg, optimizer_state=None):
    # Independent disposable clones. Original imported weight is NEVER updated.
    old, new = copy.deepcopy(model), copy.deepcopy(model)
    old.train(); new.train()
    device = next(model.parameters()).device
    label = torch.as_tensor(inp["labels"], device=device)
    args = (case["space"].types, inp["edges"], inp["edge_families"], inp["query"], inp["f"], inp["t"]/T)
    opt_old = torch.optim.AdamW(old.parameters(), **optimizer_cfg)
    opt_new = torch.optim.AdamW(new.parameters(), **optimizer_cfg)
    if optimizer_state is not None:
        opt_old.load_state_dict(copy.deepcopy(optimizer_state))
        opt_new.load_state_dict(copy.deepcopy(optimizer_state))
    head_before = old.backbone.lin_out_E.weight.detach().clone()
    p_old = old(*args)
    context = FrozenTypedContext(new, *args[:3], state_version=0)
    p_new = context.predict(*args[3:], state_version=0)
    a = torch.nn.functional.binary_cross_entropy(p_old, label, reduction="sum")*inp["ce_weight"]
    b = torch.nn.functional.binary_cross_entropy(p_new, label, reduction="sum")*inp["ce_weight"]
    loss_error = compare_float(a,b,"loss")
    a.backward(); b.backward()
    max_grad = 0.
    participating, absent = [], []
    for (name,x),(other,y) in zip(old.named_parameters(),new.named_parameters()):
        assert name == other and (x.grad is None) == (y.grad is None)
        if x.grad is None:
            absent.append(name)
        else:
            participating.append(name)
            max_grad = max(max_grad,compare_float(x.grad,y.grad,"gradient"))
    assert participating
    opt_old.step(); opt_new.step()
    head_delta = float((old.backbone.lin_out_E.weight.detach()-head_before).abs().max())
    if not head_delta > 0:
        raise AssertionError("disposable optimizer did not update expected edge head")
    max_parameter = max(compare_float(x,y,"updated_parameter") for x,y in zip(old.parameters(),new.parameters()))
    context.close()
    return dict(loss_error=loss_error, gradient_error=max_grad, updated_parameter_error=max_parameter,
                participating_parameters=len(participating), no_gradient_parameters=absent,
                optimizer_calls=2, original_weight_updated=False, expected_head_delta=head_delta,
                loaded_read_only_optimizer_state=optimizer_state is not None)


def compare_rng(before, after):
    assert pickle.dumps(before["generator"]) == pickle.dumps(after["generator"])
    assert pickle.dumps(before["python"]) == pickle.dumps(after["python"])
    assert pickle.dumps(before["numpy"]) == pickle.dumps(after["numpy"])
    assert torch.equal(before["torch"],after["torch"])
    if before["accelerator"] is not None:
        assert all(torch.equal(a,b) for a,b in zip(before["accelerator"],after["accelerator"]))


def threshold_differences(ids, families, old_q, new_q, uniforms):
    different = (uniforms < old_q) != (uniforms < new_q)
    return [dict(slot_id=int(ids[k]), family=int(families[k]), uniform=float(uniforms[k]),
                 old_q=float(old_q[k]), new_q=float(new_q[k]),
                 distance_old=float(abs(uniforms[k]-old_q[k])),
                 distance_new=float(abs(uniforms[k]-new_q[k]))) for k in np.flatnonzero(different)]
