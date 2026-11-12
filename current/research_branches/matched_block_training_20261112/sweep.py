"""Full-support query sweeps: fixed Xt and parameters until every block finishes.

The terminal-feedback objective sums slot log probabilities. Splitting that sum
does NOT introduce a 1/R average or R importance multiplier. CE validation uses
the original whole-schema denominator. Each block rebuilds its differentiable
context, avoiding reuse of a freed CN autograd graph across backward calls.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research_branches/one_step_vs_terminal_plan_20261111'))
import update_preflight as s
w, np, torch = s.w, s.np, s.torch

def blocks(model, case, edges, families, t, rounds, seed, check=lambda: None):
    plan = s.RecordedPlan(w.common.plan_for(case, [int(seed), int(t), 61], rounds))
    coverage = np.zeros(case['space'].total, dtype=np.uint8)
    state_hash = s.digest(edges), s.digest(families)
    versions = [(p, p._version) for p in model.parameters()]
    for i in range(rounds):
        check()
        assert all(p._version == v for p, v in versions), 'parameters changed inside sweep'
        assert state_hash == (s.digest(edges), s.digest(families)), 'Xt changed inside sweep'
        ids, fam, query = plan.block(i)
        assert not coverage[ids].any(), 'duplicate active slot'
        coverage[ids] += 1
        if not len(ids):
            continue
        logits, prob = w.runtime.predict(model, case['space'].types, edges, families,
            query, fam, t/1000, details=True)
        w.common.check_prob_device(prob, fam, case, model.prior)
        yield ids, fam, logits, prob, plan
        del logits, prob
    expected = np.repeat(case['active'].astype(np.uint8), case['space'].counts)
    assert np.array_equal(coverage, expected), 'incomplete active support'
    assert plan.blocks == rounds
    assert all(p._version == v for p, v in versions)

def feedback(model, case, edges, families, t, rounds, seed, target_full, coefficient,
             expected_query_hash, check=lambda: None):
    total = 0.; count = 0; final_plan = None
    for ids, fam, logits, prob, plan in blocks(model, case, edges, families, t, rounds, seed, check):
        target = torch.as_tensor(target_full[ids], device=logits.device).bool()
        lp = torch.log_softmax(logits.double(), dim=-1)
        loss = coefficient*torch.where(target, lp[:, 1], lp[:, 0]).sum()
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite feedback loss')
        loss.backward()
        total += float(loss.detach()); count += 1; final_plan = plan
        del loss, lp, logits, prob
    assert final_plan is not None and final_plan.h.hexdigest() == expected_query_hash
    return dict(loss=total, block_backwards=count, query_hash=expected_query_hash,
                exact_coverage=True, slots=int(np.repeat(case['active'],case['space'].counts).sum()))

def predict_all(model, case, edges, families, t, rounds, seed, check=lambda: None):
    # Used for read-only CE validation. This small output buffer is not a learned
    # all-pair head, and does not retain block activations.
    device = next(model.parameters()).device
    all_logits = torch.zeros((case['space'].total,2), device=device)
    all_prob = torch.zeros(case['space'].total, device=device)
    plan = None
    with torch.no_grad():
        for ids, fam, logits, prob, plan in blocks(model,case,edges,families,t,rounds,seed,check):
            index = torch.as_tensor(ids, device=device)
            all_logits[index] = logits; all_prob[index] = prob
    return all_logits, all_prob, plan.h.hexdigest()
