# Frozen-policy terminal structure feasibility test

This is a diagnostic adapter, **not a training entry point**. It never creates an optimizer or changes the frozen EMA parameters. It uses existing A late-CN checkpoints; it is not an O/A quality comparison and is not a new architecture.

Derivatives are with respect to the frozen EMA readout treated as the policy parameters. They are not gradients through historical EMA updates or the original raw training trajectory. Any future optimization starting from this readout needs its own explicit initialization/optimizer contract. These trajectories cannot be silently reused as on-policy data after the policy changes.

`prepare.py` freezes source/weight/data hashes, seven costs, four trajectory seeds and eight independently selected times per trajectory. `supervise.py` checks that GPU0/1 are idle, starts two isolated workers, and independently enforces the fixed release time and total GPU occupancy. The frozen absolute deadline cannot be restarted by rerunning a script. Original source and checkpoints are read-only.

`worker.py` uses production `reverse_step`, query plans, uniforms, kernel and transactional state boundary. At selected times only, its `StepPredictor` adapter retains the autograd graph of the **same probability tensor consumed by sampling**. It does not substitute a later recomputation for the actual behavior probability. In eval mode the inspected model path has no gradient-enabled-dependent branch; dropout calls in the attention implementation are commented out. Numerical equivalence to an independently rerun full path is not asserted. Historical update-repeat tolerances remain unchanged and unresolved.

For a fixed policy, reference and parameter-independent initial prior, the terminal-risk score identity is

`grad E[C(G0)] = E[C(G0) sum_t grad log P_theta(G[t-1] | G[t], query_plan[t])]`.

Hard graph states are detached observations; differentiating them would be a different estimator. Query plans are independent of model parameters and retain their production conditioning. Each transition log probability is the sum over unordered legal slots, not a per-edge mean. Fixed-support coordinates contribute zero log probability and zero gradient. There is no epsilon floor, reward clipping, reward standardization, alternate sampler or reference-dependent sampling decision.

Eight distinct times are sampled uniformly without replacement from 1..1000 independently of the path, so their score-gradient sum is multiplied by 1000/8. Different graph paths are independently seeded. Each terminal cost uses the other three paths' mean as its leave-one-out baseline. This baseline is independent of that path and preserves the expected gradient; the four resulting contributions share a baseline and are not independent variance replicates. Two disjoint two-path estimates are reported as a descriptive noise check, not a significance test or proof of optimizer progress.

For a two-path batch the LOO estimate is exactly `(C1-C2)*(S1-S2)/2`. Thus the seven costs share the same two-pair geometric cosine up to signs. Seven positive/negative cosine entries are **not seven independent repetitions**. The reported contribution RMS dispersion is descriptive, not a standard error or an estimate of population signal-to-noise ratio. LOO and uncentered estimates may disagree on a small batch without changing their expected gradient; this differs from centering a biased eager clean-head surrogate.

Seven terminal costs reuse the existing CPU definitions: typed joint relation-degree assignment distance; relation marginal degree W1; low-degree discrepancy; top-two hub discrepancy; typed node-triangle distribution discrepancy; global typed-triangle L1 error; and typed component-composition assignment distance. Reference-derived scales remain fixed. **These are reported separately.** The old composite `old_F` is retained only as a historical descriptive number; no new training scalarization or lambda has been adopted. The evaluation's joint-degree TV is a different metric from the assignment-distance cost, and both are retained.

`evaluate.py` uses the existing complete evaluator for original graphs and derived LCCs, keeping nulls, per-type/relationship results, motifs, closure, MP1/2/3, tail/hub, overlap and legacy HGEN metrics. LCC connectivity is not credited to the model; retention and the original-reference comparison are explicit. It also computes score-gradient diagnostics without invoking any model. `audit_saved.py` regenerates query plans and uniform arrays on CPU, checks every recorded plan/draw and selected transition, verifies final graph identity, and rechecks frozen source/asset hashes.

Example commands (the actual run's protocol contains absolute deadlines):

```bash
python research_branches/terminal_structure_real_20261109/prepare.py
python research_branches/terminal_structure_real_20261109/supervise.py results/terminal_structure_real_20261109T233600
python research_branches/terminal_structure_real_20261109/evaluate.py results/terminal_structure_real_20261109T233600
python research_branches/terminal_structure_real_20261109/audit_saved.py results/terminal_structure_real_20261109T233600
```

Large local gradient tensors and checkpoints are not intended for a public source review package. This panel cannot establish generated-quality improvement (zero updates), cross-training-seed stability (one model per dataset), or convergence of a proposed training method. No automatic successor experiment is authorized.
