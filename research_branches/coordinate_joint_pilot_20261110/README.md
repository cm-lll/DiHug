# Two-coordinate CPU probability pilot

This is a local feasibility diagnostic on saved conditional clean predictions, NOT a neural optimizer implementation. It uses the 84 already fixed parent/changed probability pairs (four models, seven inputs, three saved displacement positions). It does not recompute the neural output, alter AdamW/EMA, or generate a reverse chain.

## Fixed rule

Let z0=logit(p0), d=logit(p1)-logit(p0) on strictly interior supported slots, w=p0(1-p0). For each relation f:

```
c_f = sum_{e in f} w_e*d_e / sum_{e in f} w_e
density_e = c_f
arrangement_e = d_e - c_f
```

Fixed 0/1 slots remain unchanged. This decomposition obeys sum(w*arrangement)=0 within each relation and density/arrangement are orthogonal under the local Bernoulli Fisher metric. Therefore the arrangement component does not change expected relation counts to FIRST ORDER. Finite changes can change them; it is not a hard quota or an exact finite count-preserving retraction. A common density shift preserves conditional relative log-odds arrangements.

With J=(density, arrangement), W_k=sum(w*J_k^2), and G_j the original five risk gradients with respect to logits, C_jk=G_j dot J_k. Solve:

```
min_s 1/2 * sum_k W_k*(s_k-1)^2
subject to C*s <= 0, 0 <= s_k <= 1.
```

The box prohibits expansion and reversal; these are deliberate engineering restrictions. It is not a theorem that this two-dimensional subspace contains a useful direction. Zero is feasible; zero or near-zero means progress may be lost, not successful learning. The transformed 1/2-dimensional Euclidean projection enumerates candidate faces/vertices, handles zero direction norms without ridge, verifies feasibility, and is independently compared to SLSQP on 30 fixed synthetic cases before real data.

Write p'=sigmoid(z0+J*s). There is no finite-risk accept/reject, backtracking, or parameter sweep. The inequalities only concern estimated first-order changes; nonlinear effects and estimation error can violate them in actual risks.

The same-scale control is sigmoid(z0+beta*d), where
beta=sqrt(sum(W*s^2)/sum(W)). It matches the baseline-Fisher LOGIT displacement norm, not probability L2, network parameter L2, or exact KL. The Fisher metric approximates twice local Bernoulli KL; finite steps do not share exact KL merely because this norm matches.

## Original objectives and evaluation

All five original risks remain: relation counts, anonymous joint relation degrees, reference-present typed triangles, reference-absent feasible typed triangles, component count. Three gradients/risks use existing exact Bernoulli moments, including variance/shared-edge covariance. Joint-degree/component direction estimates use LOO on the first 128 of the old frozen 256 independent uniforms. Evaluation uses the last 128 uniforms, paired across arms. Samples used for the direction and evaluation are disjoint, but these are already viewed development inputs, not a new independent confirmation set.

No structure target is removed, no loss weight is tuned. Baseline probabilities and hard costs are reused; only candidate/matched-control CPU conditional hard draws are scored, deduplicating exact-identical arms. Means and paired MC standard errors versus unchanged are retained. Eighty-four conditional cases are not independent training repetitions. Each input receives its own hypothetical correction; a shared neural network may not realize all corrections simultaneously.

Interpretation requires comparison to original, unchanged, and same-norm uniform scaling. A lower mean hiding triangle/hub/relationship costs is insufficient, and a zero update cannot be presented as an effective repair. No connectivity claim follows from triangle statistics alone; the component risk is explicitly evaluated. This ticket does not choose or train a final model.

## Reproduction

```sh
timeout 600s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python research_branches/coordinate_joint_pilot_20261110/run.py
```

Run refuses to overwrite output. CPU single-thread, science stop540s/process limit600s. All sources and reused probabilities/costs are bound by SHA256 before results. Separate directories preserve old assets. No GPU access or automatic next stage.
