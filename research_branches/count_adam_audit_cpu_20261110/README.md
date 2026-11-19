# Saved count-risk and AdamW audit

This reads four saved pre/post boundaries and their original K=16 hard draws.
It instantiates no model or optimizer, calls no forward/backward/step, and
draws no graphs. All tensor operations are CPU saved-array algebra.

## Exact count risk (same HARD objective, not a soft surrogate)

For relation r let independent B_e ~ Bernoulli(p_e), N_r=sum B_e, reference
count m_r, d_r=max(1,m_r), and R be the original number of relations, including
fixed-zero ones. The old objective is F=mean_r (N_r-m_r)^2/d_r^2.
With mu_r=sum p_e and v_r=sum p_e(1-p_e), its exact expectation is

    E F = sum_r [(mu_r-m_r)^2 + v_r] / (R d_r^2).

For a binary logit margin z_e belonging to r:

    d E F / d z_e = p_e(1-p_e)[2(mu_r-m_r)+1-2p_e] / (R d_r^2).

The variance term v_r MUST remain. Using only a squared difference of expected
counts changes the objective and is not this audit. Exact evaluation is O(M)
in legal slots and requires no graph enumeration on real data. Fixed support
probabilities 0/1 have zero margin derivative.

The risk for one relation has exactly zero margin derivative on other
relations' independent Bernoulli variables. Finite K=16 global score-function
estimates nevertheless include such cross-relation sampling covariance.
They are unbiased over resampling, not necessarily accurate in a particular
batch. This audit reports total and own/off-relation terms, not a new estimator
selection search. Sixty-four small graphs verify exact risk and gradient.

## Saved AdamW identity

Parameter IDs follow the source optimizer construction `AdamW(m.parameters())`.
The saved EMA names match ordered model keys excluding the sole prior buffer;
all parameter and moment shapes match. There are 474 parameter tensors, 472
with optimizer states, and two unused unchanged tensors. No model is built.

Let m+,v+ be saved new moments, and m-,v- old moments. For each used parameter:

    inferred_g = (m+ - beta1 m-)/(1-beta1)
    denominator = sqrt(v+/(1-beta2**step)) + eps
    historical = -lr beta1 m- / ((1-beta1**step) denominator)
    current = -lr (m+ - beta1 m-) / ((1-beta1**step) denominator)
    decay = -lr weight_decay theta-

Their sum is compared with the actual saved theta+ - theta-. The inferred
gradient is approximate because the moments were saved in FP32. New v is
checked against beta2 v-+(1-beta2)inferred_g**2. This is algebraic consistency,
NOT update replay. No arrays are loaded into a model or submitted to AdamW.

Current/historical terms share the ACTUAL new denominator. They are additive
accounting terms, not independent causal experiments. The optional old-denominator
column freezes the actual new numerator and substitutes old bias-corrected v;
it is a numerical sensitivity calculation, never an implemented alternative.

## Reproduction

Initial isolated run, output directory must not already exist:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 python research_branches/count_adam_audit_cpu_20261110/watch.py
```

To preserve completed output when independently reproducing, import `run.py`
as a module, assign its `OUT` to a new isolated directory, then call `main()`.
The fixed protocol/tests remain the same. No files under the old source or
checkpoints are modified. `protocol.json` binds all inputs and source hashes.

Main measured 12.55 s, external supervisor wall 13.06 s, peak 591.35 MiB,
one CPU thread, GPU zero. Source inspection/preparation/reporting is separate.

## Next decision, not automatically executed

Candidate: replace ONLY this count term's K=16 Monte Carlo margin gradient
with its exact gradient, keep the same hard objective and other terms. A
four-boundary temporary one-update comparison is necessary to check the neural
Jacobian, combination and actual AdamW behavior. Existing evidence does NOT
show exact count feedback fixes graph quality, diversity, connectivity, or
other terms' estimator noise. No such update has been run in this ticket.
