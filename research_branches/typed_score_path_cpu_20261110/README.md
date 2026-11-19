# Fixed saved-update probability path audit

No neural model calls, optimizer updates, parameter interpolation, or reverse
diffusion chains. CPU draws only from frozen saved edge probabilities.

For every one of four historical models at six fixed input times, use
`p(a)=(1-a)*p_before+a*p_after`, a in {0,.25,.5,.75,1}. Each point uses the same
32 uniform arrays as the original endpoint evaluation. Both endpoints are
reproduced bit-for-bit; intermediate graphs are new CPU diagnostic draws.
The score is frozen from `typed_score_cpu_20261110`: individual contributions
under one joint degree/local assignment, plus separately matched typed
component-size/composition distance. No new scalar weights or objectives.

For interior support, if R(a)=E[F(B)], B_e~Bernoulli(p_e(a)), then

    R'(a) = E[F(B) sum_e (p_after_e-p_before_e)
                             (B_e-p_e(a))/(p_e(a)(1-p_e(a)))].

The script estimates this with the unbiased independent leave-one-out baseline,
equivalently the sample covariance with denominator K-1. Uncertainty is a
delete-one jackknife diagnostic, not a sign guarantee or a confidence claim
against unseen rare events. A boundary with p=0 or 1 and nonzero direction is
reported unavailable rather than clipped. A boundary with zero direction can
be omitted. These are probability-coordinate derivatives, not parameter
gradients or actual AdamW directions. Common uniform finite graph draws produce
paired risk differences; differentiating those discrete sample jumps is NOT
the gradient estimator used here.

64 small graphs, 4096 independent K=2 pairs and a central finite-difference
comparison verify the formula and LOO expectation. No unbiasedness claim for
any downstream nonlinear gradient combination is made.

Run in the existing environment from repository root:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 python research_branches/typed_score_path_cpu_20261110/watch.py
```

Output directory must not exist; the worker refuses to overwrite it. The
watchdog caps this child at 240 seconds and 2 GiB RSS, one compute thread.
For completed results run `summarize.py`; this is CPU reporting only.
Scientific main execution: 56.18 s; external wall including imports 56.93 s;
peak RSS 421.77 MiB. Preparation/reporting time is separate. GPU zero.

The 24 conditions are already-seen development examples, not independent
confirmation. Neither the lowest alpha nor the lowest score is adopted for
training. The same reference has fixed node/type inventory. No statement about
full generative quality, untested graphs, or long training follows.
