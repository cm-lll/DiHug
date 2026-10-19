# Anonymous joint-role constrained decoding and sampling, independent CPU v1

Definition baseline124d227752fbce651ca20a11e730b19e5de270f7 only; this package
was written independently, not extracted from another assistant's code/ZIP.
Only the public complete IMDB NPZ and its preprocessing receipt are copied.
Runtime has no dependency on previous guidance, model weights, Torch, BW,
training, a denoiser, or any full diffusion-chain runner.

## Objects and tasks

Canonical unordered slots have one binary relation per unordered coarse-type
pair. This matches the current schema restriction; overlapping relations on
the same node pair are unsupported, not silently collapsed. Node types and
inventory remain fixed. C contains all simple undirected schema-legal graphs
whose FULL R-dimensional integer degree-vector multiset in each type equals
the target. It does NOT include a connectivity requirement.

For finite logits ell, log q(x) = sum_e[x_e ell_e - log(1+exp(ell_e))].
Use stable logaddexp directly, never infer hard support from rounded sigmoids.
Actual0/1 probabilities can be supplied explicitly through a separate support
vector (-1 free,0 fixed absent,1 fixed present) or from_probabilities. Let
Cq=C intersect supp(q); empty Cq is failure/undefined conditioning, not fallback.

Task1: given one COMPLETE per-node role matching d, maximize logq over C_d.
Task2: sample over the WHOLE anonymous Cq with P_T(x)=exp(logq(x)/T)/Z_T.
T1 is original conditional q; T2 changes the target to square-root weights.
Uniform feasible uses Cq and ignores soft model scores. Deterministic small
MAP uses global enumeration of Cq. None of these is interchangeable.

The scalable decoder implements ONLY Task1. Given d, each relation has its own
disjoint edge variables and additive objective, so blocks factorize. This
decomposition would fail with connectivity, triangles or other cross-block
constraints, which are deliberately not imposed here. All relations use the
SAME joint role assignment, not independently permuted degree coordinates.

## Decoder and numerical certification

Cross-type: degree-constrained bipartite b-matching/min-cost-flow LP with
edge capacities0..1, SciPy linprog highs-ds. Multiplying rows of one partition
by−1 gives an oriented incidence matrix, totally unimodular; integer degree
right sides/bounds imply integral vertices. Independent binary/degree checks
are nevertheless mandatory. Compute a separate applicable lower bound
D(y)=b'y+sum_e min(0,c_e-(A'y)_e), valid for any y over0<=x<=1.

Same-type: binary simple-graph f-factor formulation, SciPy milp/HiGHS, not
the generally fractional degree-only LP. Each undirected variable contributes
once to EACH endpoint. A0/1 vector is independently validated against every
assigned degree, whole joint multiset and hard support before graph export.

Frozen settings are in frozen_config.json, committed before results at the
configuration-freeze commit. LP primal/dual tolerance1e-7, external integrality
tolerance1e-6, exact integer degree equality after rounding, objective bound
gap tolerance2e-7+1e-9*max(1,abs(cost)); MIP relative gap0/node cap100000.
HiGHS threads1 is explicitly forwarded via SciPy's documented installed
_highs_wrapper option (SciPy warns about pass-through options; warnings are
retained). We do not install anything. Same-type primal/dual MIP internals use
the installed backend defaults; external verification is separately stated.
Certificates use floating arithmetic, not rigorous interval arithmetic.
Solver success alone is insufficient. Timeout/limit with a verified incumbent
is feasible_unproven; uncompleted blocks mean incomplete. A graph is never
marked globally anonymous-optimal by this decoder.

Zero target degrees and mandatory edges are processed exactly. Finite logits
remain free, including ell=1000 with a numerically rounded sigmoid1. There is
no epsilon, edge repair, randomized tie breaker or support relaxation.

## Six-node truth and reported quantities

Enumerate all2^15=32768 graphs. An independent Python adjacency-set census
checks the5005 six-edge candidates (degree2 forces6 edges). Simple2-regular
components are cycles of length>=3: only C6 or C3+C3 occur. Counts are
(6-1)!/2=60 labelled cycles and binom(6,3)/2=10 triangle pairs, total70.
Exact type-preserving topology classes are obtained from exhaustive permutation
orbits, not hashes or edge overlap; constrained homogeneous support has only
TWO topologies even though it contains70 labelled graphs.

All three required probability patterns are frozen. A further6-node typed
fixture has3 nodes of each type, relations00/01/11, and full joint roles from
five engineered edges. Its separate coordinate marginals are insufficient;
an explicit wrong graph with matching marginals is retained. Heterogeneous
engineering probabilities are a frozen literal array, not model predictions.

For each of4 cases report independent, deterministic anonymous MAP, uniform
feasible, exact conditional T1 and T2. Use complete sums for exact metrics,
then4096 iid uniforms, seed2026101907 with case-index SeedSequence; uniforms
are shared across methods for descriptive pairing. Finite sums are FP64
enumeration, not Monte Carlo approximations or symbolic exact arithmetic.

Entropy H=-sum p logp (nats), effective graph count exp(H), inverse Simpson,
maximum graph mass, exact expected distinct draws sum_x[1-(1-p_x)^4096],
observed distinct draws, role pass rate, original mean logq, connectivity and
triangles are reported. Expected two-draw physical-edge symmetric difference
is2 sum_e pi_e(1-pi_e), using each distribution's edge marginals pi_e.
Also report exact type-preserving topology counts, frequencies, entropy and
observed topology classes. Labelled support size is not anonymous topology
diversity. No failed/out-of-C independent draws are discarded.

## Proofs, counterexamples and limitations

1. Expected degrees do not enforce realized roles: on4 nodes with all p=1/3,
   every expected degree is1, but only3 perfect matchings satisfy all degrees1;
   probability=3*(1/3)^2*(2/3)^4=16/243, not1.
2. Binary legal slots plus exact per-node relation equations and one shared
   complete matching imply the required joint multiset. Independent coordinate
   matchings do not imply this (explicit typed counterexample in tests/results).
3. If a reference realizes the multiset and the schema permits every pair in
   each type-pair family, any within-type complete role allocation is realized
   by a type-preserving permutation of that reference. Repeated roles cause
   multiple witness permutations, not extra graph probability. Node-specific
   model support need not survive that permutation; fixed allocation may fail
   even while some other anonymous allocation remains feasible. Tests include
   a supported3-node counterexample and two odd components with fractional LP.
4. Anonymous conditioning weights role assignments by Z_d=sum_{x in C_d}q(x).
   Uniform d followed by q(x|d) instead gives q(x)/(K Z_d). It is wrong unless
   these partitions coincide. Exact heterogeneous partition values and TV are
   computed, including duplicate-role accounting. No roles are locked by the
   exact anonymous sampler. No MCMC is implemented; no mixing claim is made.
5. On a finite fixed support, dE_T[logq]/dT=-Var_T(logq)/T^2 and
   dH_T/dT=Var_T(logq)/T^3. Higher temperature trades expected score for entropy;
   it need not improve connectivity, edge distance or every structural metric.
6. Relabelling types/schema/logits/support together bijects feasible states and
   preserves their scores/partition, so the FULL distribution is equivariant.
   A numbered representative of tied MAP states is not. Tests preserve a tied
   MAP counterexample, not a false equivariance certificate for the optimizer.

## Optional engineering scale test

Only after CPU gates: complete IMDB reference, exact schema and anonymous roles,
one fixed lexicographic allocation, PCG64 Normal(0,1) logits with frozen seed
2026101909. Score generator receives schema only, never reference edges or roles.
It can assess feasibility/runtime, NOT trained-model quality. There is a single
inclusive300-second cap including startup, preprocessing, solve, validation,
save and cleanup; no retries/extended budget. No IMDB conditional sampler or
anonymous-global MAP is attempted. Actual trained final logits are missing.
