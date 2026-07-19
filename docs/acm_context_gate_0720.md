# ACM Rooted-Context Gate

## Scope

This is a causal gate for deciding whether the optional rooted-context residual
should enter the ACM generation protocol. It is not a final ACM generation
result. The legacy edit scorer is trained from scratch on ACM, then frozen; the
context residual is trained and evaluated on disjoint switch trajectories.

The first attempted gate used closed-form marginal states with role-alignment
candidates. That protocol was invalid for this question because the context
residual is used only during degree-preserving switch refinement. Under the
correct protocol, training states preserve sampled role degrees, are corrupted
by typed switches, and are evaluated with closure-oriented switch candidates.

## Corrected gate

- Dataset: 10,942 nodes, 273,903 edges, four active relation families.
- Legacy pilot: the formal `rs` kernel (relocation + typed 2-switch), two
  trajectories, 500 forward edits, 100 recorded examples, and three epochs.
  This is intentionally shallow and is not the final ACM model.
- Context gate: four trajectories, eight states per trajectory, 16 candidates,
  256 random switches between states, 20 epochs.
- Split: 24 training states and eight held-out states by trajectory.
- Candidate signal: 93.75% nonzero utilities, 42.77% positive utilities, and a
  positive candidate in every state.

Held-out results:

| scorer | selected rooted-closure utility |
|---|---:|
| frozen legacy MLP | 9.380e-6 |
| legacy + rooted context | 9.960e-6 |
| oracle candidate | 9.960e-6 |

The legacy MLP already captures about 94.2% of the candidate-set oracle utility.
The context residual raises this to 99.9997%, a 6.18% relative utility gain,
and selects residual scale 0.25. Therefore ACM passes the context
decision-quality gate. A paired target-free generation run from identical
Binomial initial graphs is still required before reporting final ACM metrics.

Training-sample construction remains CPU work in this gate. The formal
generation hot loop uses the resident CUDA graph state and canonical-ordered
candidate backend; optional HGT context and MLP scoring can be batched across
independent chains.
