# Latent Edit Representation Gate (2026-07-19)

This note records intermediate evidence only. It does not claim that the new
representation improves free generation until the response-aware edit gate and
end-to-end sampling gate pass.

## Protocol

- Datasets: IMDB, PubMed, DBLP, HGEN Syn-200.
- Forward state: closed-form typed marginal corruption at noise fractions
  `0.25, 0.50, 0.75, 1.00`.
- Validation trajectory is held out from representation training.
- Edge AP is compared with the current-edge indicator, not only class
  prevalence.
- Role reconstruction is compared with the optimal type/family median under
  MAE; this prevents a constant predictor from passing.
- Anonymous roles contain no original node IDs and are fixed for one
  trajectory.

## Phase 1: Unconditioned HGT Warm-up

Edge AP gain over the current-edge baseline at noise fractions
`0.25 / 0.50 / 0.75`:

| Dataset | Edge AP gain | Role result |
|---|---:|---|
| IMDB | `+0.066 / +0.098 / +0.101` | below type-median baseline |
| PubMed | `+0.133 / +0.277 / +0.277` | above baseline at 0.25 and 0.50 |
| DBLP | `+0.091 / +0.110 / +0.115` | below type-median baseline |
| Syn-200 | `+0.054 / +0.067 / +0.030` | below type-median baseline |

Interpretation: graph encoding carries clean-edge information on every dataset,
but inferring the sampled anonymous role from a noisy graph is not a valid
universal objective. The role is therefore used as a sampled condition in the
next phase rather than reconstructed from top noise.

## Phase 2: Fixed Role Condition and Clean Teacher

The model adds a fixed role condition, an EMA clean-graph teacher, and three
conditional prototypes per node. Removing graph message passing while retaining
the same node features and role condition gives the following full-context gain
at `0.25 / 0.50 / 0.75`:

| Dataset | Edge AP context gain | Latent clean-cosine context gain |
|---|---:|---:|
| IMDB | `+0.001 / +0.024 / +0.005` | `+0.142 / +0.155 / +0.170` |
| PubMed | `+0.007 / +0.009 / +0.011` | `+0.173 / +0.211 / +0.249` |
| DBLP | `+0.041 / +0.055 / -0.003` | `+0.047 / +0.039 / +0.038` |
| Syn-200 | `+0.022 / +0.018 / -0.002` | `+0.102 / +0.126 / +0.148` |

The role-shuffle cosine distance is positive on every dataset (`0.17-0.23`), so
the encoder responds to the sampled condition. The direct edge head remains
mostly role-driven on IMDB and PubMed, while graph context consistently improves
the latent geometry on all four datasets.

## Decision

Proceed to the local response gate, not directly to long sampling. During
training, one legal edit per state may be fully re-encoded to supervise a local
response predictor. At inference, candidate edits must use the learned local
response and periodic full refresh; no per-candidate full-graph encoding is
allowed. The next hard criterion is paired held-out edit utility versus the old
typed-closure MLP, followed by four-dataset free sampling only if utility
improves.
