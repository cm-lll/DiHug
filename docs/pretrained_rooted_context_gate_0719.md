# Pretrained Rooted-Context Gate (2026-07-19)

This experiment asks whether the role-conditioned HGT representation from the
latent-alignment gate improves the existing optional rooted-context residual.
It does not introduce a second context scorer. The existing
`RootedContextEditResidual` remains a frozen legacy MLP plus a candidate-
conditioned endpoint-context residual.

## Controlled variants

All variants use the same IMDB source graph, legacy MLP checkpoint, candidate
count, seed, and held-out trajectory protocol.

- `A legacy random`: the frozen paper implementation, with a randomly
  initialized context encoder trained on switch-chain states.
- `B pretrained legacy`: Phase-2 `node_input`, HGT layers, and role input are
  loaded; the encoder is frozen while the candidate residual is trained on the
  same switch-chain states as A.
- `C pretrained closed-form`: B, but training states come directly from typed
  closed-form marginal corruption at noise fractions 0.25/0.50/0.75/1.00 and
  candidates include relocation and switch.
- `D pretrained joint`: B with the pretrained encoder jointly fine-tuned.
- `E pretrained closed-form joint`: C with the encoder jointly fine-tuned.

The legacy MLP is frozen in every variant. The residual output is zero-
initialized, so each run starts as the exact legacy scorer.

## IMDB shallow gate

| Variant | Base utility | Context utility | Relative gain | Selected scale |
|---|---:|---:|---:|---:|
| A legacy random | `0.001567` | `0.001904` | `+21.6%` | `0.75` |
| B pretrained legacy, frozen | `0.001567` | `0.001830` | `+16.8%` | `0.75` |
| C pretrained closed-form, frozen | `0.00002068` | `0.00002140` | `+3.5%` | `1.00` |
| D pretrained legacy, joint | `0.001567` | `0.001601` | `+2.2%` | `0.50` |
| E pretrained closed-form, joint | `0.00002068` | `0.00002140` | `+3.5%` | `0.50` |

## Decision

The pretrained representation does not improve the existing rooted-context
residual. Freezing it remains below the original context training, and joint
fine-tuning reduces the gain further. Closed-form high-noise states provide a
much weaker rooted-closure ranking signal under the current free candidate
budget.

The result rejects a training-protocol explanation for the earlier IMDB
context regression: Phase-2 clean/prototype geometry is informative as a graph
representation, but it is not aligned with the local rooted-closure edit
ranking objective. No four-dataset or free-sampling run is promoted from this
gate. The frozen paper protocol remains unchanged.
