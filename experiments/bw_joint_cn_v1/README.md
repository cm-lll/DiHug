# BW joint local/global CN v1 — isolated source review

This directory publishes the already delivered package for review only. No
training, denoiser inference, sampling or BW execution was performed for upload.
The old CN v1/v2 directories and `main` are not modified.

- Source implementation commit: `74ec27efee0bbbd6f87b430ac5b6b1c1c21a9ee1`.
- Published review base: `a821aa13cc70ffd590c2003381eb4d9e7fab991f` (CN v2).
- [Complete package source](package/).
- [CN v2 → joint v1 patch](package/cn_v2_to_joint_v1.diff).
- [Design, dimensions, four arms and unissued staged plan](package/DESIGN_AND_STAGED_PLAN.md).
- [CPU mechanism and local test evidence](package/evidence/).
- [Previously completed deployed-package acceptance](acceptance/).
- [Notebook commands](package/说明.txt).
- [Exact deployment ZIP](release/dihug_sparse_bw_joint_cn_v1.zip).

The ZIP is unchanged, 117790 bytes. SHA256:

```text
9c46a081d53d820f062f669a50d489246a80859029ab4de04fc884c5a597a651
```

For four-arm review, inspect `package/hsgd/model.py`,
`package/hsgd/computational.py`, `package/role_degree/role_sampling.py`, and
`package/tests/test_fusion.py`. A/C share a newly trained matched no-CN model;
B/D share a newly trained CN model. Role guidance remains terminal-only and is
not a training input. The shared nonlinear head reads CN features before both
logits; there is no independent CN logit addition. No formal training/full-chain
execution entry is included here.

For resource guards, inspect `package/hsgd/common_neighbor.py`,
`package/run.py`, and `package/hsgd/fusion_preflight.py`: pre-expansion exact
workload limits, explicit bounded BW command and process timeout. CPU evidence
does not certify BW performance or full-chain quality.

Historical acceptance JSONs are preserved as delivered: their
`external_upload_attempted: false` fields describe acceptance before this
publication. `upload_provenance.json` records this separately authorized upload.
No private weights, credentials, notebook login information, or unrelated
role-degree probability caches are included. The original small IMDB reference
and its preprocessing provenance remain inside the exact authorized ZIP.
