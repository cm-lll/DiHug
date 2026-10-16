# Joint CN v1.1: stable CE and decision-response repair

Base is the fixed review commit `6379e992c107a1d9c2dd124d3e976d2aa2004f2e`.
The original `experiments/bw_joint_cn_v1/` and its ZIP are unchanged.
Only loss numerics and response verification are repaired; architecture,
optimizer, noise, query plan, posterior, sampling and resource guards are kept.

- [Source](package/)
- [Patch relative to joint CN v1](package/joint_cn_v1_to_v1_1.diff)
- [Exact changes and limits](package/REVIEW_FIXES.md)
- [Original design, preserved](package/DESIGN_AND_STAGED_PLAN.md)
- [New test evidence](package/evidence/)
- [New deployment ZIP](release/dihug_sparse_bw_joint_cn_v1_1.zip)
- [ZIP SHA256](release/SHA256SUMS)

Training and diagnostic CE now use one checked stable two-class-logits helper.
Training retains `reduction=sum` followed by the original `ce_weight=R/S`.
Exact prior0/1 legal labels have exactly zero loss/gradient; incompatible labels
are rejected. Normal-logit values/gradients agree with the old mathematical
objective; extreme wrong-confidence FP32 logits retain corrective gradients.

The old two-logit response remains an interface metric. New metrics use the
present-minus-absent logit gap and actual supported p0, including CN×global
four-corner mixed responses. Shared-shift counterexamples expose the original
metric's ambiguity. Random-weight responses show capability, not learned
coordination or generated-graph quality.

Original27 test cases remain byte-identical;12 focused CPU cases were added.
No BW, optimizer updates/formal training or scientific full chains are run for
this repair. Original tiny engineering fixtures remain part of the requested
regressions. No weights, credentials or notebook login details are included.

CPU regression command after upload/extraction:

```bash
cd /root/private_data/gnn_projects/dihug_sparse_bw_joint_cn_v1_1
sha256sum -c SHA256SUMS
bash run.sh test --output /root/private_data/gnn_projects/bw_joint_cn_v1_1/test01
```

No BW or training command is issued by this review.
