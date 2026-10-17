# Stepwise anonymous joint-role guidance v1

Independent implementation on fixed joint CN v1.1
`dd53b6431303263e740abedf67e01d7dfc13d15b`.

Read `package/STEPWISE_ROLE_README.md` for formulas, exact support, data flow,
new sampling law, four-arm usage, bounded CPU commands and unmeasured BW scope.
Implementation: `package/stepwise_role/core.py` and `sampling.py`.
Relative additive patch: `package/joint_cn_v1_1_to_stepwise_role_v1.diff`.
Deployment ZIP and checksum: `release/`.

Original hsgd, role_degree, dataset, original39 tests remain unchanged (31
protected files). Add20 targeted cases, total59 CPU cases. A fresh extracted
ZIP also passed59 cases; see review_acceptance.json and deployed evidence.
No training updates, scientific graphs or GPU use. Actual two6000-update BW
weight imports and a complete matched IMDB Xt/p0 cost test are NOT executed.
The target-inventory measurement is not a neural/guidance throughput claim.

The adapter's formal receipt checks are metadata sanity only; they do not
certify an inaccessible checkpoint. Existing model import must independently
verify checkpoint/source/config/data before four-arm scientific use.
No full scientific generation/training driver is provided or launched.
