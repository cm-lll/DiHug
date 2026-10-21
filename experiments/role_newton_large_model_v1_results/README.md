# Complete large-model Newton CPU resource probes

Read RESULTS.md first: random initialization,engineering only;not model quality,training or a complete chain.
Source/weights/data/roles/config/environment and all six fixed forward states bound in evidence/freeze.json and dataset bindings.
All available complete cold/repeat paths independently audited;incomplete and unrun cases remain explicitly labeled.
Every read has L/P/logP/capacity/support/tail/context certificates;complete full-slot p0/q/state/coverage retained.
Cold/repeat outputs are genuinely recomputed;only identical-time degree-law tables reused.

To avoid transmitting identical cold/repeat bytes twice,the ZIP stores content-addressed objects and logical_manifest.json.
This changes storage only,not numeric arrays or scientific settings. Every logical path is restored and SHA256-checked:
```
unzip -p role_newton_large_model_v1_review.zip materialize.py > materialize_review.py
python materialize_review.py role_newton_large_model_v1_review.zip new_review_directory
```
New directory required,no overwrite;does not call a model. Restored source and results match review_bundle/SHA256SUMS.
Independent restored-bundle same43 tests and CLI --help are checked;unpacked receipt stored outside immutable ZIP.
No PT weights,credentials,notebook login or remote-access tools included. No automatic installation.
Do not relaunch run.py:exclusive launch and original cumulative60min ledger are one-use;no budget reset is authorized.
