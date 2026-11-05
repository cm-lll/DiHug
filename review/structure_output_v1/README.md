# Output-side structure candidate (small graphs only)

Read [the frozen protocol](PROTOCOL_ZH.md) for purpose, formula, limitations, comparison and budget. This branch keeps the input model exactly as at `53f0b03c3c8d6de320b24317272abb52cfc95f7e`. It adds one loss on predicted clean-edge marginals; no new network/head, optimizer, reverse kernel or sampling correction.

Code: `experiments/structure_output_v1/role_loss.py` defines the objective; `slot_update.py` combines it with unchanged CE; `formal_worker.py` handles training and state boundaries; `formal_sampling.py` binds the unchanged generation algorithm. `tests/test_output_role_loss.py` provides independent CPU checks. `supervise_formal.py` enforces ownership and budgets; it requires local bound assets and explicit approval files, so this source package does not automatically start a run.

Environment used: Python3.9, Torch2.1.0+CUDA12.1, PyG2.3.1, NumPy1.26, one CPU thread, FP32/TF32 off, RTX2080Ti, ordinary nondeterministic GPU execution. Existing repeated-update tolerance remains failed; exact state serialization is separately checked. No dependencies installed.

CPU tests: `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_output_role_loss.py`. GPU entrypoints require `STRUCTURE_ASSETS` (local original asset root) and `STRUCTURE_RUN` (isolated output directory). The original private datasets, checkpoints and bindings are intentionally excluded. CPU result reporting uses `review/structure_output_v1/finalize_formal_cpu.py`, with CUDA disabled, and retains all metric fields and missing items.

Five CPU tests passed, including independent enumeration of1024 graphs, probability triangle expectation/gradient, within-type permutation, autograd numerical check and fixed support. Two disposable updates and two reverse steps per dataset passed integration. Auxiliary logit gradients were finite and nonzero. Second update IMDB1.023s, PubMed1.306s; peak allocated2071/2834MiB. These are engineering results, **not generation-quality evidence**. Short reverse steps are not used as whole-chain throughput: the admission estimate uses historical full chains with margin.

The primary control is the simultaneous enhanced-input CE run. Ordinary A is background only. Formal results must be checked after completion; this commit does not claim any structural benefit. No multi-graph or large-graph adaptation is provided.
