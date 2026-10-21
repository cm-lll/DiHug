# DP versus Newton: paired full generation

Scientific execution source e6f0cb15 (receipt fix after prechecks), original solver unchanged from4061ccbb.
One45-minute ledger began at1792562319; preparation and engineering failures count. No restart/reset.
Only3 broad coverage final6000 models;88 fresh chains per arm/seed,528 total. CPU only, no training.
11 targets and eight seeds are frozen in results/freeze.json. No old generated graph substitutes a new run.
Every Newton read independently verifies original capacity tolerance, KKT where support is full,
logP/P/tail support and learned-context omission bound; failures stop without fallback.
The six new processes call the original chain, model, query coverage, posterior and uniform draw functions.

Pure CPU regression, no model fitting or chain:
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTEST_ADDOPTS= \
 python -m pytest -q experiments/role_newton_generation_v1/test_integration.py \
 experiments/role_newton_reader_v1/test_newton.py experiments/role_newton_reader_v1/test_receipt.py \
 experiments/role_newton_reader_v1/test_imports.py experiments/role_logscale_reader_v1/test_scaling.py \
 experiments/role_logscale_reader_v1/test_additional.py experiments/role_logscale_reader_v1/test_numeric_guard.py
```
Expected39 tests, zero skips. run.py is a one-use execution entry bound to the original ledger;
launch.json exclusive creation prevents starting it again. Do not recreate activation/output to replay this ticket.
Original private model files remain local, public delivery contains hashes and bindings only.
STOP: create results/role_newton_generation_20261021_v1/STOP. Parent cleans its own child groups only.
Independent evaluation and publication scripts are bound separately before audit, not model changes.
