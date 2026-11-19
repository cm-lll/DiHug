# Single-step hard-structure continuation

Purpose: compare one IID noisy input / 64 hard clean samples with four IID
noisy inputs / 16 hard clean samples each. A late CN and the five existing hard
structural risks are unchanged. Counts and present/absent triangle risks use
their exact expectations; joint degree and components use independent-output
LOO score gradients. Training has **no reverse chain and no terminal critic**.

Each arm resumes the same CE-trained complete u2000 checkpoint, including AdamW
moments and EMA. It performs 1000 additional optimizer updates. B4 averages all
four input gradients before one optimizer/EMA update. This changes input
exposure and compute, not the fixed-parameter expected objective. CE is logged
but has coefficient zero. No projection, clipping, rejection, learning-rate
change, structural quota or changes to reverse sampling.

The first input matches the old recorded history every update in both arms.
Additional inputs and hard draws use independently keyed, stateless RNG streams.
Their root seeds, update and input index fully determine the streams. Full
checkpoints preserve the parent training RNG plus the new stream definitions.
Saved state identity and next-input replay are checked; ordinary CUDA repeated
updates are **not** certified bitwise identical.

Eight tasks: IMDB 2026110411/412 and PubMed 2026110421/422, two arms per seed.
Each final EMA3000 readout generates four unchanged R1/T1000 chains with seeds
2026102821/22 and 2026110231/32. Every query plan and full-slot sampling uniform
is checked against the frozen historical contract. Existing ordinary CE models
are background only, not the matched structural-training arm.

Outputs are under `results/single_step_formal_20261111`. Do not rerun launch.py:
the exclusive launch lock prevents duplicate allocation. `window.json` fixes
the 240-minute total window, 210-minute scientific line and 16 GPU-hour limit,
including preflight allocation. `supervise.py` and independent `watchdog.py`
count allocation-to-exit time and terminate only this ticket's verified worker
process groups. CPU evaluation begins after each GPU worker exits. Original
graph metrics are primary; LCC and all relation/structure metrics are separate.

Progress: `supervisor_progress.json`, per-task `progress.json`, `training.jsonl`.
Completion: `supervisor_receipt.json`, `completion.json`, `REPORT_ZH.md`, full
per-graph/per-model/across-seed CSVs. Missing samples do not receive complete
model means. Four samples are not four independent training runs.

Environment: `python`.
Preparation order: prepare.py; preflight.py on an available host GPU; launch.py.
Never overwrite an existing run to reproduce it; use an isolated copy/new output
directory and rebind all inputs. Production/historical sources remain unchanged.
