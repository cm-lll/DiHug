# DiHug typed-edit implementation for review

This snapshot covers the original paper's revised single-reference heterogeneous
graph generation method as of 2026-11-19. Each reference is fitted separately.
Stage one uses the original scorer to approach anonymous joint relation-degree
targets through legal edits. Stage two uses degree-preserving switches and a
learned score given by the difference in a graph potential. The potential uses
typed triangle counts and bipartite four-cycle counts within cross-type relations.
Candidate-list cross-entropy supervises known inverse edits.

| Component | Source |
| --- | --- |
| Original graph representation, typed edits and GPU candidate generation | [Frozen implementation](frozen_r2/src/dihug/core/) |
| Stage-one scoring, candidate selection and transition checks | [confirmation_prefix_v2.py](../../research_branches/typed_edit_autonomous_20261118/confirmation_prefix_v2.py) |
| Exact structure changes and connectivity checks | [candidate_consequences.py](../../research_branches/typed_edit_autonomous_20261118/candidate_consequences.py) and [potential.py](../../research_branches/typed_edit_autonomous_20261118/potential.py) |
| Stage-two training and matched generation | [confirmation_fit.py](../../research_branches/typed_edit_autonomous_20261118/confirmation_fit.py) and [confirmation_deploy.py](../../research_branches/typed_edit_autonomous_20261118/confirmation_deploy.py) |
| IMDB coverage and PubMed matched augmentation | [cycle_repair_confirmation entry points](../../research_branches/typed_edit_autonomous_20261118/) named `cycle_confirmation_*.py` and `crossdomain_confirmation_*.py` |
| ACM candidate prescreening with archived trajectories | [Prescreen code, protocol and summary](../../research_branches/acm_candidate_prescreen/) |

The historical runners expect the corresponding reference files, checkpoints,
training banks and acceptance manifests under `results/`. Those runtime assets
must be supplied separately. Some historical checks bind absolute asset paths
and source hashes. Running on another host requires validated bindings for that
host. The code keeps those checks in place. The ACM prescreen README documents
the archive and output path settings for its offline evaluation.

`SOURCE_MANIFEST.json` records the original and distributed SHA-256 for each
Python file. The source was selected from completed confirmation manifests with
supporting dependencies. The path changes replace host-specific checkout paths
with paths relative to the checked-out repository. The graph operators, losses
and scoring formulas are preserved. The original studies used Python 3.9,
PyTorch 2.1, NumPy 1.26.4, SciPy 1.11.0 and NetworkX 2.8.7. The GPU runners
require a compatible CUDA-enabled PyTorch build.

The ACM prescreen experiments measure single-step candidate selection on fixed
archived trajectories. Full-generation performance for that prescreen remains
an open evaluation question.
