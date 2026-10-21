# Fixed-weight DP versus Newton generation, 45 cumulative minutes

Baseline4061ccbb, unchanged solver ad562a70, three broad final6000 models from7b45d502.
Two arms DP/Newton × three original train seeds × same11 frozen targets × eight original sampling seeds2026102031..38 =528 NEW T64/R4 chains.
No fitting, old-control substitution, rejection sampling, output control, graph repair or proof-graph input.
The original chain function, kernel, model, logits, degree laws and sampling RNG are reused unchanged.
Initial/query/uniform/RNG hashes are checked across both arms/targets/seeds. Different states/graphs are expected.
Every Newton read independently checks row/normalizedcolumn1e-8, full-support KKT2e-10,
log normalization2e-12, exact mathematical support, perrow1e-280/total1e-276 tail limits and original resource/time gates.
Actual learned role values are finite and tail-context error bound<=1e-250. No step is accepted on solver.success alone.
Full ordinary-P original versus bounded-tail-interface success counts reported separately on NEW Newton trajectory states.
Failure saves matrix/receipt/last complete graph+RNG and stops the panel, no silent DP fallback, altered parameters or new window.
All raw graphs,64-step states, Newton L/P/logP, per-read receipts, paired random hashes and source/weight/target bindings retained.
All528 graphs independently checked for schema/degree/joint frequencies/W1/connectivity/triangles/paths/closure;
8-draw cells compared by exact type/relation-preserving isomorphism after WL prefilter, with edge distances.
Summaries first average8draws within trainingseed, then weight3trainseeds equally;528 graphs not528 independent trainings.
Main descriptive differences: Genre W1, MG signed edge error, sorted degree/hubs/lowtail, other types joint/degree errors,
closure/connectivity and topology duplicates. No retrospective quality pass threshold; improvements/tradeoffs all reported.
The prior approximate1-W1 is a historical benchmark, not a promise or retrospectively adjustable acceptance gate.
CPU six independent1-thread processes, no CUDA initialization. One activation from first preparation,2700s total,180s parent cleanup/audit reserve.
No new large-network, GPU/BW, training, tuning or next experiment automatically follows.
