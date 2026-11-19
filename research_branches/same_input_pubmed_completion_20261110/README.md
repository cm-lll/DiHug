# Complete only the two missing PubMed one-step conditions

User approved completion after the previous partial report. Keep the old window,
partial results and missing accounting unchanged. This is a new bounded 5min /
0.05GPU-hour completion window; scientific stop210s, final300s, max one2080Ti.
Run original task indices2/3 only. IMDB results are reused without model calls.
No loss, optimizer, displacement projection, input, evaluation or tolerance
change. Worker differs from original only in output directory and final newline;
CPU evaluator is identical modulo terminal newline. The first strict comparison
failure and its terminal-newline-only correction are recorded before GPU start.

Supervisor runs detached from conversational tool sessions and journals each
allocation, PID, heartbeat and release. It checks idle GPU0 before each worker,
limits occupied time including imports/load/save, and completes CPU evaluation
after releasing GPU. It never terminates unrelated processes. New frozen source
hashes and protocol identify this wrapper; existing checkpoints remain private.

Two temporary updates,16 forwards,10 VJPs. Same-input readout uses identical
train mode; six other-time endpoints use prior eval protocol and cached before
states. Three exact risks and two paired K256 risks are unchanged. No new formal
training/full reverse generation, no threshold or learning-rate search.
