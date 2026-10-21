# Large full-model Newton CPU step: frozen engineering protocol

Baseline8855b78b. Original DP/Newton/model/kernel/query methods retained unchanged.
Complete datasets IMDB1653/4079,PubMed1565/12284,complete4-type10-family anonymous joint roles,verified source SHA.
Source-bound large config8layers,256/64/128,heads8,FF256/64/256,joint128,CN32,FP32/T1000/20blocks.
Existing three broad final6000 weight bytes/source/env checked. Strict tensor loading and semantic binding reported separately.
Different prior and T64 training binding invalidate cross-scale quality claims even if shapes load.
Use brand-new random initialized CapacityDenoiser from SAME existing class and width config,seed2026102147;no partial transfer/no training.
All times random weights,not generation quality. All six states independently forward-noised from known reference,not free chain states.
Noise seed[2026102148,dataset_index,t],sample uniform RNG[2026102149,dataset_index,t],query original seed[2026102149+index,t,61].
Order IMDB20/500/1000,thenPubMed20/500/1000. No resampling or state replacement.
Before model step: exact wedge2m and unchanged querypath2m guards;Newton2m-elements/256MiB-workspace;full memory estimate<=6GiB.
Every process RSS<=8GiB parent monitor;address-space hard16GiB;item cold+repeat420s including imports,construction,checks,IO,cleanup.
Newton unchanged max200iterations,4000eval,30s/matrix,cold bias,line search,damping,support and tail gates.
Each new read independently residual<=1e-8,KKT<=2e-10,lognorm<=2e-12,rowtail<=1e-280,total<=1e-276.
Actual role values/context checked for tail bound<=1e-250 and predeclaredFP32 arithmetic bound.
No DP fallback,role truncation,smaller graph,layer/query reduction,mixed precision,new backend or optimizer.
One frozen Xt per20-block full_sweep,all active slots exactly once;only complete returned state committed.
Cold clears one-time degree-law tables;repeat keeps only identical-time degree-law tables. Every Newton/graph/backbone/head reruns.
CN contextual node cache only inside original one-step context;no reuse across cold/repeat. No 1000-time precache.
Use original runtime.reverse_step(trace=True),which computes one EXTRA diagnostic posterior per block;report overhead,not pure uninstrumented throughput.
Timing source wrappers only:degree_law/degrees/Newton/modules. Network_seconds and submodules are inclusive;do not add nested counters.
Role loop residual=role_attention minus convolution/Newton/role encoder+value;includes likelihood lookup/assembly,context merge and loop bookkeeping,not pure lookup alone.
Model construct,query/context/assembly,backbone,CN,head,posterior,sample/audit,certificate IO,baseline currentRSS and process peak all separate.
Repeat restores same frozen input/uniform RNG,no old prediction reuse;compare new p0/q/state and role allocation.
All six-state full-step results retained inclfailures. Every next item needs remaining>=420+720s,otherwise not-run explicitly.
Unique3600s clock from first preparation1792566249,reserve12min audit/publication. No new full chains/train/GPU/BW/notebook.
Full forward success does not validate backward memory,quality,cross-backend reproducibility or all1000-step costs.
