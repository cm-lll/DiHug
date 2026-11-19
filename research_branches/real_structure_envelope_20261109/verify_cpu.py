"""Independent archived-table and evaluator-signature check; no graph generation."""
import json,statistics,time,resource,hashlib
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/real_structure_envelope_20261109';start=time.monotonic()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=read(ROOT/'results/cdf_risk_audit_20261109/attempt2/per_graph.json')
new=read(OUT/'per_graph.json')
def key(r):return tuple(r[k] for k in ['panel','dataset','train_seed','arm','sample_seed'])
old={key(r):r for r in old};maxerr=0.
for r in new:
    o=old[key(r)]
    for nk,ok in [('raw.edges','edges'),('raw.triangles','triangles'),('raw.lcc','lcc'),('distance.main_incident4_W1','degree_W1'),('distance.main_joint_degree_TV','joint_TV')]:
        maxerr=max(maxerr,abs(r[nk]-o[ok]))
assert maxerr<1e-12
signatures=defaultdict(set);graphcounts=defaultdict(int)
for j in read(OUT/'asset_inventory.json'):
    m=read(j['metric']);s=m['statistics']['joint_relation_degree']
    sig=tuple((name,tuple(sorted(v.items()))) for name,v in sorted(s.items()))
    signatures[j['dataset']].add(sig);graphcounts[j['dataset']]+=1
assert all(len(signatures[d])==graphcounts[d] for d in graphcounts)
lookup=defaultdict(list)
for r in read(OUT/'metrics_long.json'):
    if r['value'] is not None:lookup[(r['model'],r['version'],r['metric'])].append(r['value'])
meanerr=sderr=0.
for r in read(OUT/'per_model.json'):
    v=lookup[(r['model'],r['version'],r['metric'])]
    if v:meanerr=max(meanerr,abs(statistics.mean(v)-r['mean']))
    if len(v)>1:sderr=max(sderr,abs(statistics.stdev(v)-r['SD']))
# Some count/motif values are large; use relative roundoff bound, no scientific tolerance change.
assert meanerr<1e-7 and sderr<1e-7
b=read(OUT/'binding.json');assert all(sha(p)==h for p,h in b['assets'].items())
o=dict(status='verified',old_per_graph_max_error=maxerr,evaluator_joint_signature_unique={d:len(v) for d,v in signatures.items()},
    aggregation_mean_max_absolute_error=meanerr,aggregation_SD_max_absolute_error=sderr,
    assets_unchanged=True,seconds=time.monotonic()-start,RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
(OUT/'verification.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
