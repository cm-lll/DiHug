"""Aggregate completed CPU records; independent finite Bernoulli identity check."""
import json,itertools,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/joint_degree_signal_20261109';V=OUT/'verified'
def save(name,x): (OUT/(name+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
rows=json.loads((V/'slot_feedback.json').read_text());summary=[]
for ds in ['IMDB','PubMed']:
 for t in [1,500,1000]:
  rr=[r for r in rows if r['dataset']==ds and r['t']==t]
  for method in ['old','new']:
   good=[r for r in rr if r[method+'_flip_delta']<-1e-12]
   mass=sum(abs(r[method+'_logit_derivative']) for r in good)
   bad=[r for r in good if r['triangle_error_change']>1e-12]
   summary.append(dict(dataset=ds,t=t,method=method,selected_slots=len(rr),rewarded=len(good),triangle_cost_count=len(bad),
     hub_cost_count=sum(r['hub_error_change']>1e-12 for r in good),isolates_cost_count=sum(r['isolates_error_change']>0 for r in good),
     rewarded_abs_logit_feedback=mass,triangle_cost_feedback_share=sum(abs(r[method+'_logit_derivative']) for r in bad)/mass if mass else None,
     CE_opposed_count=sum(r[method+'_CE_opposed'] for r in rr),
     scope='unweighted selected-coordinate sums, not population/parameter gradient, probability or training effect'))
save('summary',summary)
# A nonseparable discrete loss verifies sign and p(1-p) factor; no graph draws.
p=[.17,.46,.83];states=list(itertools.product([0,1],repeat=3))
def loss(b):return (b[0]+2*b[1]-1)**2+3*b[1]*b[2]-.4*b[0]*b[2]
def law(b):return math.prod(pi if bi else 1-pi for pi,bi in zip(p,b))
errors=[]
for e in range(3):
 exact=sum(law(b)*loss(b)*(b[e]-p[e]) for b in states)
 conditional=0.
 for b in states:
  a=list(b);a[e]=0;c=list(b);c[e]=1
  conditional+=law(b)*p[e]*(1-p[e])*(loss(c)-loss(a))
 errors.append(abs(exact-conditional))
assert max(errors)<1e-12
save('tests',dict(finite_Bernoulli_states=8,coordinates=3,max_identity_error=max(errors),
 scope='exact finite sum identity; not network gradient, unbiased finite variance claim, or new graphs'))
save('engineering_notes',dict(first_attempt='Query-order equality assertion stopped before computing signal; failure.json preserved.',
 correction='Archived conditional prediction order differs from final-graph fixture. Verified full canonical pair IDs, relation IDs and labels before building score in archived query order; no check removed or formula changed.',
 final_directory=str(V),original_assets_unchanged=True))
save('decision',dict(status='CPU audit complete; no training authorized',
 result='Score-to-logit mapping is defined and computable on old conditional assets. Mixed signal/cost; not a new winner or rejection.',
 missing='Known PubMed0421 final-graph sample directory has no saved per-edge prediction vectors. Existing conditional panels are IMDB0411/PubMed0422, raw1040/2040; cannot substitute identities.',
 next='All12 archived probability panels: evaluate union of existing selected coordinates on both saved hard draws (<=5530). Additionally PubMed t1000 both historical parents and both draws: full18336 slots (73344). <=78874 coordinate evaluations, duplicates reusable. No new graph draws, formula or lambda search.',
 proposed_CPU_budget=dict(seconds=900,threads=1,RSS_MiB=2048,GPU=0),
 budget_basis='Current 2765 slots plus load/checks6.29s; <=78874 coordinate evaluations ~180s linear extrapolation plus3x headroom within900s. Full-slot panel may have different cost; benchmark first panel and stop if bound cannot cover. Two draws remain insufficient for variance convergence. No model calls.',
 training=False))
bindings=json.loads((V/'binding.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in bindings.items())
save('completion',dict(status='complete',source_assets_verified=len(bindings),
 scientific_receipt=json.loads((V/'receipt.json').read_text()),
 notes='One failed CPU setup attempt preserved; 6.29s is successful worker time, not total conversational task time.',
 output_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [V/'slot_feedback.json',V/'panel_summary.json',OUT/'summary.json',OUT/'tests.json']}))
print(json.dumps(dict(status='complete',assets_verified=len(bindings),max_identity_error=max(errors))))
