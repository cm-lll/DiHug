"""Tabulate all existing vector/count fields, with reference, no new score."""
import json,pathlib,csv,collections,math
import numpy as np
OUT=pathlib.Path(__file__).parent
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(name,rows):
 if not rows:return
 with (OUT/'tables'/name).open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def patterns(s):
 out={'typed_triangles':s['typed_triangles'],'typed_four_cycles':s['typed_four_cycles']}
 for i in range(3):out['MP'+str(i+1)]=s['mp'][i]['counts'];out['walk'+str(i+1)]=s['typed_walks'][i]['counts']
 return out
def main():
 r=read(OUT/'evaluation.json');ref=read(OUT/'run/derived_lcc_evaluation/IMDB_reference.json')['statistics'];rp=patterns(ref)
 z=np.load(OUT/'reference/IMDB.npz',allow_pickle=False);names=['Director','Actor','Movie','Genre'];codes=['D','A','M','G']
 write('schema.csv',[dict(family=i,type0=int(a),type1=int(b),type0_name=names[a],type1_name=names[b],relation_code=codes[a]+codes[b],reference_edges=ref['family_edges'][i],reference_capacity=ref['capacities'][i]) for i,(a,b) in enumerate(z['schema'])])
 write('type_mapping.csv',[dict(type_index=i,name=n,code=codes[i],reference_count=ref['typed_degree'][n]['count']) for i,n in enumerate(names)])
 keys={k:set(v) for k,v in rp.items()};per=[];closure=[];spectra=[];hist=[];joint=[]
 for x in r['records']:
  for k,v in patterns(x['statistics']).items():keys[k].update(v)
 for x in r['records']:
  ident={k:x[k] for k in ('dataset','train_seed','arm','checkpoint','sample_seed','version')};s=x['statistics']
  for kind,v in patterns(s).items():
   rv=rp[kind];a,b=sum(v.values()),sum(rv.values())
   for k in sorted(keys[kind]):
    n,m=v.get(k,0),rv.get(k,0);fn=n/a if a else None;fm=m/b if b else None
    per.append(dict(**ident,kind=kind,pattern=k,generated=n,reference=m,signed_difference=n-m,absolute_error=abs(n-m),generated_composition=fn,reference_composition=fm,
     composition_difference=fn-fm if fn is not None and fm is not None else None,reason='zero_total_composition' if fn is None or fm is None else None))
  for key,v in s['typed_twohop_closure'].items():
   rv=ref['typed_twohop_closure'][key]
   for label in ('path','unique_pair'):
    num,den=v[label+'_numerator'],v[label+'_denominator'];rn,rd=rv[label+'_numerator'],rv[label+'_denominator'];value=v[label+'_ratio'];reference=rv[label+'_ratio']
    closure.append(dict(**ident,pattern=key,denominator_contract=label,numerator=num,denominator=den,ratio=value,reference_numerator=rn,reference_denominator=rd,reference_ratio=reference,
     numerator_signed_difference=num-rn,numerator_absolute_error=abs(num-rn),denominator_signed_difference=den-rd,denominator_absolute_error=abs(den-rd),
     ratio_difference=value-reference if value is not None and reference is not None else None,ratio_absolute_error=abs(value-reference) if value is not None and reference is not None else None,
     reason=v[label+'_reason'],reference_reason=rv[label+'_reason']))
  subsets=[('ALL',s['total_degree'],ref['total_degree'])]+[(t,s['typed_degree'][t],ref['typed_degree'][t]) for t in ref['typed_degree']]
  for typ,d,rd in subsets:
   for rank in range(max(len(d['sorted_degrees']),len(rd['sorted_degrees']))):
    v=d['sorted_degrees'][rank] if rank<len(d['sorted_degrees']) else None;rv=rd['sorted_degrees'][rank] if rank<len(rd['sorted_degrees']) else None
    spectra.append(dict(**ident,node_type=typ,rank=rank+1,generated_degree=v,reference_degree=rv,reason='rank_absent_after_node_loss' if v is None else None))
   for relation,h,rh in [('total',d['histogram'],rd['histogram'])]+[(str(i),v['histogram'],rd['by_relation'][i]['histogram']) for i,v in enumerate(d.get('by_relation',[]))]:
    n,m=sum(h),sum(rh)
    for deg in range(max(len(h),len(rh))):
     a=h[deg] if deg<len(h) else 0;b=rh[deg] if deg<len(rh) else 0
     hist.append(dict(**ident,node_type=typ,relation=relation,degree=deg,generated_count=a,reference_count=b,generated_fraction=a/n if n else None,reference_fraction=b/m if m else None))
  for typ,v in s['joint_relation_degree'].items():
   rv=ref['joint_relation_degree'][typ];n,m=sum(v.values()),sum(rv.values())
   for k in sorted(set(v)|set(rv)):
    a,b=v.get(k,0),rv.get(k,0);joint.append(dict(**ident,node_type=typ,integer_degree_vector=k,generated_count=a,reference_count=b,generated_fraction=a/n if n else None,reference_fraction=b/m if m else None))
 write('all_pattern_counts_compositions.csv',per);write('all_typed_closure_details.csv',closure);write('complete_degree_spectra.csv',spectra);write('all_degree_histograms.csv',hist);write('complete_joint_degree_histograms.csv',joint)
 # Per-seed means of count/absolute-error/composition are computed from both sample graphs.
 grouped=collections.defaultdict(dict)
 for row in per:
  key=tuple(row[k] for k in ('dataset','train_seed','arm','checkpoint','version','kind','pattern'));grouped[key][row['sample_seed']]=row
 means=[]
 for key,byseed in grouped.items():
  ident=dict(zip(('dataset','train_seed','arm','checkpoint','version','kind','pattern'),key))
  good=set(byseed)=={2026102821,2026102822}
  for metric in ('generated','reference','signed_difference','absolute_error','generated_composition','reference_composition','composition_difference'):
   values=[v[metric] for v in byseed.values()];value=sum(values)/2 if good and all(v is not None for v in values) else None
   means.append(dict(**ident,metric=metric,value=value,reason=None if value is not None else 'two_samples_required_or_undefined'))
 write('all_pattern_seed_means.csv',means)
 print('detail rows',dict(patterns=len(per),closure=len(closure),spectra=len(spectra),hist=len(hist),joint=len(joint)))
if __name__=='__main__':main()

