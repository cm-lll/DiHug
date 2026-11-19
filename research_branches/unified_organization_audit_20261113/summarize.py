"""Independent CPU graph-invariant checks and descriptive tables."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,csv,hashlib
from pathlib import Path
import numpy as np
import networkx as nx
from run import Score,ROOT,OUT,PANEL,table,save

def main():
    rows=list(csv.DictReader((OUT/'perturbations.csv').open()));hist=list(csv.DictReader((OUT/'existing_graphs.csv').open()))
    protocol=json.loads((PANEL/'protocol.json').read_text());scales=json.loads((ROOT/'results/typed_heat_degree_audit_20261111/reference_scaling.json').read_text());bits=np.load(OUT/'cpu_perturbation_bits.npz')
    checked=0;missing=[];summary=[];groups=[];bounds=[]
    for ds,seed in [('IMDB',2026110411),('PubMed',2026110421)]:
        sc=Score(ds,protocol['parents'][f'{ds}_{seed}_A']['data'],scales)
        for row in rows:
            if row['dataset']!=ds or row['kind']=='reference':continue
            key=f'{ds}_{row["kind"]}_{row["seed"]}_{row["requested"]}';b=np.unpackbits(bits[key])[:len(sc.f)].astype(bool)
            g=nx.Graph();g.add_nodes_from(range(sc.n));g.add_edges_from(sc.q[:,b].T.tolist())
            assert g.number_of_edges()==int(row['edges']);assert sum(nx.triangles(g).values())//3==int(row['triangles'])
            comps=list(nx.connected_components(g));assert len(comps)==int(row['components']);assert max(map(len,comps))==int(row['LCC'])
            if row['kind'] in ('swap','concentrate'):
                assert np.array_equal(np.bincount(sc.f,weights=b,minlength=sc.r),sc.count)
            if row['kind']=='swap':assert np.array_equal(sc.degrees(b),sc.rd)
            if row['requested']!=row['accepted']:missing.append(key)
            checked+=1
        for kind in ('add','delete','concentrate','swap'):
            rs=[r for r in rows if r['dataset']==ds and r['kind']==kind]
            for level in sorted(set(int(r['requested']) for r in rs)):
                sub=[r for r in rs if int(r['requested'])==level];result=dict(dataset=ds,kind=kind,operations=level,repeats=len(sub))
                for m in ('H','N','raw_joint_transport','count_lower_bound','edges','triangles','LCC','low_fraction_error','hub_mass_error'):
                    a=np.array([float(v[m]) for v in sub]);result[m+'_mean']=float(a.mean());result[m+'_sd']=float(a.std(ddof=1))
                summary.append(result)
            for sd in sorted(set(r['seed'] for r in rs)):
                sub=sorted([r for r in rs if r['seed']==sd],key=lambda r:int(r['requested']))
                groups.append(dict(dataset=ds,kind=kind,seed=sd,H_strict_positive=all(float(r['H'])>1e-12 for r in sub),
                    H_downward_intervals=sum(float(b['H'])<float(a['H'])-1e-12 for a,b in zip(sub,sub[1:])),
                    transport_downward_intervals=sum(float(b['raw_joint_transport'])<float(a['raw_joint_transport'])-1e-12 for a,b in zip(sub,sub[1:]))))
        for sd in sorted(set(r['train_seed'] for r in hist if r['dataset']==ds)):
            for arm in ('A','A_typed'):
                sub=[r for r in hist if r['dataset']==ds and r['train_seed']==sd and r['arm']==arm]
                result=dict(dataset=ds,train_seed=sd,arm=arm,samples=len(sub))
                for m in ('H','N','raw_joint_transport','relation_count_L1','triangles','LCC','low_fraction_error','hub_mass_error'):
                    a=np.array([float(v[m]) for v in sub]);result[m+'_mean']=float(a.mean());result[m+'_sampling_sd']=float(a.std(ddof=1))
                bounds.append(result)
    table('perturbation_summary.csv',summary);table('monotonicity.csv',groups);table('historical_model_summary.csv',bounds)
    changed=[p for p,h in json.loads((OUT/'bindings.json').read_text()).items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed
    save('independent_checks.json',dict(graphs_checked=checked,networkx_edges_triangles_components_agree=True,all_operation_targets_completed=not missing,missing=missing,changed_sources=changed,production_H_max_error=max(float(r['production_H_difference']) for r in hist),model_calls=0,GPU_hours=0))
    print((OUT/'independent_checks.json').read_text())

if __name__=='__main__':main()
