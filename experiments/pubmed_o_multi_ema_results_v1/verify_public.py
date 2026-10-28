"""Offline CPU verification of public archive; no weights/model or formal clock."""
import argparse,collections,hashlib,io,json,pathlib,time,zipfile
import numpy as np
from audit_results import graph_stats,equal,tv

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    started=time.time();root=pathlib.Path(a.root);out=pathlib.Path(a.output)
    if out.exists():raise FileExistsError(out)
    def check():
        if time.time()-started>=120:raise TimeoutError('bounded offline CPU verification120seconds')
    manifest=(root/'SHA256SUMS').read_text().splitlines()
    for line in manifest:
        check();expected,name=line.split('  ',1);target=(root/name).resolve();assert root.resolve() in target.parents
        assert digest(target)==expected,name
    with zipfile.ZipFile(root/'runtime_source.zip') as z:
        with np.load(io.BytesIO(z.read('pubmed_o_multi_ema_v1/package/assets/PubMed.npz')),allow_pickle=False) as data:
            types=data['types'];schema=data['schema'];ref=graph_stats(types,data['edges'].T,schema)
    result=json.loads((root/'run/unified_evaluation.json').read_text());records=[]
    for cell in result['cells'].values():
        for row in cell['graphs']:
            check();folder=root/'run'/('PubMed_s%d_O'%row['train_seed'])/row['readout']/('sample_%d'%row['sample_seed'])
            with np.load(folder/'graph.npz',allow_pickle=False) as data:
                np.testing.assert_array_equal(data['types'],types);np.testing.assert_array_equal(data['schema'],schema)
                s=graph_stats(types,data['edges'],schema);np.testing.assert_array_equal(s['families'],data['families'])
            m=row['metrics']
            for k in ['edges','triangles','lcc','isolates','maximum_degree','average_clustering']:equal(s[k],m[k])
            assert s['mp']==[v['total'] for v in m['mp']]
            equal(tv(s['typed_triangles'],ref['typed_triangles']),m['primary']['typed_triangle_composition_TV'])
            w=j=0.
            for typ in np.unique(types):
                mask=types==typ;rels=[r for r,pair in enumerate(schema) if typ in pair];values=[]
                for r in rels:
                    aa,bb=s['degree'][mask,r],ref['degree'][mask,r];n=int(max(aa.max(),bb.max()))+1
                    values.append(np.abs(np.cumsum(np.bincount(aa,minlength=n)/len(aa)-np.bincount(bb,minlength=n)/len(bb))).sum())
                w+=float(mask.mean())*np.mean(values)
                aa,bb=collections.Counter(map(tuple,s['degree'][mask])),collections.Counter(map(tuple,ref['degree'][mask]));n=int(mask.sum())
                j+=float(mask.mean())*.5*sum(abs(aa[k]/n-bb[k]/n) for k in aa.keys()|bb.keys())
            equal(w,m['W1_definitions']['incident4']);equal(.4*w,m['W1_definitions']['all10_compat']);equal(j,m['primary']['type_weighted_joint_degree_TV'])
            records.append((row['train_seed'],row['readout'],row['sample_seed'],abs(s['edges']-926)))
    assert len(records)==64
    pair_checks=0
    for kind,decay in result['assessment']['decays'].items():
        for row in decay['per_seed']:
            values=lambda k:[r[3] for r in records if r[0]==row['train_seed'] and r[1]==k]
            a,b=values(kind),values('raw');assert len(a)==len(b)==2
            equal(sum(a)/2-sum(b)/2,row['difference']);pair_checks+=1
    receipt=dict(status='passed',payload_hashes=len(manifest),independent_graphs=len(records),paired_seed_decay_checks=pair_checks,seconds=time.time()-started,scope='offline CPU public payload+discrete graph metrics; no private weights, no torch/model, no scientific timer activation')
    check();out.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
