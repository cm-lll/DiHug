"""Inventory fixed sampled paths without changing any input bank."""
import pathlib,json,numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
out=[]
for d in ('IMDB','PubMed'):
 ref=np.load(ROOT/'data'/d/'reference.npz');all_edges={tuple(map(int,row)) for row in ref['edges']}
 for p in sorted((ROOT/'data'/d).glob('bank*.npz')):
  z=np.load(p);nodes=set();edges=set()
  for row in z['nodes']:
   row=[int(x) for x in row if x<192];nodes.update(row);edges.update(tuple(sorted((u,v))) for u,v in zip(row[:-1],row[1:]))
  out.append(dict(dataset=d,bank=p.name,node_coverage=len(nodes),edge_coverage=len(edges),reference_edges=len(all_edges),missing_nodes=sorted(set(range(192))-nodes),missing_edges=sorted(all_edges-edges),full_reference_available=True,no_train_test_split=True,bank_changed=False))
(ROOT/'path_bank_coverage.json').write_text(json.dumps(out,indent=2))
print(out)
