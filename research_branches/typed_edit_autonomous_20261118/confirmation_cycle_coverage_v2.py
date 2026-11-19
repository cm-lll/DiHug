"""Read-only coverage by count coordinate and true inverse direction."""
import confirmation_fit as fit
import confirmation_common as contract
from round01 import t,b,np,Path,ROOT,SRC,time
OUT=ROOT/'results/typed_edit_autonomous_20261118/confirmation_v1/cycle_coverage_v2'

def main():
    b.setup();fit.source_check(fit.CACHE);fit.source_check(fit.OUT)
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    files=[Path(__file__),SRC/'CYCLE_COVERAGE_SCHEMA.md',ROOT/'results/typed_edit_autonomous_20261118/confirmation_v1/cycle_coverage/failure.json',fit.CACHE/'manifest.json',fit.CACHE/'acceptance.json',fit.OUT/'manifest.json',fit.OUT/'acceptance.json']
    files+=list(fit.CACHE.glob('*/repeat*/new_train.npz'))+list(fit.CACHE.glob('*/schema.json'))+list(fit.OUT.glob('inputs/*.npz'))
    files+=[b.RUN/f'data/{ds}/graph.json' for ds in fit.DATASETS]
    contract.bind(OUT,files,new_training_updates=0,new_graphs=0)
    rows=[]
    for ds in fit.DATASETS:
        ref=t.metrics.counts(b.graph(ds));cols=b.read(fit.CACHE/ds/'schema.json')['columns']
        banks=[('old',0,np.load(fit.OUT/f'inputs/{ds}_old_train.npz'))]
        banks += [('new',r,np.load(fit.CACHE/ds/f'repeat{r}/new_train.npz')) for r in fit.REPEATS]
        for kind,repeat,z in banks:
            k=len(cols);encoded=z['x'][np.arange(len(z['y'])),z['y'],k:]
            inverse=z['delta'][np.arange(len(z['y'])),z['y']] if 'delta' in z.files else encoded
            counts_all=z['counts'] if 'counts' in z.files else z['x'][:,0,:k]
            assert np.array_equal(np.sign(inverse),np.sign(encoded))
            assert np.array_equal(counts_all,counts_all.astype(np.int64))
            for j,(family,key) in enumerate(cols):
                counts=counts_all[:,j];delta=inverse[:,j];reference=ref[family][key]
                rows.append(dict(dataset=ds,bank=kind,repeat=repeat,family=family,key=key,reference=reference,records=len(counts),minimum=int(counts.min()),maximum=int(counts.max()),mean=float(counts.mean()),above_reference=int((counts>reference).sum()),below_reference=int((counts<reference).sum()),inverse_positive=int((delta>0).sum()),inverse_negative=int((delta<0).sum()),above_and_inverse_negative=int(((counts>reference)&(delta<0)).sum()),below_and_inverse_positive=int(((counts<reference)&(delta>0)).sum())))
    contract.verify(OUT);contract.put(OUT,'summary.json',dict(rows=rows,causal_explanation_not_proven=True,changes_to_confirmation=False))
    contract.put(OUT,'acceptance.json',dict(passed=True,coordinate_rows=len(rows),new_fits=0,new_graphs=0,seconds=time.monotonic()-start))
    print('Frozen direction coverage accepted',len(rows),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():contract.put(OUT,'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise

