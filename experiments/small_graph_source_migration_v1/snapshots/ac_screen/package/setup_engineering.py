"""Synthetic, type-separated engineering dictionaries; never formal."""
import json,numpy as np
from features import SPEC,synthetic_bases
from prepare import ROOT,save,sha
def main():
    cfg=json.loads((ROOT/'config.json').read_text())
    cfg.update(version='typed_refex_rolx_192_v2',rounds=1,inputs_per_update=2,
        max_updates_per_dataset_arm=3,feature_spec=SPEC,
        query_contract='ALL active slots in ONE forward per input; S=18336, loss=(sum CE input1 + sum CE input2)/(2*S)',
        source_note='normal2 sequential backward; synthetic engineering dictionaries ONLY; no formal initialization')
    for name,entry in cfg['datasets'].items():
        with np.load(ROOT/entry['file']) as z:
            basis,receipt=synthetic_bases(len(np.unique(z['types'])),len(z['schema']))
        path=ROOT/'assets'/(name+'_synthetic_basis.npz')
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,basis=basis)
        entry.update(basis_file=str(path.relative_to(ROOT)),basis_sha256=sha(path),basis_kind='synthetic engineering only')
        save(ROOT/'assets'/(name+'_synthetic_receipt.json'),receipt)
    save(ROOT/'config.json',cfg)
if __name__=='__main__':main()
