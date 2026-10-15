"""Explicit read-only old B weight/state import. NEVER resumes an old chain."""
import json
from pathlib import Path
import numpy as np
import torch
from hsgd.bw_preflight.common import config, imdb_case, sha, trusted_load
from hsgd.bw_preflight.verify import audit, source_info, rng_check, integer


def import_fixed_inputs(b_dir, source_package, chain_dir=None):
    b_dir, source_package = Path(b_dir).resolve(), Path(source_package).resolve()
    if (source_package/"manifest_v4.json").is_file() is False:
        raise RuntimeError("explicit pristine v4 source package required")
    source = source_info(source_package)
    if source["config"] != config():
        raise RuntimeError("v4 architecture/config differs; cannot implicitly adapt weights")
    verified = audit(b_dir,source_package,"B")
    if verified["exit_code"] != 0 or not verified["runtime_version_binding_verified"]:
        raise RuntimeError("B evidence not verified with actual runtime version binding")
    weight = b_dir/"preflight_last.pt"
    ck = trusted_load(weight)
    case = imdb_case(source_package/"data/IMDB_movie_reference.npz", config())
    if ck["completed_updates"] != 100:
        raise RuntimeError("only B100 engineering checkpoint supported")
    expected_prior = torch.as_tensor(case["prior"],dtype=torch.float64)
    if not torch.equal(ck["model"]["prior"],expected_prior):
        raise RuntimeError("checkpoint family prior differs from data")
    record = dict(weight_path=str(weight),weight_sha256=sha(weight),data_sha256=config()["imdb_sha256"],
                  B_audit=verified,completed_engineering_updates=100,
                  optimizer_state_import="read-only B100 moments for disposable paired update101; no training-resume/RNG restoration",
                  old_chain_resume_certified=False,original_binding=ck["binding"])
    state = np.isin(np.arange(case["space"].total),case["clean_ids"]).astype(np.uint8)
    t = 500
    if chain_dir is not None:
        path = Path(chain_dir).resolve()/"chain_last.pt"
        saved = trusted_load(path)
        binding = dict(ck["binding"],weight_sha256=sha(weight))
        if saved["binding"] != binding:
            raise RuntimeError("old state/source-weight/code/data binding mismatch")
        t=integer(saved["remaining_time"],"saved remaining time")
        steps=integer(saved["completed_reverse_steps"],"saved reverse steps")
        if t+steps != config()["T"] or t == 0:
            raise RuntimeError("saved step counters invalid for fixed-state performance check")
        rng_check(saved["rng"])
        state=np.asarray(saved["state"])
        if state.shape!=(case["space"].total,) or not ((state==0)|(state==1)).all():
            raise RuntimeError("saved state shape/support invalid")
        for f in range(case["n_families"]):
            pi=case["prior"][f]
            if pi in (0.,1.) and not (state[case["space"].offsets[f]:case["space"].offsets[f+1]]==int(pi)).all():
                raise RuntimeError("saved fixed-family state outside exact support")
        state=state.copy()
        summary=Path(chain_dir)/"summary.json"
        record.update(state_path=str(path),state_sha256=sha(path),remaining_time=t,
                      completed_reverse_steps=steps,RNG_validated_but_not_restored=True,
                      summary=json.loads(summary.read_text()) if summary.exists() else None)
    else:
        record.update(state_source="fixed reference engineering input at t500; no generated context substituted")
    return ck,case,state,t,record
