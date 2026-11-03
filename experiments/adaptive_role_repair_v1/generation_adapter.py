"""Truthful dictionary/initialization adapter for the new repair ticket.

Importing this module is stdlib-only and does not construct a model or touch a
GPU. Model construction and torch.load are explicit functions for the sole new
authorized worker. No historical source, dictionary or checkpoint is modified.
This is NOT a launcher, training loop, sampling loop, or approval mechanism.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/adaptive_role_repair_20261103T032520'
BASELINE = ROOT/'results/late_final_single_input_20261102T132233'
EXPANSION = ROOT/'results/late_sampling_expansion_20261102T160103'
SCI = ROOT/'research_branches/typed_structure_early_cn_expansion_review/experiments/typed_structure_early_cn_expansion_v1'
PACKAGE = SCI/'package'
DATASET = 'PubMed'
TYPE_NAMES = ('Gene', 'Disease', 'Chemical', 'Species')
ROLE_KEYS = frozenset(('role_basis', 'role_offsets', 'role_projection.weight'))
METHODS = frozenset(('baseline', 'heldout', 'rolx_style_mdl', 'ard'))
SAMPLE_SEEDS = (2026102821, 2026102822, 2026110231, 2026110232)
_INSTALLED_ENCODER = None


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def seed32(root, **labels):
    payload = json.dumps(dict(root=int(root), **labels), sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def canonical_configuration(seed):
    """Frozen normal1 computation config; dictionary provenance stays separate.

    The inherited feature_spec.rank is historical metadata, NOT a loader or
    selected-K rule. Actual K/method/source hashes must be in dictionary_record.
    """
    job = next(j for j in read(BASELINE/'tasks.json') if j['dataset'] == DATASET and j['arm'] == 'E')
    cfg = copy.deepcopy(job['config'])
    cfg['seed'] = int(seed)
    if cfg['inputs_per_update'] != 1 or cfg['T'] != 1000 or cfg['rounds'] != 1:
        raise ValueError('baseline computational configuration is not normal1/R1/T1000')
    return cfg


def verify_configuration(cfg, seed):
    if cfg != canonical_configuration(seed):
        raise ValueError('configuration drift from frozen normal1 E computation')


def load_dictionary(record):
    """Load a newly frozen whole-PubMed dictionary; never forge legacy selection.

    Required fields: dataset, method, artifact, artifact_sha256, K, offsets,
    type_names, status='complete', frozen=True, provenance (nonempty mapping).
    Caller binds this record and all its source receipts in the new ledger.
    """
    import numpy as np
    if record.get('dataset') != DATASET or record.get('method') not in METHODS:
        raise ValueError('wrong dataset or unapproved dictionary method')
    if record.get('status') != 'complete' or record.get('frozen') is not True:
        raise ValueError('dictionary is incomplete or not frozen')
    encoder_kind = record.get('encoder_kind', 'nnls')
    if encoder_kind not in ('nnls', 'ard_map') or ((encoder_kind == 'ard_map') != (record['method'] == 'ard')):
        raise ValueError('only NNLS or an explicitly bound ARD MAP encoder is supported')
    if not isinstance(record.get('provenance'), dict) or not record['provenance']:
        raise ValueError('truthful dictionary fit/selection provenance required')
    if tuple(record.get('type_names', ())) != TYPE_NAMES:
        raise ValueError('dictionary type semantics changed')
    path = Path(record['artifact'])
    if not path.is_absolute():
        path = ROOT/path
    if sha(path) != record['artifact_sha256']:
        raise ValueError('dictionary artifact hash mismatch')
    with np.load(path, allow_pickle=False) as z:
        H = z['basis'].copy()
        offsets = z['offsets'].astype(np.int64)
        K = np.diff(offsets)
        if 'K' in z and not np.array_equal(z['K'], K):
            raise ValueError('embedded K disagrees with offsets')
        extra = ({key:np.asarray(z[key],dtype=np.float64).copy()
                  for key in ('row_norms','lambdas','phi')} if encoder_kind == 'ard_map' else {})
    if offsets.shape != (5,) or offsets[0] != 0 or offsets[-1] != len(H) or (K <= 0).any():
        raise ValueError('four nonempty typed dictionaries required; zero rank cannot be silently promoted')
    if offsets.tolist() != record['offsets'] or K.tolist() != record['K']:
        raise ValueError('dictionary metadata dimensional binding mismatch')
    if H.shape != (int(K.sum()), 630) or not np.isfinite(H).all() or (H < 0).any():
        raise ValueError('nonnegative finite630feature dictionary required')
    if not np.allclose(np.linalg.norm(H, axis=1), 1, atol=1e-12, rtol=1e-12):
        raise ValueError('dictionary basis is not unitL2')
    if encoder_kind == 'ard_map':
        if extra['row_norms'].shape != (len(H),) or extra['lambdas'].shape != (len(H),) or extra['phi'].shape != (4,):
            raise ValueError('ARD retained norms/lambda/typephi dimensional mismatch')
        if (not all(np.isfinite(v).all() for v in extra.values()) or (extra['row_norms'] <= 0).any()
                or (extra['lambdas'] <= 0).any() or (extra['phi'] < 1e-12).any()):
            raise ValueError('ARD requires positive retained norms/lambda and original trainingphi>=1e-12')
    metadata = {key:hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for key,value in extra.items()}
    return dict(basis=H, offsets=offsets, encoder_kind=encoder_kind, **extra), dict(record, actual_dimension=int(K.sum()),
        inherited_rank_metadata_is_not_selection_rule=True, artifact_absolute=str(path),
        encoder_parameter_sha256=metadata)


def ard_map_encode(X, unit_H, row_norms, lambdas, phi, observed=None):
    """Pure CPU math, no CUDA environment mutation and no model imports.

    Same frozen raw-space augmented NNLS as solver_repair. Hraw=unit_H*norms;
    solve raw>=0 with penalty phi/lambda, then return canonical C=raw*norms.
    KKT is checked for the phi-scaled objective at the original 1e-7 tolerance.
    """
    import numpy as np
    from scipy.optimize import nnls
    from threadpoolctl import threadpool_limits
    X, U = np.asarray(X,dtype=np.float64), np.asarray(unit_H,dtype=np.float64)
    norms, lam = np.asarray(row_norms,dtype=np.float64), np.asarray(lambdas,dtype=np.float64)
    if X.ndim != 2 or U.ndim != 2 or X.shape[1] != U.shape[1]:
        raise ValueError('MAP input dimensions')
    k = len(U)
    if (norms.shape != (k,) or lam.shape != (k,) or not np.isfinite(X).all() or not np.isfinite(U).all()
            or (X < 0).any() or (U < 0).any() or not np.isfinite(norms).all() or (norms < 0).any()
            or not np.isfinite(lam).all() or (lam <= 0).any() or not math.isfinite(phi) or phi < 1e-12):
        raise ValueError('invalid MAP factors, lambda or frozen phi')
    unorm = np.linalg.norm(U,axis=1)
    if not np.all((unorm == 0) | np.isclose(unorm,1,atol=1e-12,rtol=1e-12)):
        raise ValueError('MAP requires canonical basis')
    if np.any((norms == 0) != (unorm == 0)):
        raise ValueError('zero raw/canonical rows disagree')
    cols = np.arange(X.shape[1]) if observed is None else np.asarray(observed)
    if cols.ndim != 1 or cols.dtype.kind not in 'iu' or len(np.unique(cols)) != len(cols) or (cols < 0).any() or (cols >= X.shape[1]).any():
        raise ValueError('invalid MAP observed columns')
    started = time.perf_counter()
    Hobs = (U*norms[:,None])[:,cols]
    F = X[:,cols]
    if k == 0 or len(X) == 0:
        raw = np.zeros((len(X),k),dtype=np.float64)
    else:
        A = np.concatenate([Hobs.T,np.diag(np.sqrt(phi/lam))],axis=0)
        with threadpool_limits(limits=1):
            raw = np.vstack([nnls(A,np.concatenate([row,np.zeros(k)]),maxiter=max(240,30*k))[0] for row in F])
    C = raw*norms[None,:]
    with threadpool_limits(limits=1):
        residual = raw@Hobs-F
        grad = residual@Hobs.T+phi*raw/lam[None,:]
        dual = float(np.max(np.maximum(-grad,0),initial=0))
        comp = float(np.max(np.abs(raw*grad),initial=0))
        kkt = max(dual,comp)
        pg = np.where(raw > 0,grad,np.minimum(grad,0))
        objective = float(np.sum(residual*residual)/(2*phi)+.5*np.sum(raw*raw/lam[None,:]))
    receipt = dict(method='ARD_fixed_H_halfnormal_MAP_augmented_NNLS',
        converged=bool(np.isfinite(C).all() and math.isfinite(objective) and kkt <= 1e-7),
        kkt_max=kkt,kkt_scaled_objective=kkt,kkt_limit=1e-7,
        kkt_original_objective_dual=dual/phi,kkt_original_objective_complementarity=comp/phi,
        projected_gradient_infinity_scaled=float(np.max(np.abs(pg),initial=0)),
        objective=objective,reconstruction_sse=float(np.sum(residual*residual)),
        prior_quadratic=float(.5*np.sum(raw*raw/lam[None,:])),phi=float(phi),
        observed_columns=cols.tolist(),original_feature_width=X.shape[1],K=k,zero_rank=k == 0,
        raw_coefficients=raw,row_norms_sha256=hashlib.sha256(np.ascontiguousarray(norms).tobytes()).hexdigest(),
        lambdas_sha256=hashlib.sha256(np.ascontiguousarray(lam).tobytes()).hexdigest(),
        basis_sha256=hashlib.sha256(np.ascontiguousarray(U).tobytes()).hexdigest(),
        maxiter=max(240,30*k),seconds=time.perf_counter()-started)
    if not receipt['converged']:
        raise FloatingPointError('regularized MAP NNLS KKT failure '+str(kkt))
    return C, receipt


def install_dictionary_encoder(record, basis):
    """One process, one frozen method; only ARD installs its MAP callable.

    Replaces the imported symbol in model_variants in this process only. No old
    source is edited, no new parameters/modules are added, and the same callable
    remains active when the frozen sampler reloads the final readout.
    """
    global _INSTALLED_ENCODER
    import numpy as np
    import model_variants
    kind = basis.get('encoder_kind','nnls')
    if (kind == 'ard_map') != (record['method'] == 'ard'):
        raise ValueError('ARD requires MAP and other methods require original NNLS')
    if kind == 'nnls':
        if _INSTALLED_ENCODER is not None or model_variants.encode_typed.__module__ != 'role_dictionary':
            raise ValueError('NNLS process encoder unexpectedly patched')
        return dict(method=record['method'],encoder_kind='nnls',patched=False)
    if kind != 'ard_map' or record['method'] != 'ard' or _INSTALLED_ENCODER is not None:
        raise ValueError('ARD encoder must install once in its own process')
    frozen = {key:np.array(basis[key],copy=True) for key in ('basis','offsets','row_norms','lambdas','phi')}
    for value in frozen.values():
        value.flags.writeable = False

    def typed_map(features, matrix, offsets, types):
        X = np.asarray(features,dtype=np.float64)
        types = np.asarray(types,dtype=np.int64)
        if (not np.array_equal(matrix,frozen['basis']) or not np.array_equal(offsets,frozen['offsets'])
                or types.shape != (len(X),) or (types < 0).any() or (types >= 4).any()):
            raise ValueError('ARD process dictionary identity/type mismatch')
        out = np.zeros((len(X),len(matrix)),dtype=np.float64)
        per_type = {}
        for typ,(lo,hi) in enumerate(zip(frozen['offsets'][:-1],frozen['offsets'][1:])):
            ids = np.flatnonzero(types == typ)
            coefficients, receipt = ard_map_encode(X[ids],frozen['basis'][lo:hi],
                frozen['row_norms'][lo:hi],frozen['lambdas'][lo:hi],float(frozen['phi'][typ]))
            out[ids,lo:hi] = coefficients
            # Keep receipt JSON-safe in the live context; exact raw coefficient
            # values are recoverable from C/norms and are not network inputs.
            receipt['raw_coefficients_sha256'] = hashlib.sha256(np.ascontiguousarray(receipt.pop('raw_coefficients')).tobytes()).hexdigest()
            per_type[str(typ)] = receipt
        return out, dict(per_type=per_type,kkt_max=max(r['kkt_max'] for r in per_type.values()),
                         offsets=frozen['offsets'].tolist(),
                         layout='type-major canonical MAP coefficients C=raw*norms; other types zero; no row normalization',
                         encoder_kind='ard_map',artifact_sha256=record['artifact_sha256'])

    model_variants.encode_typed = typed_map
    _INSTALLED_ENCODER = record['artifact_sha256']
    return dict(method='ard',encoder_kind='ard_map',patched=True,
                artifact_sha256=record['artifact_sha256'],dtype='float64 CPU coefficients, originalfloat32 network transfer',
                normalization='rawMAP times saved row_norms',new_network_modules=0,
                persists_through_training_and_same_process_sampling=True)


def scientific_imports():
    """Import the audited package only when the authorized worker requests it."""
    for path in (PACKAGE, SCI):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    import torch
    from model_variants import StructuralDenoiser, copy_common
    from experiment_contract import reset_training_streams
    return torch, StructuralDenoiser, copy_common, reset_training_streams


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def validate_late_model(model):
    if hasattr(model, 'cn_edge_projection') or model.joint_head[0].in_features != 776 or model.arm != 'E':
        raise ValueError('model is not frozen E-lateCN')


def baseline_initial(case, cfg, seed, basis, bank_record=None):
    """Explicitly construct an untrained baseline E, optionally from old theta0.

    A trained checkpoint is forbidden. No optimizer state is inherited. Caller
    must reset training streams AFTER creation, create empty AdamW/EMA, and save
    this state before the first forward.
    """
    verify_configuration(cfg, seed)
    torch, constructor, copy_common, _ = scientific_imports()
    if bank_record is not None:
        path = Path(bank_record['file'])
        if sha(path) != bank_record['sha256']:
            raise ValueError('untrained bank hash mismatch')
        bank = torch.load(path, map_location='cpu')
        if (bank.get('updates') != 0 or bank.get('train_seed') != seed or bank.get('dataset') != DATASET
                or bank.get('arm') != 'E' or bank.get('config') != cfg):
            raise ValueError('only exact zero-update baseline initial bank is allowed')
        model = constructor(case, cfg, 'E', basis)
        # Preserve supplied legacy basis identity before loading its old bank.
        before = {key: tensor_sha(model.state_dict()[key]) for key in ('role_basis', 'role_offsets')}
        if any(tensor_sha(bank['model'][key]) != value for key, value in before.items()):
            raise ValueError('baseline bank dictionary differs from supplied baseline dictionary')
        model.load_state_dict(bank['model'], strict=True)
        actual = {key: tensor_sha(value) for key, value in model.state_dict().items()}
        if actual != bank_record['tensor_sha256']:
            raise ValueError('actual untrained bank tensor hashes disagree')
        proof = dict(source='archived_zero_update_E_bank', bank_file=str(path), bank_sha256=bank_record['sha256'],
                     tensor_sha256=actual, historical_initialization=bank.get('initialization'))
    else:
        # Same original initialization construction for the predeclared third seed.
        torch.manual_seed(seed)
        anchor = constructor(case, cfg, 'A')
        module_seed = seed32(seed, dataset=DATASET, arm='E', purpose='new_module_initialization')
        torch.manual_seed(module_seed)
        model = constructor(case, cfg, 'E', basis)
        copied = copy_common(anchor, model)
        proof = dict(source='fresh_original_initialization_rule', module_seed=module_seed,
                     copied_from_A_anchor=copied,
                     tensor_sha256={key:tensor_sha(value) for key,value in model.state_dict().items()})
        del anchor
    validate_late_model(model)
    if torch.count_nonzero(model.refex_projection.weight) or torch.count_nonzero(model.role_projection.weight):
        raise ValueError('baseline initial structural projections must be exactly zero')
    return model, dict(proof, updates=0, train_seed=seed, dataset=DATASET,
                       optimizer_inherited=False, trained_weights_used=False,
                       training_stream_reset_still_required=True)


def candidate_initial(case, cfg, seed, method, basis, baseline_state):
    """Copy common INITIAL parameters, excluding all role-specific state.

    baseline_state must be an authenticated zero-update payload with dataset,
    train_seed, config and model. Different role columns are never aligned by
    index, even when their dimensions happen to match. Role projection remains
    exactly zero under the original model's initialization rule.
    """
    if method not in METHODS-{'baseline'}:
        raise ValueError('unknown candidate method')
    verify_configuration(cfg, seed)
    if (baseline_state.get('updates') != 0 or baseline_state.get('train_seed') != seed
            or baseline_state.get('dataset') != DATASET or baseline_state.get('config') != cfg):
        raise ValueError('candidate requires authenticated zero-update baseline identity')
    torch, constructor, _, _ = scientific_imports()
    module_seed = seed32(seed, dataset=DATASET, method=method, purpose='adaptive_role_repair_module_initialization')
    torch.manual_seed(module_seed)
    model = constructor(case, cfg, 'E', basis)
    validate_late_model(model)
    source, target = baseline_state['model'], model.state_dict()
    if set(source)-ROLE_KEYS != set(target)-ROLE_KEYS:
        raise ValueError('common parameter set changed')
    copied = []
    for name in sorted(set(target)-ROLE_KEYS):
        if source[name].shape != target[name].shape or source[name].dtype != target[name].dtype:
            raise ValueError('common parameter shape/dtype changed: '+name)
        target[name].copy_(source[name])
        copied.append(name)
    model.load_state_dict(target, strict=True)
    if any(tensor_sha(model.state_dict()[name]) != tensor_sha(source[name]) for name in copied):
        raise ValueError('copied common tensors are not byte-identical')
    if torch.count_nonzero(model.role_projection.weight):
        raise ValueError('new role projection must retain exactly-zero original initialization')
    proof = dict(dataset=DATASET, train_seed=seed, method=method, updates=0, module_seed=module_seed,
        common_tensors=copied, common_tensor_count=len(copied), all_common_exact=True,
        excluded_role_keys=sorted(ROLE_KEYS), role_columns_semantically_unpaired=True,
        role_projection_initialization='exact zeros, original frozen rule',
        tensor_sha256={key:tensor_sha(value) for key,value in model.state_dict().items()},
        trained_checkpoint_dictionary_swap=False, optimizer_inherited=False,
        training_stream_reset_still_required=True)
    return model, proof


def reset_training(seed, device):
    """Call only after constructing/loading the initial model and optimizer."""
    _, _, _, reset = scientific_imports()
    return reset(seed, DATASET, device)


def sample_sources(train_seed):
    """Read-only mapping of the four predeclared baseline sample identities."""
    if train_seed not in (2026110221, 2026110222):
        return []
    result = []
    for sample_seed in SAMPLE_SEEDS:
        root = BASELINE if sample_seed in SAMPLE_SEEDS[:2] else EXPANSION
        folder = root/'run'/f'{DATASET}_{train_seed}_E_late'/'ema_0.99'/f'sample_{sample_seed}'
        result.append(dict(dataset=DATASET, train_seed=train_seed, sample_seed=sample_seed,
                           graph=str(folder/'graph.npz'), summary=str(folder/'summary.json')))
    return result
