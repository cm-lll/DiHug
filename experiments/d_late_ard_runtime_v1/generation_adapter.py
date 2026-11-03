"""Only D/A routing is new; frozen E/ARD science is reused verbatim."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'experiments/adaptive_role_repair_v1/generation_adapter.py'
spec=importlib.util.spec_from_file_location('frozen_role_adapter',path)
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
METHODS=frozenset(('A','D','baseline','ard'))
SAMPLE_SEEDS=frozen.SAMPLE_SEEDS
BASELINE=frozen.BASELINE
canonical_configuration=frozen.canonical_configuration
verify_configuration=frozen.verify_configuration
reset_training=frozen.reset_training
scientific_imports=frozen.scientific_imports
def baseline_folder(seed):
    if seed in (2026110221,2026110222):return BASELINE/'run'/f'PubMed_{seed}_E_late'
    return ROOT/'results/adaptive_role_repair_20261103T032520/generation/run'/f'PubMed_{seed}_baseline'
def load_dictionary(record):
    if record['method'] in ('A','D'):
        if record != dict(method=record['method'],dataset='PubMed',role_branch=False,status='not_applicable',frozen=True):
            raise ValueError('Unexpected no-role dictionary metadata')
        return None,record
    return frozen.load_dictionary(record)
def install_dictionary_encoder(record,basis):
    if record['method'] in ('A','D'):
        if basis is not None:raise ValueError('A/D cannot contain a role dictionary')
        return dict(method=record['method'],encoder_kind='none',patched=False)
    return frozen.install_dictionary_encoder(record,basis)
