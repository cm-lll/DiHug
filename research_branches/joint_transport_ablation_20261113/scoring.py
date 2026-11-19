"""Isolated wiring: new reward, unchanged frozen production model/runtime."""
import os,sys,importlib.util
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from transport import TransportScore,ROOT
spec=importlib.util.spec_from_file_location('legacy_ablation_scoring',ROOT/'research_branches/paper_ablation_20261112/scoring.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
s,base,w,np,torch=legacy.s,legacy.base,legacy.w,legacy.np,legacy.torch
LegacyTypedScore=legacy.TypedScore
GenericScore=legacy.GenericScore
TypedScore=TransportScore
case_from=legacy.case_from
