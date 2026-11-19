"""Imports only the frozen original typed-edit implementation."""
import os,sys,json,hashlib,time,math,copy
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'results/typed_edit_complexity_20261116'
FROZEN=(Path(__file__).resolve().parents[2] / 'releases/typed_edit_20261119/frozen_r2')
sys.path.insert(0,str(FROZEN/'scripts'))
import r2_common as c
import numpy as np,torch
from dihug.core.graph import TypedGraph,FamilySpec
from dihug.core.sampling_session import graph_payload,restore_graph
from dihug.core.mixed_edit_diffusion import MarginalGraphPrior,mixed_edit_feature_width
from dihug.core.role_guided_edit_diffusion import ExchangeableRoleDegreePrior
from dihug.core.cardinality_conditioned_edit import FamilyCountCondition
def read(p):return json.loads(Path(p).read_text())
def put(p,x):c.atomic(p,x)
def sha(p):return c.sha(p)
def setup():
    import tempfile
    assert Path(tempfile.gettempdir()).resolve()==ROOT/'.runtime_tmp'
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def graph(ds):return restore_graph(read(RUN/'data'/ds/'graph.json'))
def architecture(width,arm):
    if arm=='M':return 256,3
    if arm=='F':return 64,1
    # Include the affine LayerNorm (2h parameters) after every hidden layer.
    target=(width+1)*256+2*(256+1)*256+257+3*2*256
    return max(1,round((target-1)/(width+4))),1
def model(width,arm,seed,device='cpu'):
    torch.manual_seed(seed);return c.SwitchDenoiserMLP(width,*architecture(width,arm),.1).to(device)
