"""Read-only transition observer, inserted into a hash-frozen original sampler."""
import inspect
from common import *
import dihug.core.gpu_role_guided_edit as original

def instrumented():
    source=inspect.getsource(original.sample_gpu_role_guided_edit_diffusion)
    replacements=[
        ('    decoder_temperature: float = 1.0,\n','    decoder_temperature: float = 1.0,\n    observer=None,\n'),
        ('        for completed in range(1, int(steps) + 1):\n',
         '        if observer is not None and not role_phase:\n            observer(state.to_graph(initial_graph), target_rows.copy(), 0)\n        for completed in range(1, int(steps) + 1):\n'),
        ('            if role_phase:\n                plateau_observations += 1\n',
         '            if observer is not None and transition_step == completed and transition_reason == "threshold":\n                observer(state.to_graph(initial_graph), target_rows.copy(), completed)\n            if role_phase:\n                plateau_observations += 1\n')]
    for old,new in replacements:
        assert source.count(old)==1
        source=source.replace(old,new)
    source='from __future__ import annotations\n'+source
    ns=dict(vars(original));exec(compile(source,'<typed_edit_transition_observer>','exec'),ns)
    return ns['sample_gpu_role_guided_edit_diffusion'],source

sample,source=instrumented()
