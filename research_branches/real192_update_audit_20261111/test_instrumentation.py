"""CPU-only: observation insertion preserves the frozen scientific function."""
import ast, inspect, json, tempfile
from pathlib import Path
from types import SimpleNamespace
import worker

with tempfile.TemporaryDirectory() as path:
    tracker=SimpleNamespace(folder=Path(path))
    fn=worker.build_instrumented_update(tracker)
    tree=ast.parse((tracker.folder/'executed_update.py').read_text())
    class RemoveObservations(ast.NodeTransformer):
        def visit_Expr(self,node):
            value=node.value
            if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and isinstance(value.func.value,ast.Name) and value.func.value.id=='tracker':
                return None
            return self.generic_visit(node)
    original=ast.parse(inspect.getsource(worker.base.update))
    assert ast.dump(RemoveObservations().visit(tree),include_attributes=False)==ast.dump(original,include_attributes=False)
    x=worker.torch.tensor([1.,-2.]);assert worker.dot(x,x)==5.
    assert abs(worker.cosine(x,-x)+1.)<1e-14
    assert worker.cosine(x,worker.torch.zeros_like(x)) is None
    assert len(worker.PROBES)==7 and min(worker.PROBES)==2001 and max(worker.PROBES)==2050
worker.rt.save(worker.OUT/'cpu_checks.json',dict(scientific_AST_unchanged_after_removing_observation_calls=True,
                   vector_metrics_pass=True,selected_probe_updates=sorted(worker.PROBES),GPU_calls=0))
print('CPU instrumentation checks passed')
