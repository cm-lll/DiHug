import runpy
from pathlib import Path
def test_auxiliary_imports_do_not_resolve_legacy_panel():
    here=Path(__file__).resolve().parent
    for n in ('reference_cost.py','high_precision.py'):
        code=runpy.run_path(str(here/n),run_name='test_import_only')
        assert callable(code['main'])
        assert 'role_logscale_reader_v1' in code['scaling_reference'].__file__
