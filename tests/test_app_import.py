import importlib.util
from pathlib import Path

def test_app_import():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    assert app_path.exists(), "app.py not found at repo root"

    spec = importlib.util.spec_from_file_location("app", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
