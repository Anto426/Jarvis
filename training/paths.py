from pathlib import Path
import os
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
PATHS_CONFIG = ROOT_DIR / "config" / "paths.yaml"


def resolve_project_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def load_paths():
    with PATHS_CONFIG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["paths"]


def get_path(name, create=False):
    paths = load_paths()
    env_name = f"JARVIS_PATH_{name.upper()}"
    path = resolve_project_path(os.environ.get(env_name, paths[name]))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def configure_cache_env():
    paths = load_paths()
    cache_dir = resolve_project_path(paths.get("cache_dir", ".cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    defaults = {
        "HF_HOME": paths.get("hf_home", str(cache_dir / "huggingface")),
        "HF_DATASETS_CACHE": paths.get("datasets_cache_dir", str(cache_dir / "huggingface" / "datasets")),
        "TORCH_HOME": paths.get("torch_home", str(cache_dir / "torch")),
        "XDG_CACHE_HOME": str(cache_dir),
    }

    for key, value in defaults.items():
        resolved = resolve_project_path(value)
        resolved.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(resolved))

    return cache_dir
