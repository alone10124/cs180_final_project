"""Loading configs/*.yaml. Every can constant comes from configs/can.yaml.

The configs folder is found from $SODACAN_CONFIGS, else next to the package (editable
install or a clone), else the current directory's configs/.
"""
import copy
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def config_dir():
    for candidate in (os.environ.get("SODACAN_CONFIGS"), REPO_ROOT / "configs", Path.cwd() / "configs"):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    raise FileNotFoundError("configs/ not found; set SODACAN_CONFIGS")


def load_yaml(name_or_path):
    path = Path(name_or_path)
    if not path.suffix:
        path = path.with_suffix(".yaml")
    if not path.exists():
        path = config_dir() / path.name
    return yaml.safe_load(path.read_text())


def merge(base, overrides):
    """Recursive dict update that returns a new dict."""
    out = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Can:
    radius: float
    height: float
    usable_z: tuple
    material: dict
    atlas: tuple = (192, 512)
    min_cos: float = 0.15
    extra: dict = field(default_factory=dict)

    @property
    def z_lo(self):
        return float(self.usable_z[0])

    @property
    def z_hi(self):
        return float(self.usable_z[1])


def load_can(path="can", **overrides):
    """Can geometry and material. Keyword overrides replace config entries (dicts merge)."""
    raw = merge(load_yaml(path), overrides)
    known = {"radius", "height", "usable_z", "material", "atlas", "min_cos"}
    return Can(
        radius=float(raw["radius"]),
        height=float(raw["height"]),
        usable_z=tuple(raw.get("usable_z", (0.0, raw["height"]))),
        material=dict(raw.get("material", {"model": "mirror"})),
        atlas=tuple(raw.get("atlas", (192, 512))),
        min_cos=float(raw.get("min_cos", 0.15)),
        extra={k: v for k, v in raw.items() if k not in known},
    )


def load_pipeline(path="pipeline", **overrides):
    return merge(load_yaml(path), overrides)
