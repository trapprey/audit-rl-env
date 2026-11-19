import copy
import hashlib
import json
from pathlib import Path

from . import spec

CONFIG_DIR = Path(__file__).parent / "configs"


class ConfigError(ValueError):
    pass


def _load(name):
    path = CONFIG_DIR / f"{name}.json"
    if not path.is_file():
        raise ConfigError(f"unknown preset {name!r}")
    return json.loads(path.read_text())


def preset_names():
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json") if p.stem != "default")


def _merge(base, extra, path=""):
    for key, value in extra.items():
        if key not in base:
            raise ConfigError(f"unknown config key {path}{key}")
        if isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value, f"{path}{key}.")
        else:
            base[key] = copy.deepcopy(value)
    return base


def _range(value, name, low=0, high=10_000):
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an int or [min, max]")
    if isinstance(value, int):
        value = [value, value]
    if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in value)):
        raise ConfigError(f"{name} must be an int or [min, max]")
    if not (low <= value[0] <= value[1] <= high):
        raise ConfigError(f"{name} is out of range")
    return value


def resolve(config=None):
    config = copy.deepcopy(config or {})
    if not isinstance(config, dict):
        raise ConfigError("config must be an object")
    cfg = _load("default")
    preset = config.pop("preset", None)
    if preset is not None:
        _merge(cfg, _load(preset))
    _merge(cfg, config)
    cfg["preset"] = preset
    if cfg["period"] not in ("month", "quarter"):
        raise ConfigError("period must be month or quarter")
    cfg["customers"] = _range(cfg["customers"], "customers", 3, 60)
    cfg["invoices"] = _range(cfg["invoices"], "invoices", 10, 400)
    cfg["late_invoices"] = _range(cfg["late_invoices"], "late_invoices", 0, 30)
    cfg["foreign_currencies"] = _range(cfg["foreign_currencies"], "foreign_currencies", 1, 4)
    share = cfg["cross_currency_share"]
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 <= share <= 1:
        raise ConfigError("cross_currency_share must be within [0, 1]")
    for kind in spec.FINDING_TYPES:
        value = cfg["errors"][kind]
        _range(value, f"errors.{kind}", 0, spec.MAX_ISSUES)
    for name in spec.INTERACTIONS:
        _range(cfg["interactions"][name], f"interactions.{name}", 0, 3)
    for key, value in cfg["distractors"].items():
        if not isinstance(value, bool):
            raise ConfigError(f"distractors.{key} must be boolean")
    if cfg["ledger_order"] not in ("chronological", "shuffled"):
        raise ConfigError("ledger_order must be chronological or shuffled")
    ps = cfg["presentation_seed"]
    if ps is not None and (isinstance(ps, bool) or not isinstance(ps, int)):
        raise ConfigError("presentation_seed must be an int or null")
    return cfg


def digest(cfg):
    body = {k: v for k, v in cfg.items() if k not in ("presentation_seed", "preset", "ledger_order")}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
