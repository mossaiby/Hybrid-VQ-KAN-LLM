"""Config loading: JSON files with `inherit` (str or list, paths relative to the file),
deep-merged in order, then CLI overrides of the form  key.sub=value  (value parsed as JSON)."""
import copy
import json
from pathlib import Path


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k.startswith("_"):  # "_comment" keys are documentation only
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    inherit = raw.pop("inherit", None)
    cfg = {}
    if inherit:
        for p in ([inherit] if isinstance(inherit, str) else inherit):
            cfg = deep_merge(cfg, _load((path.parent / p).resolve()))
    return deep_merge(cfg, raw)


def set_by_path(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path, overrides=()):
    cfg = _load(Path(path).resolve())
    for ov in overrides or ():
        key, _, val = ov.partition("=")
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass  # keep as plain string
        set_by_path(cfg, key.strip(), val)
    return cfg


def ramp(progress, spec):
    """Linear ramp from spec['start'] to spec['end'] between ramp_start and ramp_end (fractions of training)."""
    a, b = spec.get("ramp_start", 0.0), spec.get("ramp_end", 0.0)
    s, e = spec.get("start", 0.0), spec.get("end", 0.0)
    if progress <= a:
        return s
    if progress >= b or b <= a:
        return e
    return s + (e - s) * (progress - a) / (b - a)
