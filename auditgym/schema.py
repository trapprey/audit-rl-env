import json
import re
from functools import lru_cache
from pathlib import Path

SCHEMA_DIR = Path(__file__).parent / "schemas"

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


@lru_cache(maxsize=None)
def load(name):
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text())


def names():
    return sorted(p.name[: -len(".schema.json")] for p in SCHEMA_DIR.glob("*.schema.json"))


def validate(instance, name):
    errors = []
    _check(instance, load(name), "$", errors)
    return errors


def _resolve(ref):
    if not ref.endswith(".schema.json"):
        raise ValueError(f"unsupported $ref {ref}")
    return load(ref[: -len(".schema.json")])


def _check(value, schema, path, errors):
    if len(errors) > 50:
        return
    if "$ref" in schema:
        _check(value, _resolve(schema["$ref"]), path, errors)
        return
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}")
        return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
        return
    kinds = schema.get("type")
    if kinds is not None:
        kinds = [kinds] if isinstance(kinds, str) else kinds
        if not any(_TYPES[k](value) for k in kinds):
            errors.append(f"{path}: expected {kinds}, got {type(value).__name__}")
            return
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: too long")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: {value!r} does not match {schema['pattern']}")
    if _TYPES["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: too many items")
        if "items" in schema:
            for index, item in enumerate(value):
                _check(item, schema["items"], f"{path}[{index}]", errors)
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing {key}")
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                _check(item, properties[key], f"{path}.{key}", errors)
            elif extra is False:
                errors.append(f"{path}: unexpected key {key}")
            elif isinstance(extra, dict):
                _check(item, extra, f"{path}.{key}", errors)
