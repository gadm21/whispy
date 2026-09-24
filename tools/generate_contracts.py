#!/usr/bin/env python3
"""Generate TypeScript/Dart contract types from the canonical JSON schemas.

The JSON schemas under ``contracts/schemas/`` are the single semantic
source. This tool emits equivalent type declarations so TypeScript
(ResearchPortal/website) and Dart (thoth-app) clients never hand-maintain
divergent definitions.

Usage::

    python tools/generate_contracts.py ts   > dist/contracts.ts
    python tools/generate_contracts.py dart > dist/contracts.dart
    python tools/generate_contracts.py validate contracts/fixtures/*.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / "contracts" / "schemas"

TS_PRIMITIVES = {
    "string": "string", "number": "number", "integer": "number",
    "boolean": "boolean", "object": "Record<string, unknown>",
    "array": "unknown[]",
}
DART_PRIMITIVES = {
    "string": "String", "number": "double", "integer": "int",
    "boolean": "bool", "object": "Map<String, dynamic>",
    "array": "List<dynamic>",
}


def _pascal(name: str) -> str:
    return "".join(p.capitalize() for p in re.split(r"[-_]", name))


def _camel(name: str) -> str:
    p = _pascal(name)
    return p[:1].lower() + p[1:]


def _types(prop: dict) -> list:
    t = prop.get("type")
    if isinstance(t, list):
        return t
    return [t] if t else []


def _ts_type(prop: dict) -> str:
    if "$ref" in prop:
        return _pascal(Path(prop["$ref"]).stem.replace(".schema", ""))
    types = _types(prop)
    if not types:
        return "unknown"
    parts = []
    for t in types:
        if t == "null":
            parts.append("null")
        elif t == "array":
            item = prop.get("items") or {}
            parts.append(f"{_ts_type(item)}[]")
        elif t == "object" and "additionalProperties" in prop:
            ap = prop["additionalProperties"]
            inner = _ts_type(ap) if isinstance(ap, dict) else "unknown"
            parts.append(f"Record<string, {inner}>")
        else:
            parts.append(TS_PRIMITIVES.get(t, "unknown"))
    return " | ".join(dict.fromkeys(parts))


def _dart_type(prop: dict) -> str:
    if "$ref" in prop:
        return _pascal(Path(prop["$ref"]).stem.replace(".schema", ""))
    types = _types(prop)
    nullable = "null" in types
    core = [t for t in types if t != "null"] or ["object"]
    t = core[0]
    if t == "array":
        item = prop.get("items") or {}
        base = f"List<{_dart_type(item)}>"
    elif t == "object" and isinstance(prop.get("additionalProperties"), dict):
        base = f"Map<String, {_dart_type(prop['additionalProperties'])}>"
    else:
        base = DART_PRIMITIVES.get(t, "dynamic")
    return base + ("?" if nullable else "")


def emit_ts(schemas: dict) -> str:
    out = ["// GENERATED from contracts/schemas/*.schema.json — do not edit.",
           "// Regenerate: python tools/generate_contracts.py ts", ""]
    for name, schema in schemas.items():
        cls = _pascal(name)
        desc = schema.get("description")
        if desc:
            out.append(f"/** {desc} */")
        out.append(f"export interface {cls} {{")
        required = set(schema.get("required") or [])
        for prop, spec in (schema.get("properties") or {}).items():
            opt = "" if prop in required else "?"
            comment = f"  /** {spec['description']} */" if spec.get("description") else None
            if comment:
                out.append(comment)
            out.append(f"  {prop}{opt}: {_ts_type(spec)};")
        out.append("}")
        out.append("")
    return "\n".join(out)


def emit_dart(schemas: dict) -> str:
    out = ["// GENERATED from contracts/schemas/*.schema.json — do not edit.",
           "// Regenerate: python tools/generate_contracts.py dart", ""]
    for name, schema in schemas.items():
        cls = _pascal(name)
        desc = schema.get("description")
        if desc:
            out.append(f"/// {desc}")
        out.append(f"class {cls} {{")
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        fields = []
        for prop, spec in props.items():
            dt = _dart_type(spec)
            if prop in required and not dt.endswith("?"):
                fields.append(f"  required this.{_camel(prop)},")
                out.append(f"  final {dt} {_camel(prop)};")
            else:
                fields.append(f"  this.{_camel(prop)},")
                out.append(f"  final {dt}{'' if dt.endswith('?') or dt == 'dynamic' else '?'} {_camel(prop)};")
        out.append("")
        out.append(f"  {cls}({{")
        out.extend(fields)
        out.append("  });")
        out.append("")
        # fromJson / toJson
        out.append(f"  factory {cls}.fromJson(Map<String, dynamic> j) => {cls}(")
        for prop, spec in props.items():
            c = _camel(prop)
            optional = prop not in required
            dt = _dart_type(spec)
            if optional and not dt.endswith("?") and dt != "dynamic":
                dt += "?"
            nullable = dt.endswith("?") or dt == "dynamic"
            base = dt.rstrip("?")
            if base.startswith("Map<String,") or base.startswith("List<"):
                cast = f" as {base}" + ("?" if nullable else "")
                out.append(f"    {c}: j['{prop}']{cast},")
            elif base == "String":
                out.append(f"    {c}: j['{prop}'] as {dt},")
            elif base == "double":
                if nullable:
                    out.append(f"    {c}: j['{prop}'] != null ? (j['{prop}'] as num).toDouble() : null,")
                else:
                    out.append(f"    {c}: (j['{prop}'] as num).toDouble(),")
            elif base == "int":
                if nullable:
                    out.append(f"    {c}: j['{prop}'] != null ? (j['{prop}'] as num).toInt() : null,")
                else:
                    out.append(f"    {c}: (j['{prop}'] as num).toInt(),")
            elif base == "bool":
                out.append(f"    {c}: j['{prop}'] as {dt},")
            else:
                out.append(f"    {c}: j['{prop}'],")
        out.append("  );")
        out.append("")
        out.append("  Map<String, dynamic> toJson() => {")
        for prop in props:
            out.append(f"    '{prop}': {_camel(prop)},")
        out.append("  };")
        out.append("}")
        out.append("")
    return "\n".join(out)


def _validate(instance: dict, schema: dict, path: str = "") -> list:
    """Minimal validator: required fields + declared primitive types."""
    errors = []
    for req in schema.get("required") or []:
        if req not in instance or instance[req] is None:
            errors.append(f"{path}{req}: missing required field")
    type_map = {"string": str, "number": (int, float), "integer": int,
                "boolean": bool, "object": dict, "array": list}
    for prop, spec in (schema.get("properties") or {}).items():
        if prop not in instance or instance[prop] is None:
            continue
        types = [t for t in _types(spec) if t != "null"]
        if types and not any(isinstance(instance[prop], type_map.get(t, object))
                             and not (t == "integer" and isinstance(instance[prop], bool))
                             for t in types):
            errors.append(f"{path}{prop}: expected {'|'.join(types)}")
        if "enum" in spec and instance[prop] not in spec["enum"]:
            errors.append(f"{path}{prop}: {instance[prop]!r} not in {spec['enum']}")
        if spec.get("const") and instance[prop] != spec["const"]:
            errors.append(f"{path}{prop}: expected const {spec['const']!r}")
    return errors


def main(argv: list) -> int:
    schemas = {}
    for f in sorted(SCHEMA_DIR.glob("*.schema.json")):
        schemas[f.stem.replace(".schema", "")] = json.loads(f.read_text())

    if not argv or argv[0] not in ("ts", "dart", "validate"):
        print(__doc__)
        return 2
    if argv[0] == "ts":
        print(emit_ts(schemas))
        return 0
    if argv[0] == "dart":
        print(emit_dart(schemas))
        return 0

    # validate <fixture.json ...> — fixture name must match a schema name.
    rc = 0
    for arg in argv[1:]:
        path = Path(arg)
        name = path.stem.split(".")[0]
        schema = schemas.get(name)
        if schema is None:
            print(f"{path}: no schema named {name!r}")
            rc = 1
            continue
        errors = _validate(json.loads(path.read_text()), schema)
        if errors:
            rc = 1
            for e in errors:
                print(f"{path}: {e}")
        else:
            print(f"{path}: OK")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
