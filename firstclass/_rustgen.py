"""`Cls.rust_source()`: emit `#[repr(C)]` Rust definitions for a firstclass class."""
from __future__ import annotations

import re

from . import _specs as S

_KEYWORDS = {
    "as", "break", "const", "continue", "else", "enum", "extern", "false", "fn", "for", "if", "impl", "in",
    "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return", "static", "struct", "trait", "true",
    "type", "unsafe", "use", "where", "while", "async", "await", "dyn", "abstract", "become", "box", "do",
    "final", "macro", "override", "priv", "typeof", "unsized", "virtual", "yield", "try",
}
_NO_RAW = {"self", "Self", "super", "crate"}  # cannot be written as r#name


def _ident(name: str) -> str:
    if name in _NO_RAW:
        return name + "_"
    return f"r#{name}" if name in _KEYWORDS else name


def _scalar(spec: S.Scalar) -> str:
    m = re.fullmatch(r"(u?)int(\d+)_t", spec.c_name)
    if m:
        return ("u" if m.group(1) else "i") + m.group(2)
    return {"float": "f32", "double": "f64", "bool": "bool"}[spec.c_name]


def rust_type(spec) -> str:
    if isinstance(spec, S.Pointer):
        return "*mut c_void"
    if isinstance(spec, S.Scalar):
        return _scalar(spec)
    if isinstance(spec, (S.CharArray, S.ByteArray)):
        return f"[u8; {spec.n}]"
    if isinstance(spec, S.WCharArray):
        return f"[u16; {spec.n}]"
    if isinstance(spec, S.ArrayOf):
        return f"[{rust_type(spec.elem)}; {spec.n}]"
    if isinstance(spec, S.Nested):
        return spec.schema.cls.__name__
    if isinstance(spec, S.EnumSpec):
        return spec.cls.__name__
    raise TypeError(f"no Rust type for {type(spec).__name__}")


def _enum_definition(spec: S.EnumSpec) -> str:
    lines = [f"#[repr({_scalar(spec.storage)})]", "#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
             f"pub enum {spec.cls.__name__} {{"]
    taken = set()
    for m in spec.cls:
        if m.value in taken:  # Rust rejects duplicate discriminants, so aliases become comments
            lines.append(f"    // {m.name} = {int(m.value)} (alias)")
        else:
            taken.add(m.value)
            lines.append(f"    {m.name} = {int(m.value)},")
    lines.append("}")
    return "\n".join(lines)


def rust_source(schema) -> str:
    schema.require_binary()
    out, seen = [], set()

    def emit(sc):
        if sc.cls in seen:
            return
        seen.add(sc.cls)
        for dep in sc._nested_schemas():
            emit(dep)
        for _, spec in sc.fields:
            while isinstance(spec, S.ArrayOf):
                spec = spec.elem
            if isinstance(spec, S.EnumSpec) and spec.cls not in seen:
                seen.add(spec.cls)
                out.append(_enum_definition(spec))
        items, _, _ = sc.layout()
        lines = [f"// byteorder: {sc.byteorder}", "#[repr(C)]" if sc.aligned else "#[repr(C, packed)]",
                 "#[derive(Clone, Copy)]", f"pub struct {sc.cls.__name__} {{"]
        groups = 0
        for item in items:
            if item[0] == "field":
                f, spec = sc.fields[item[1]]
                lines.append(f"    pub {_ident(f.name)}: {rust_type(spec)},")
            else:  # Rust has no bitfields: one integer holds the whole group
                _, storage, _, members = item
                names = ", ".join(f"{sc.fields[i][0].name}: {bits} bits" for i, _, bits in members)
                lines.append(f"    pub bits_{groups}: {_scalar(storage)}, // {names}")
                groups += 1
        lines.append("}")
        out.append("\n".join(lines))

    emit(schema)
    body = "\n\n".join(out) + "\n"
    return ("use std::os::raw::c_void;\n\n" if "c_void" in body else "") + body
