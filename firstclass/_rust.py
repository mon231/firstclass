"""`from_rust`: parse simple `#[repr(C)]` structs and C-like enums out of Rust source.

Rust types are lowered to the same declarations the C header parser produces, so the class
building (layout, enums, arrays, skipping with a reason) is shared with `from_header`.
"""
from __future__ import annotations

import ctypes
import os
import re
from pathlib import Path

from ._header import (
    EnumDecl, FieldDecl, HeaderTypes, StructDecl, Tok, TypedefDecl, _assemble, _Ctx, _ExprError, _match, _Skip,
)

# --- tokenizer ---------------------------------------------------------------

_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<str>b?r?\#*"(?:\\.|[^\\"])*"\#*)
  | (?P<chr>b?'(?:\\.[^']*|[^\\'])')
  | (?P<life>'[A-Za-z_]\w*)
  | (?P<num>0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?)
    (?P<suffix>(?:[iu](?:8|16|32|64|128|size)|f32|f64))?
  | (?P<id>(?:r\#)?[A-Za-z_]\w*)
  | (?P<op>::|->|=>|<<=|>>=|\.\.=|\.\.\.|\.\.|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^~!<>=?:;,.(){}\[\]\#@$])
  | (?P<other>.)
""", re.X)


def _strip_comments(text: str) -> str:
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:  # block comments nest in Rust
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            out.append(" ")
        elif text[i] == "'" and (m := re.compile(r"'(?:\\.[^']*|[^\\'])'").match(text, i)):
            out.append(m.group())
            i = m.end()
        elif text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _number(text: str) -> str:
    text = text.replace("_", "")
    low = text.lower()
    if low.startswith(("0x", "0b")):
        return text
    if low.startswith("0o"):
        return "0" + text[2:] if len(text) > 2 else "0"
    return str(int(float(text))) if "." in text or "e" in low else str(int(text))


def _tokenize(text: str) -> list[Tok]:
    out = []
    for m in _TOKEN.finditer(_strip_comments(text)):
        if m.group("ws") or m.group("other"):
            continue
        if m.group("num") is not None:
            out.append(Tok("num", _number(m.group("num")) if "." not in m.group("num") else m.group("num")))
        elif m.group("str") is not None:
            out.append(Tok("str", m.group("str")))
        elif m.group("chr") is not None:
            out.append(Tok("chr", m.group("chr").lstrip("b")))
        elif m.group("life") is not None:
            out.append(Tok("life", m.group("life")))
        elif m.group("id") is not None:
            out.append(Tok("id", m.group("id").removeprefix("r#")))
        else:
            out.append(Tok("op", m.group("op")))
    return out


def _strip_casts(toks):
    """`X as u8` -> `X` (the value is what matters for sizes and discriminants)."""
    out, i = [], 0
    while i < len(toks):
        if toks[i].kind == "id" and toks[i].text == "as":
            i += 1
            while i < len(toks) and (toks[i].kind == "id" or (toks[i].kind == "op" and toks[i].text == "::")):
                i += 1
            continue
        out.append(toks[i])
        i += 1
    return out


def _split_commas(toks, angles=False):
    parts, cur, depth = [], [], 0
    for tok in toks:
        if tok.kind == "op":
            x = tok.text
            if x in ("(", "[", "{") or (angles and x == "<"):
                depth += 1
            elif x in (")", "]", "}") or (angles and x == ">"):
                depth -= 1
            elif angles and x == ">>":
                depth -= 2
            elif x == "," and depth == 0:
                parts.append(cur)
                cur = []
                continue
        cur.append(tok)
    if cur:
        parts.append(cur)
    return parts


def _is_op(toks, i, s):
    return 0 <= i < len(toks) and toks[i].kind == "op" and toks[i].text == s


# --- type lowering -------------------------------------------------------------

_PRIMITIVES = {
    "u8": ("uint8_t",), "u16": ("uint16_t",), "u32": ("uint32_t",), "u64": ("uint64_t",), "usize": ("size_t",),
    "i8": ("int8_t",), "i16": ("int16_t",), "i32": ("int32_t",), "i64": ("int64_t",), "isize": ("ssize_t",),
    "f32": ("float",), "f64": ("double",), "bool": ("bool",),
    "c_char": ("char",), "c_schar": ("signed", "char"), "c_uchar": ("unsigned", "char"),
    "c_short": ("short",), "c_ushort": ("unsigned", "short"), "c_int": ("int",), "c_uint": ("unsigned", "int"),
    "c_long": ("long",), "c_ulong": ("unsigned", "long"), "c_longlong": ("long", "long"),
    "c_ulonglong": ("unsigned", "long", "long"), "c_float": ("float",), "c_double": ("double",),
    "c_void": ("void",), "size_t": ("size_t",), "ssize_t": ("ssize_t",), "wchar_t": ("wchar_t",),
}
_UNSUPPORTED = {
    "char": "Rust char is a 4-byte Unicode scalar (use u32, or u16 units for UTF-16)",
    "u128": "128-bit integers unsupported", "i128": "128-bit integers unsupported",
    "str": "str is not repr(C)", "String": "String is not repr(C)", "Vec": "Vec is not repr(C)",
}


class _RustParser:
    def __init__(self, toks, ctx, assume_repr_c):
        self.t, self.i, self.ctx, self.assume = toks, 0, ctx, assume_repr_c

    # type -> (words, is_pointer, dims, next_index)
    def parse_type(self, t, i=0):
        if i >= len(t):
            raise _Skip("missing type")
        tok = t[i]
        if tok.kind == "op" and tok.text == "[":
            close = _match(t, i)
            inner = t[i + 1:close]
            depth, semi = 0, None
            for k, x in enumerate(inner):
                if x.kind == "op":
                    depth += x.text in ("[", "(", "{")
                    depth -= x.text in ("]", ")", "}")
                    if x.text == ";" and depth == 0:
                        semi = k
                        break
            if semi is None:
                raise _Skip("slices are not repr(C); use [T; N]")
            words, ptr, dims, _ = self.parse_type(inner[:semi], 0)
            try:
                n = self.ctx.eval(_strip_casts(inner[semi + 1:]))
            except _ExprError as e:
                raise _Skip(f"array length: {e}") from None
            return words, ptr, [n] + dims, close + 1
        if tok.kind == "op" and tok.text == "*":
            i += 1
            if i < len(t) and t[i].kind == "id" and t[i].text in ("const", "mut"):
                i += 1
            *_, i = self.parse_type(t, i)
            return ("void",), True, [], i
        if tok.kind == "op" and tok.text in ("&", "&&"):
            i += 1
            while i < len(t) and (t[i].kind == "life" or (t[i].kind == "id" and t[i].text == "mut")):
                i += 1
            *_, i = self.parse_type(t, i)
            return ("void",), True, [], i
        if tok.kind == "id" and tok.text in ("fn", "unsafe", "extern"):
            while i < len(t) and (t[i].kind == "str" or (t[i].kind == "id" and t[i].text in ("unsafe", "extern"))):
                i += 1
            if not (i < len(t) and t[i].kind == "id" and t[i].text == "fn"):
                raise _Skip("unsupported type")
            i += 1
            if _is_op(t, i, "("):
                i = _match(t, i) + 1
            if _is_op(t, i, "->"):
                *_, i = self.parse_type(t, i + 1)
            return ("void",), True, [], i
        if tok.kind == "op" and tok.text == "(":
            raise _Skip("tuples and () are not repr(C)")
        if tok.kind != "id":
            raise _Skip(f"unsupported type syntax near {tok.text!r}")
        j, name = i, tok.text
        while _is_op(t, j + 1, "::") and j + 2 < len(t) and t[j + 2].kind == "id":
            j += 2
            name = t[j].text
        j += 1
        if _is_op(t, j, "<"):
            if name not in ("Option", "Box", "NonNull"):
                raise _Skip(f"generic type {name}<..> unsupported")
            words, ptr, dims, k = self.parse_type(t, j + 1)
            if name == "Option" and not (ptr and not dims):
                raise _Skip("Option<T> is only supported for pointers and references")
            if _is_op(t, k, ">>"):
                t[k:k + 1] = [Tok("op", ">"), Tok("op", ">")]
            if not _is_op(t, k, ">"):
                raise _Skip("unbalanced generics")
            return ("void",), True, [], k + 1
        if name in _UNSUPPORTED:
            raise _Skip(_UNSUPPORTED[name])
        return _PRIMITIVES.get(name, (name,)), False, [], j

    # attributes and items
    def take_attrs(self):
        attrs, t = [], self.t
        while _is_op(t, self.i, "#"):
            j = self.i + 1 + _is_op(t, self.i + 1, "!")
            if not _is_op(t, j, "["):
                break
            close = _match(t, j)
            attrs.append(t[j + 1:close])
            self.i = close + 1
        return attrs

    def skip_item(self, j):
        t = self.t
        while j < len(t):
            if t[j].kind == "op":
                if t[j].text == ";":
                    return j + 1
                if t[j].text in ("(", "["):
                    j = _match(t, j)
                elif t[j].text == "{":
                    return _match(t, j) + 1
            j += 1
        return len(t)

    @staticmethod
    def repr_of(attrs):
        flags, packed, align = set(), None, None
        for a in attrs:
            if a and a[0].text == "repr" and len(a) > 2:
                for part in _split_commas(a[2:-1]):
                    word = part[0].text
                    num = int(part[2].text) if len(part) >= 4 and part[2].kind == "num" else None
                    if word == "packed":
                        packed = num or 1
                    elif word == "align":
                        align = num
                    else:
                        flags.add(word)
        return flags, packed, align

    @staticmethod
    def is_cfg_test(attrs):
        return any(a and a[0].text == "cfg" and any(x.text == "test" for x in a) and not any(x.text == "not" for x in a)
                   for a in attrs)

    def parse_all(self):
        t, n = self.t, len(self.t)
        while self.i < n:
            start = self.i
            attrs = self.take_attrs()
            if self.i >= n:
                break
            j = self.i
            if t[j].kind == "id" and t[j].text == "pub":
                j += 1
                if _is_op(t, j, "("):
                    j = _match(t, j) + 1
            kw = t[j].text if j < n and t[j].kind == "id" else None
            if self.is_cfg_test(attrs):
                self.i = self.skip_item(j)
            elif kw == "struct":
                self.parse_struct(j, attrs)
            elif kw == "enum":
                self.parse_enum(j, attrs)
            elif kw == "union":
                name = t[j + 1].text if j + 1 < n else "?"
                decl = StructDecl(name, "union", None, set())
                self.ctx.structs[name] = decl
                self.i = self.skip_item(j)
            elif kw == "const" and j + 2 < n and t[j + 1].kind == "id" and t[j + 1].text not in ("fn", "unsafe", "extern"):
                self.parse_const(j)
            elif kw == "type":
                self.parse_alias(j)
            elif kw == "mod" and _is_op(t, j + 2, "{"):
                close = _match(t, j + 2)
                _RustParser(t[j + 3:close], self.ctx, self.assume).parse_all()
                self.i = close + 1
            else:
                self.i = self.skip_item(j)
            if self.i <= start:
                self.i = start + 1

    def parse_const(self, j):
        t = self.t
        end = self.skip_item(j)
        eq = next((k for k in range(j, end) if t[k].kind == "op" and t[k].text == "="), None)
        if eq is not None and t[j + 1].text != "_":
            try:
                self.ctx.constants[t[j + 1].text] = self.ctx.eval(_strip_casts(t[eq + 1:end - 1]))
            except (_ExprError, ValueError):
                pass
        self.i = end

    def parse_alias(self, j):
        t, end = self.t, self.skip_item(j)
        self.i = end
        if j + 2 < end and t[j + 1].kind == "id" and _is_op(t, j + 2, "="):
            body = [Tok("op", ">") if x.text == ">>" else x for x in t[j + 3:end - 1]]
            try:
                words, ptr, dims, _ = self.parse_type(body, 0)
                self.ctx.typedefs[t[j + 1].text] = TypedefDecl(words, ptr, dims, None)
            except _Skip as e:
                self.ctx.typedefs[t[j + 1].text] = TypedefDecl(("void",), False, [], str(e))

    def parse_struct(self, j, attrs):
        t = self.t
        if j + 1 >= len(t) or t[j + 1].kind != "id":
            self.i = self.skip_item(j)
            return
        name, k = t[j + 1].text, j + 2
        flags, packed, align = self.repr_of(attrs)
        decl = StructDecl(name, "struct", None, set())
        if _is_op(t, k, "<"):
            decl.reason = "generic structs unsupported"
        elif "transparent" in flags:
            decl.reason = "repr(transparent) unsupported"
        elif align:
            decl.reason = f"repr(align({align})) unsupported"
        elif packed and packed > 1:
            decl.reason = f"repr(packed({packed})) unsupported"
        elif "C" not in flags and not self.assume:
            decl.reason = "layout unspecified: add #[repr(C)] (or pass assume_repr_c=True)"
        if packed == 1:
            decl.pack = 1
        self.ctx.structs[name] = decl
        end = self.skip_item(j)
        self.i = end
        if decl.reason == "generic structs unsupported":
            return
        if _is_op(t, k, "{"):
            segs = _split_commas(t[k + 1:_match(t, k)], angles=True)
            names = None
        elif _is_op(t, k, "("):
            segs = _split_commas(t[k + 1:_match(t, k)], angles=True)
            names = True
        else:
            decl.reason = decl.reason or "empty struct"
            return
        for idx, seg in enumerate(segs):
            seg = self.strip_field_prefix(seg)
            if not seg:
                continue
            if names:
                fname, type_toks = f"f{idx}", seg
            else:
                if len(seg) < 3 or seg[0].kind != "id" or not _is_op(seg, 1, ":"):
                    decl.reason = decl.reason or "unparsed field"
                    continue
                fname, type_toks = seg[0].text, seg[2:]
            type_toks = [x for x in type_toks]
            try:
                words, ptr, dims, _ = self.parse_type(type_toks, 0)
                decl.fields.append(FieldDecl(fname, words, ptr, dims, None, None))
            except _Skip as e:
                decl.fields.append(FieldDecl(fname, ("void",), False, [], None, str(e)))
        if not decl.fields and not decl.reason:
            decl.reason = "empty struct"

    @staticmethod
    def strip_field_prefix(seg):
        while _is_op(seg, 0, "#") and _is_op(seg, 1, "["):
            seg = seg[_match(seg, 1) + 1:]
        if seg and seg[0].kind == "id" and seg[0].text == "pub":
            seg = seg[1:]
            if _is_op(seg, 0, "("):
                seg = seg[_match(seg, 0) + 1:]
        return seg

    def parse_enum(self, j, attrs):
        t = self.t
        self.i = self.skip_item(j)
        if j + 2 >= len(t) or t[j + 1].kind != "id":
            return
        name, k = t[j + 1].text, j + 2
        flags, _, _ = self.repr_of(attrs)
        underlying, error = None, None
        ints = [f for f in flags if f in _PRIMITIVES and f[0] in "ui"]
        if _is_op(t, k, "<"):
            decl = EnumDecl(name, [], None, True)
            decl.error = "generic enums unsupported"
            self.ctx.enums[name] = decl
            return
        if ints:
            underlying = _PRIMITIVES[ints[0]]
        elif "C" not in flags and not self.assume:
            error = "layout unspecified: add #[repr(u8)] or #[repr(C)] (or pass assume_repr_c=True)"
        members, value, declared = [], -1, {}
        if _is_op(t, k, "{"):
            for seg in _split_commas(t[k + 1:_match(t, k)]):
                seg = self.strip_field_prefix(seg)
                while seg and _is_op(seg, 0, "#"):
                    seg = seg[_match(seg, 1) + 1:]
                if not seg or seg[0].kind != "id":
                    continue
                vname = seg[0].text
                if len(seg) > 1 and seg[1].kind == "op" and seg[1].text in ("(", "{"):
                    error = error or "enums with data are not supported (C-like enums only)"
                    continue
                if len(seg) > 1 and _is_op(seg, 1, "="):
                    saved = dict(self.ctx.constants)
                    self.ctx.constants.update(declared)
                    try:
                        value = self.ctx.eval(_strip_casts(seg[2:]))
                    except _ExprError:
                        error = error or f"cannot evaluate {vname}"
                    finally:
                        self.ctx.constants = saved
                else:
                    value += 1
                members.append((vname, value))
                declared[vname] = value
                self.ctx.constants[f"{name}::{vname}"] = value
        decl = EnumDecl(name, members, underlying, True)
        if error:
            decl.error = error
        self.ctx.enums[name] = decl


def from_rust(path=None, *, text=None, byteorder: str = "little", align: bool | None = None,
              assume_repr_c: bool = False, long_size: int | None = None) -> HeaderTypes:
    """Parse Rust source (`path`, or `text=`) into firstclass classes, like `from_header` for C.

    `#[repr(C)]` structs become `@firstclass` classes with C layout (natural padding), or packed
    for `#[repr(C, packed)]`. Enums with `#[repr(u8)]`/`#[repr(i32)]`/`#[repr(C)]` and plain
    discriminants become `IntEnum`s. Supported field types: `u8..u64`, `i8..i64`, `usize`, `isize`,
    `f32`, `f64`, `bool`, `c_int` and friends, `[T; N]` (consts allowed), nested structs, raw
    pointers, references and `fn` pointers (stored as addresses). `[u8; N]` becomes `bytes_t`,
    `[c_char; N]` becomes `char_array`, `wchar_t` / `[wchar_t; N]` the 2-byte UTF-16 types.

    Structs without `#[repr(C)]` have an unspecified layout and are skipped (see `.skipped`)
    unless `assume_repr_c=True`. Generics, unions, tuples, `char`, `u128`, `String`/`Vec` and enums
    with data are skipped with a reason.
    """
    if (path is None) == (text is None):
        raise TypeError("pass exactly one of path or text=")
    if path is not None:
        text = Path(os.fspath(path)).read_text(encoding="utf-8", errors="replace")
    if byteorder not in ("little", "big"):
        raise ValueError(f"byteorder must be 'little' or 'big', got {byteorder!r}")
    ctx = _Ctx(long_size or ctypes.sizeof(ctypes.c_long))
    _RustParser(_tokenize(text), ctx, assume_repr_c).parse_all()
    return _assemble(ctx, byteorder, align)
