"""The @firstclass decorator and the per-class Schema it attaches."""
from __future__ import annotations

import ctypes
import dataclasses
import json
import sys
import typing
import xml.etree.ElementTree as ET

from ._rustgen import rust_source
from ._specs import ArrayOf, BitField, EnumSpec, Nested, Spec, resolve_spec

_METHODS = (
    "to_bytes", "from_bytes", "unpack_from", "to_dict", "from_dict", "to_json", "from_json",
    "to_xml", "from_xml", "to_ctypes", "from_ctypes", "ctypes_type", "c_header", "rust_source", "sizeof",
)


def _round_up(n: int, a: int) -> int:
    return -(-n // a) * a


class Schema:
    def __init__(self, cls, byteorder: str, align: bool):
        if byteorder not in ("little", "big"):
            raise ValueError(f"byteorder must be 'little' or 'big', got {byteorder!r}")
        self.cls = cls
        self.bo = "<" if byteorder == "little" else ">"
        self.byteorder = byteorder
        self.aligned = align
        self._fields: list[tuple[dataclasses.Field, Spec]] | None = None
        self._layout = None
        self._ctype = None
        self._spec_map = None

    # -- structure -----------------------------------------------------------

    @property
    def fields(self):
        if self._fields is None:
            hints = typing.get_type_hints(self.cls, include_extras=True)
            self._fields = [(f, resolve_spec(hints[f.name])) for f in dataclasses.fields(self.cls)]
        return self._fields

    def require_binary(self):
        for f, spec in self.fields:
            if not spec.binary:
                raise TypeError(
                    f"{self.cls.__name__}.{f.name}: type {spec.tp!r} has no binary form; "
                    "annotate it with uint8_t, char_array[N], ... or a firstclass class"
                )

    def layout(self):
        """-> (items, total_size, widest_alignment).

        An item is ("field", index, offset) or ("bits", storage_spec, offset, [(index, shift, bits)]).
        """
        if self._layout is None:
            self.require_binary()
            msb = self.bo == ">"
            off, widest, items, group = 0, 1, [], None  # group = [storage, used_bits, members]
            for i, (_, spec) in enumerate(self.fields):
                if isinstance(spec, BitField):
                    st, unit = spec.storage, spec.storage.size() * 8
                    if group is None or group[0].size() != st.size() or group[1] + spec.bits > unit:
                        a = st.align() if self.aligned else 1
                        off = _round_up(off, a)
                        widest = max(widest, a)
                        group = [st, 0, []]
                        items.append(("bits", st, off, group[2]))
                        off += st.size()
                    shift = unit - group[1] - spec.bits if msb else group[1]
                    group[2].append((i, shift, spec.bits))
                    group[1] += spec.bits
                    continue
                group = None
                a = spec.align() if self.aligned else 1
                off = _round_up(off, a)
                items.append(("field", i, off))
                off += spec.size()
                widest = max(widest, a)
            self._layout = (items, _round_up(off, widest), widest)
        return self._layout

    def size(self) -> int:
        return self.layout()[1]

    def align(self) -> int:
        return self.layout()[2]

    def _build(self, kw: dict):
        init = {k: v for k, v in kw.items() if self._by_name[k].init}
        obj = self.cls(**init)
        for k, v in kw.items():
            if not self._by_name[k].init:
                object.__setattr__(obj, k, v)
        return obj

    @property
    def _by_name(self):
        return {f.name: f for f, _ in self.fields}  # small; fields are cached

    @property
    def spec_map(self):
        if self._spec_map is None:
            self._spec_map = {f.name: (f, spec) for f, spec in self.fields}
        return self._spec_map

    def validate_instance(self, obj):
        """Check (and normalise) every field of a freshly built instance."""
        for f, spec in self.fields:
            if hasattr(obj, f.name):
                try:
                    value = spec.check(getattr(obj, f.name))
                except (TypeError, ValueError) as e:
                    raise self._ctx(f, e) from e
                object.__setattr__(obj, f.name, value)

    def checked_setattr(self, obj, name, value):
        entry = self.spec_map.get(name)
        if entry is not None:
            try:
                value = entry[1].check(value)
            except (TypeError, ValueError) as e:
                raise self._ctx(entry[0], e) from e
        object.__setattr__(obj, name, value)

    def _ctx(self, f, exc):
        kind = TypeError if isinstance(exc, TypeError) else ValueError  # e.g. UnicodeDecodeError -> ValueError
        return kind(f"{self.cls.__name__}.{f.name}: {exc}")

    # -- bytes ---------------------------------------------------------------

    def pack(self, obj) -> bytes:
        items, total, _ = self.layout()
        buf = bytearray(total)
        order = "little" if self.bo == "<" else "big"
        for item in items:
            if item[0] == "field":
                f, spec = self.fields[item[1]]
                try:
                    raw = spec.pack(getattr(obj, f.name), self.bo)
                except (TypeError, ValueError) as e:
                    raise self._ctx(f, e) from e
                buf[item[2]:item[2] + len(raw)] = raw
                continue
            _, st, off, members = item
            unit = 0
            for i, shift, bits in members:
                f, spec = self.fields[i]
                try:
                    unit |= (spec.check(getattr(obj, f.name)) & ((1 << bits) - 1)) << shift
                except (TypeError, ValueError) as e:
                    raise self._ctx(f, e) from e
            buf[off:off + st.size()] = unit.to_bytes(st.size(), order)
        return bytes(buf)

    def unpack(self, buf, off: int = 0):
        items, total, _ = self.layout()
        if len(buf) - off < total:
            raise ValueError(f"{self.cls.__name__} needs {total} bytes, got {len(buf) - off}")
        order = "little" if self.bo == "<" else "big"
        kw = {}
        for item in items:
            if item[0] == "field":
                f, spec = self.fields[item[1]]
                try:
                    kw[f.name] = spec.unpack(buf, off + item[2], self.bo)
                except (TypeError, ValueError) as e:
                    raise self._ctx(f, e) from e
                continue
            _, st, o, members = item
            unit = int.from_bytes(bytes(buf[off + o:off + o + st.size()]), order)
            for i, shift, bits in members:
                f, spec = self.fields[i]
                v = (unit >> shift) & ((1 << bits) - 1)
                if spec.signed and v >> (bits - 1):
                    v -= 1 << bits
                kw[f.name] = v
        return self._build(kw)

    @staticmethod
    def _as_bytes_like(data):
        return data if isinstance(data, (bytes, bytearray)) else memoryview(data).cast("B")

    def unpack_from(self, buffer, offset: int = 0):
        """Like struct.unpack_from: read one object at `offset`; extra bytes after it are fine."""
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(f"offset must be a non-negative int, got {offset!r}")
        return self.unpack(self._as_bytes_like(buffer), offset)

    def from_bytes(self, data):
        data = self._as_bytes_like(data)
        if len(data) != self.size():
            raise ValueError(f"{self.cls.__name__} needs exactly {self.size()} bytes, got {len(data)}")
        return self.unpack(data, 0)

    # -- dict / json / xml ---------------------------------------------------

    def to_dict(self, obj) -> dict:
        return {f.name: spec.to_json(getattr(obj, f.name)) for f, spec in self.fields}

    def from_dict(self, d):
        if not isinstance(d, dict):
            raise TypeError(f"{self.cls.__name__}.from_dict needs a dict, got {type(d).__name__}")
        unknown = set(d) - set(self._by_name)
        if unknown:
            raise ValueError(f"{self.cls.__name__}: unknown keys {sorted(unknown)}")
        kw = {}
        for f, spec in self.fields:
            if f.name in d:
                try:
                    kw[f.name] = spec.from_json(d[f.name])
                except (TypeError, ValueError) as e:
                    raise self._ctx(f, e) from e
        return self._build(kw)

    def fill_xml(self, obj, elem: ET.Element) -> None:
        for f, spec in self.fields:
            spec.to_xml(getattr(obj, f.name), ET.SubElement(elem, f.name))

    def read_xml(self, elem: ET.Element):
        known = self._by_name
        specs = {f.name: s for f, s in self.fields}
        kw = {}
        for child in elem:
            if child.tag not in known:
                raise ValueError(f"{self.cls.__name__}: unknown element <{child.tag}>")
            if child.tag in kw:
                raise ValueError(f"{self.cls.__name__}: duplicate element <{child.tag}>")
            spec = specs[child.tag]
            try:
                kw[child.tag] = spec.from_xml(child)
            except (TypeError, ValueError) as e:
                raise self._ctx(known[child.tag], e) from e
        return self._build(kw)

    # -- ctypes / C ----------------------------------------------------------

    def ctypes_type(self):
        if self._ctype is None:
            self.require_binary()
            base = ctypes.LittleEndianStructure if self.bo == "<" else ctypes.BigEndianStructure
            ns = {"_fields_": [
                (f.name, spec.ctype(self.bo), spec.bits) if isinstance(spec, BitField)
                else (f.name, spec.ctype(self.bo))
                for f, spec in self.fields
            ]}
            if not self.aligned:
                ns["_pack_"] = 1
            if sys.version_info >= (3, 14):
                ns["_layout_"] = "ms"  # our bitfield rules; also silences the _pack_ warning
            self._ctype = type(self.cls.__name__ + "_c", (base,), ns)
        return self._ctype

    def to_ctypes(self, obj):
        c = self.ctypes_type()()
        for f, spec in self.fields:
            try:
                setattr(c, f.name, spec.to_c(getattr(obj, f.name), self.bo))
            except (TypeError, ValueError) as e:
                raise self._ctx(f, e) from e
        return c

    def from_ctypes(self, c):
        return self._build({f.name: spec.from_c(getattr(c, f.name)) for f, spec in self.fields})

    def _nested_schemas(self):
        for _, spec in self.fields:
            while isinstance(spec, ArrayOf):
                spec = spec.elem
            if isinstance(spec, Nested):
                yield spec.schema

    def c_header(self) -> str:
        self.require_binary()
        out, seen = [], set()

        def emit(schema):
            if schema.cls in seen:
                return
            seen.add(schema.cls)
            for dep in schema._nested_schemas():
                emit(dep)
            for _, spec in schema.fields:
                while isinstance(spec, ArrayOf):
                    spec = spec.elem
                if isinstance(spec, EnumSpec) and spec.cls not in seen:
                    seen.add(spec.cls)
                    out.append(spec.c_definition())
            lines = [f"/* byteorder: {schema.byteorder} */"]
            if not schema.aligned:
                lines.append("#pragma pack(push, 1)")
            lines.append(f"struct {schema.cls.__name__} {{")
            lines += [f"    {spec.c_decl(f.name)};" for f, spec in schema.fields]
            lines.append("};")
            if not schema.aligned:
                lines.append("#pragma pack(pop)")
            out.append("\n".join(lines))

        emit(self)
        return "#include <stdint.h>\n#include <stdbool.h>\n\n" + "\n\n".join(out) + "\n"


def firstclass(cls=None, /, *, byteorder: str = "little", align: bool = False, **dataclass_kwargs):
    """Like `@dataclass` (same keyword args), plus binary/JSON/XML/ctypes/C codecs.

    Unlike a plain dataclass, `order=True` is the default (compare field by field, like tuples);
    pass `order=False` to turn it off. Values are validated when an instance is created and on
    every attribute assignment: a field that cannot be encoded (out of range, too long, wrong type)
    raises `ValueError` / `TypeError` immediately, never later in `to_bytes()`.

    byteorder: 'little' (default) or 'big'.
    align: False (default) = packed layout, no padding; True = natural C alignment padding.
    """

    def wrap(cls):
        if not isinstance(cls, type):
            raise TypeError(f"@firstclass needs a class, got {type(cls).__name__}")
        kwargs = dict(dataclass_kwargs)
        kwargs.setdefault("order", kwargs.get("eq", True))  # ==, <, <=, >, >= by field order
        holder = []
        frozen = bool(kwargs.get("frozen"))
        if frozen:  # frozen __init__ bypasses __setattr__, so validate once the instance is built
            user_post = getattr(cls, "__post_init__", None)
            if getattr(user_post, "_firstclass", False):
                user_post = None  # inherited from a firstclass parent: already covered

            def __post_init__(self, *args):
                holder[0].validate_instance(self)
                if user_post is not None:
                    user_post(self, *args)
            __post_init__._firstclass = True
            cls.__post_init__ = __post_init__
        dc = dataclasses.dataclass(cls, **kwargs)
        clash = [f.name for f in dataclasses.fields(dc) if f.name in _METHODS or f.name == "__firstclass__"]
        if clash:
            raise TypeError(f"{dc.__name__}: field names clash with firstclass methods: {clash}")
        s = Schema(dc, byteorder, align)
        dc.__firstclass__ = s
        holder.append(s)
        if not frozen:  # every assignment (including those in __init__) is checked
            dc.__setattr__ = lambda self, name, value: s.checked_setattr(self, name, value)
        dc.to_bytes = lambda self: s.pack(self)
        dc.from_bytes = staticmethod(s.from_bytes)
        dc.unpack_from = staticmethod(s.unpack_from)
        dc.to_dict = lambda self: s.to_dict(self)
        dc.from_dict = staticmethod(s.from_dict)
        dc.to_json = lambda self, **kw: json.dumps(s.to_dict(self), **kw)
        dc.from_json = staticmethod(lambda text: s.from_dict(json.loads(text)))
        dc.to_xml = lambda self, indent=None: _to_xml(s, self, indent)
        dc.from_xml = staticmethod(lambda text: s.read_xml(_parse_xml(text)))
        dc.to_ctypes = lambda self: s.to_ctypes(self)
        dc.from_ctypes = staticmethod(s.from_ctypes)
        dc.ctypes_type = staticmethod(s.ctypes_type)
        dc.c_header = staticmethod(s.c_header)
        dc.rust_source = staticmethod(lambda: rust_source(s))
        dc.sizeof = staticmethod(s.size)
        return dc

    return wrap if cls is None else wrap(cls)


def _parse_xml(text):
    try:
        return ET.fromstring(text)
    except ET.ParseError as e:
        raise ValueError(f"malformed XML: {e}") from e


def _to_xml(schema: Schema, obj, indent) -> str:
    root = ET.Element(schema.cls.__name__)
    schema.fill_xml(obj, root)
    if indent is not None:
        ET.indent(root, space=" " * indent if isinstance(indent, int) else indent)
    return ET.tostring(root, encoding="unicode")
