"""Field specs: how one annotated field maps to bytes, JSON, XML and ctypes."""
from __future__ import annotations

import binascii
import ctypes
import struct
import sys
from typing import Annotated, Any, get_origin
import xml.etree.ElementTree as ET


class Spec:
    """Base class. `bo` is a struct byte-order prefix: '<' or '>'."""

    binary = True

    def align(self) -> int:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def check(self, value: Any) -> Any:
        """Validate (and normalise) a python value; raises TypeError/ValueError."""
        raise NotImplementedError

    def pack(self, value: Any, bo: str) -> bytes:
        raise NotImplementedError

    def unpack(self, buf, off: int, bo: str) -> Any:
        raise NotImplementedError

    def to_json(self, value: Any) -> Any:
        raise NotImplementedError

    def from_json(self, obj: Any) -> Any:
        raise NotImplementedError

    def to_xml(self, value: Any, elem: ET.Element) -> None:
        raise NotImplementedError

    def from_xml(self, elem: ET.Element) -> Any:
        raise NotImplementedError

    def ctype(self, bo: str):
        raise NotImplementedError

    def to_c(self, value: Any) -> Any:
        raise NotImplementedError

    def from_c(self, cvalue: Any) -> Any:
        raise NotImplementedError

    def c_decl(self, name: str) -> str:
        raise NotImplementedError


def _xml_text(elem: ET.Element) -> str:
    return elem.text or ""


class Scalar(Spec):
    def __init__(self, name: str, fmt: str, ctype, c_name: str, kind: str):
        self.name, self.fmt, self._ctype, self.c_name, self.kind = name, fmt, ctype, c_name, kind
        self._size = struct.calcsize("<" + fmt)
        if kind == "int":
            bits = self._size * 8
            signed = fmt.islower()
            self.lo, self.hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)

    def align(self):
        return self._size

    def size(self):
        return self._size

    def check(self, value):
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise TypeError(f"{self.name} needs bool, got {type(value).__name__}")
        elif self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{self.name} needs int, got {type(value).__name__}")
            if not self.lo <= value <= self.hi:
                raise ValueError(f"{value} out of range for {self.name} [{self.lo}, {self.hi}]")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{self.name} needs float, got {type(value).__name__}")
        return value

    def pack(self, value, bo):
        value = self.check(value)
        try:
            return struct.pack(bo + self.fmt, value)
        except (OverflowError, struct.error) as e:
            raise ValueError(f"{value!r} does not fit {self.name}: {e}") from e

    def unpack(self, buf, off, bo):
        return struct.unpack_from(bo + self.fmt, buf, off)[0]

    def to_json(self, value):
        return value

    def from_json(self, obj):
        return self.check(obj)

    def to_xml(self, value, elem):
        elem.text = ("true" if value else "false") if self.kind == "bool" else repr(value)

    def from_xml(self, elem):
        text = _xml_text(elem).strip()
        if self.kind == "bool":
            if text not in ("true", "false"):
                raise ValueError(f"bad bool text {text!r}")
            return text == "true"
        return self.check(int(text) if self.kind == "int" else float(text))

    def ctype(self, bo):
        return self._ctype

    def to_c(self, value):
        return self.check(value)

    def from_c(self, cvalue):
        return cvalue

    def c_decl(self, name):
        return f"{self.c_name} {name}"


class CharArray(Spec):
    """Fixed-size, NUL-padded UTF-8 string, like `char name[N]`."""

    def __init__(self, n: int):
        self.n = n

    def align(self):
        return 1

    def size(self):
        return self.n

    def _encode(self, value) -> bytes:
        if not isinstance(value, str):
            raise TypeError(f"char_array needs str, got {type(value).__name__}")
        if "\0" in value:
            raise ValueError("char_array value must not contain NUL")
        raw = value.encode("utf-8")
        if len(raw) > self.n:
            raise ValueError(f"{value!r} is {len(raw)} bytes, char_array[{self.n}] holds {self.n}")
        return raw

    def check(self, value):
        self._encode(value)
        return value

    def pack(self, value, bo):
        return self._encode(value).ljust(self.n, b"\0")

    def unpack(self, buf, off, bo):
        return bytes(buf[off:off + self.n]).split(b"\0", 1)[0].decode("utf-8")

    def to_json(self, value):
        return value

    def from_json(self, obj):
        return self.check(obj)

    def to_xml(self, value, elem):
        elem.text = value

    def from_xml(self, elem):
        return self.check(_xml_text(elem))

    def ctype(self, bo):
        return ctypes.c_char * self.n

    def to_c(self, value):
        return self._encode(value)

    def from_c(self, cvalue):
        return bytes(cvalue).split(b"\0", 1)[0].decode("utf-8")

    def c_decl(self, name):
        return f"char {name}[{self.n}]"


class ByteArray(Spec):
    """Fixed-size raw bytes, exactly N long."""

    def __init__(self, n: int):
        self.n = n

    def align(self):
        return 1

    def size(self):
        return self.n

    def check(self, value):
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(f"bytes_t needs bytes, got {type(value).__name__}")
        value = bytes(value)
        if len(value) != self.n:
            raise ValueError(f"bytes_t[{self.n}] needs exactly {self.n} bytes, got {len(value)}")
        return value

    def pack(self, value, bo):
        return self.check(value)

    def unpack(self, buf, off, bo):
        return bytes(buf[off:off + self.n])

    def to_json(self, value):
        return value.hex()

    def from_json(self, obj):
        if not isinstance(obj, str):
            raise TypeError("bytes_t JSON value must be a hex string")
        try:
            return self.check(binascii.unhexlify(obj))
        except binascii.Error as e:
            raise ValueError(f"bad hex string: {e}") from e

    def to_xml(self, value, elem):
        elem.text = value.hex()

    def from_xml(self, elem):
        return self.from_json(_xml_text(elem).strip())

    def ctype(self, bo):
        return ctypes.c_ubyte * self.n

    def to_c(self, value):
        return (ctypes.c_ubyte * self.n).from_buffer_copy(self.check(value))

    def from_c(self, cvalue):
        return bytes(cvalue)

    def c_decl(self, name):
        return f"uint8_t {name}[{self.n}]"


class ArrayOf(Spec):
    """Fixed-length array of any other spec, exactly N elements."""

    def __init__(self, elem: Spec, n: int):
        self.elem, self.n = elem, n
        self.binary = elem.binary

    def align(self):
        return self.elem.align()

    def size(self):
        return self.elem.size() * self.n

    def check(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"array_t needs list, got {type(value).__name__}")
        if len(value) != self.n:
            raise ValueError(f"array_t[..., {self.n}] needs {self.n} items, got {len(value)}")
        return [self.elem.check(v) for v in value]

    def pack(self, value, bo):
        return b"".join(self.elem.pack(v, bo) for v in self.check(value))

    def unpack(self, buf, off, bo):
        step = self.elem.size()
        return [self.elem.unpack(buf, off + i * step, bo) for i in range(self.n)]

    def to_json(self, value):
        return [self.elem.to_json(v) for v in value]

    def from_json(self, obj):
        if not isinstance(obj, (list, tuple)):
            raise TypeError("array_t JSON value must be a list")
        return self.check([self.elem.from_json(v) for v in obj])

    def to_xml(self, value, elem):
        for v in value:
            self.elem.to_xml(v, ET.SubElement(elem, "item"))

    def from_xml(self, elem):
        kids = list(elem)
        if any(k.tag != "item" for k in kids):
            raise ValueError("array children must be <item>")
        return self.check([self.elem.from_xml(k) for k in kids])

    def ctype(self, bo):
        return self.elem.ctype(bo) * self.n

    def to_c(self, value):
        arr = self.ctype(None)()
        for i, v in enumerate(self.check(value)):
            c = self.elem.to_c(v)
            if isinstance(self.elem, CharArray):  # array items need an instance, not bytes
                c = self.elem.ctype(None).from_buffer_copy(c.ljust(self.elem.n, b"\0"))
            arr[i] = c
        return arr

    def from_c(self, cvalue):
        return [self.elem.from_c(cvalue[i]) for i in range(self.n)]

    def c_decl(self, name):
        return self.elem.c_decl(f"{name}[{self.n}]")


class Nested(Spec):
    """A field whose type is itself a firstclass class."""

    def __init__(self, schema):
        self.schema = schema

    def align(self):
        return self.schema.align()

    def size(self):
        return self.schema.size()

    def check(self, value):
        if not isinstance(value, self.schema.cls):
            raise TypeError(f"expected {self.schema.cls.__name__}, got {type(value).__name__}")
        return value

    def pack(self, value, bo):
        return self.schema.pack(self.check(value))

    def unpack(self, buf, off, bo):
        return self.schema.unpack(buf, off)

    def to_json(self, value):
        return self.schema.to_dict(value)

    def from_json(self, obj):
        return self.schema.from_dict(obj)

    def to_xml(self, value, elem):
        self.schema.fill_xml(value, elem)

    def from_xml(self, elem):
        return self.schema.read_xml(elem)

    def ctype(self, bo):
        return self.schema.ctypes_type()

    def to_c(self, value):
        return self.schema.to_ctypes(self.check(value))

    def from_c(self, cvalue):
        return self.schema.from_ctypes(cvalue)

    def c_decl(self, name):
        return f"struct {self.schema.cls.__name__} {name}"


class Plain(Spec):
    """Un-annotated field: JSON/XML only, no binary form."""

    binary = False

    def __init__(self, tp):
        self.tp = tp

    def check(self, value):
        return value

    def to_json(self, value):
        return value

    def from_json(self, obj):
        return obj

    def to_xml(self, value, elem):
        if self.tp is bool:
            elem.text = "true" if value else "false"
        elif self.tp in (int, float, str):
            elem.text = str(value) if self.tp is not float else repr(value)
        else:
            raise TypeError(f"XML needs a firstclass type for {self.tp!r}")

    def from_xml(self, elem):
        text = _xml_text(elem)
        if self.tp is bool:
            if text.strip() not in ("true", "false"):
                raise ValueError(f"bad bool text {text!r}")
            return text.strip() == "true"
        if self.tp in (int, float):
            return self.tp(text.strip())
        if self.tp is str:
            return text
        raise TypeError(f"XML needs a firstclass type for {self.tp!r}")


_NATIVE_BO = "<" if sys.byteorder == "little" else ">"
_PTR_SIZE = ctypes.sizeof(ctypes.c_void_p)


class Pointer(Scalar):
    """A ctypes pointer field. The python value is the address as an int (0 = NULL).

    The bytes are the raw address, so it only round-trips inside one process/ABI.
    """

    def __init__(self, tp):
        code = "Q" if _PTR_SIZE == 8 else "I"
        super().__init__("pointer", code, ctypes.c_void_p, "void *", "int")
        self.tp = tp

    def ctype(self, bo):
        if bo != _NATIVE_BO:
            raise TypeError("ctypes pointers need the platform's native byteorder")
        return self.tp

    def to_c(self, value):
        value = self.check(value)
        if self.tp is ctypes.c_void_p:
            return value
        return ctypes.cast(ctypes.c_void_p(value), self.tp)

    def from_c(self, cvalue):
        if isinstance(cvalue, int) or cvalue is None:
            return cvalue or 0
        return ctypes.cast(cvalue, ctypes.c_void_p).value or 0

    def c_decl(self, name):
        return f"void *{name}"


_SIGNED = {1: ctypes.c_int8, 2: ctypes.c_int16, 4: ctypes.c_int32, 8: ctypes.c_int64}
_UNSIGNED = {1: ctypes.c_uint8, 2: ctypes.c_uint16, 4: ctypes.c_uint32, 8: ctypes.c_uint64}
_INT_CODES = "bBhHiIlLqQ"


def _ctypes_spec(tp):
    """Spec for a raw ctypes type used as an annotation, or None."""
    if not isinstance(tp, type):
        return None
    if issubclass(tp, ctypes.Array):
        elem = tp._type_
        if elem is ctypes.c_char:
            return CharArray(tp._length_)
        sub = resolve_spec(elem)
        return None if not sub.binary else ArrayOf(sub, tp._length_)
    if issubclass(tp, ctypes._Pointer):
        return Pointer(tp)
    if issubclass(tp, ctypes.Structure) or issubclass(tp, ctypes.Union):
        return None
    if not issubclass(tp, ctypes._SimpleCData):
        return None
    code = tp._type_
    if code in "PzZ":
        return Pointer(ctypes.c_void_p)
    if code == "?":
        return Scalar("c_bool", "?", ctypes.c_bool, "bool", "bool")
    if code == "c":
        return CharArray(1)
    if code == "f":
        return Scalar("c_float", "f", ctypes.c_float, "float", "float")
    if code == "d":
        return Scalar("c_double", "d", ctypes.c_double, "double", "float")
    if code in _INT_CODES:
        size = ctypes.sizeof(tp)
        unsigned = code.isupper()
        canon = (_UNSIGNED if unsigned else _SIGNED)[size]
        fmt = {1: "b", 2: "h", 4: "i", 8: "q"}[size]
        name = f"{'u' if unsigned else ''}int{size * 8}_t"
        return Scalar(tp.__name__, fmt.upper() if unsigned else fmt, canon, name, "int")
    return None


def resolve_spec(tp) -> Spec:
    if get_origin(tp) is Annotated:
        for meta in tp.__metadata__:
            if isinstance(meta, Spec):
                return meta
    schema = getattr(tp, "__firstclass__", None) if isinstance(tp, type) else None
    if schema is not None:
        return Nested(schema)
    return _ctypes_spec(tp) or Plain(tp)


def _positive(n) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise TypeError(f"size must be a positive int, got {n!r}")
    return n


# --- public type markers -----------------------------------------------------

uint8_t = Annotated[int, Scalar("uint8_t", "B", ctypes.c_uint8, "uint8_t", "int")]
uint16_t = Annotated[int, Scalar("uint16_t", "H", ctypes.c_uint16, "uint16_t", "int")]
uint32_t = Annotated[int, Scalar("uint32_t", "I", ctypes.c_uint32, "uint32_t", "int")]
uint64_t = Annotated[int, Scalar("uint64_t", "Q", ctypes.c_uint64, "uint64_t", "int")]
int8_t = Annotated[int, Scalar("int8_t", "b", ctypes.c_int8, "int8_t", "int")]
int16_t = Annotated[int, Scalar("int16_t", "h", ctypes.c_int16, "int16_t", "int")]
int32_t = Annotated[int, Scalar("int32_t", "i", ctypes.c_int32, "int32_t", "int")]
int64_t = Annotated[int, Scalar("int64_t", "q", ctypes.c_int64, "int64_t", "int")]
float32_t = Annotated[float, Scalar("float32_t", "f", ctypes.c_float, "float", "float")]
float64_t = Annotated[float, Scalar("float64_t", "d", ctypes.c_double, "double", "float")]
bool_t = Annotated[bool, Scalar("bool_t", "?", ctypes.c_bool, "bool", "bool")]
char_t = Annotated[str, CharArray(1)]


class char_array:
    """`char_array[N]`: NUL-padded UTF-8 string in N bytes."""

    def __class_getitem__(cls, n):
        return Annotated[str, CharArray(_positive(n))]


class bytes_t:
    """`bytes_t[N]`: exactly N raw bytes."""

    def __class_getitem__(cls, n):
        return Annotated[bytes, ByteArray(_positive(n))]


class array_t:
    """`array_t[T, N]`: exactly N items of T (scalar, char_array, or firstclass class)."""

    def __class_getitem__(cls, params):
        try:
            elem, n = params
        except (TypeError, ValueError):
            raise TypeError("array_t needs two params: array_t[element_type, length]") from None
        return Annotated[list, ArrayOf(resolve_spec(elem), _positive(n))]
