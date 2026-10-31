"""Field specs: how one annotated field maps to bytes, JSON, XML and ctypes."""
from __future__ import annotations

import binascii
import ctypes
import enum
import struct
import sys
from typing import Annotated, Any, get_origin
import xml.etree.ElementTree as ET


def _endian(ctype, bo):
    """The byte-swapped (or plain) variant of a ctypes simple type for struct byte order `bo`."""
    try:
        return getattr(ctype, "__ctype_be__" if bo == ">" else "__ctype_le__", ctype)
    except TypeError:  # c_bool, c_char, ...: no swapped variant, and byte order is meaningless anyway
        return ctype


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
            if self.fmt == "f":
                try:
                    struct.pack("<f", value)
                except (OverflowError, struct.error) as e:
                    raise ValueError(f"{value!r} does not fit {self.name}: {e}") from e
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
        if self.kind == "bool" and bo != _NATIVE_BO:
            return ctypes.c_uint8  # a swapped ctypes Structure rejects c_bool; same one-byte layout
        return _endian(self._ctype, bo)

    def to_c(self, value, bo):
        return self.check(value)

    def from_c(self, cvalue):
        return bool(cvalue) if self.kind == "bool" else cvalue

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

    def to_c(self, value, bo):
        return self._encode(value)

    def from_c(self, cvalue):
        return bytes(cvalue).split(b"\0", 1)[0].decode("utf-8")

    def c_decl(self, name):
        return f"char {name}[{self.n}]"


class WCharArray(Spec):
    """Fixed-size UTF-16 string: N two-byte units (like `wchar_t name[N]` on Windows), NUL-padded.

    Characters outside the BMP take two units (a surrogate pair). The unit byte order follows the
    containing class's byteorder.
    """

    def __init__(self, n: int):
        self.n = n

    def align(self):
        return 2

    def size(self):
        return 2 * self.n

    def _units(self, value) -> list[int]:
        if not isinstance(value, str):
            raise TypeError(f"wchar_array needs str, got {type(value).__name__}")
        if "\0" in value:
            raise ValueError("wchar_array value must not contain NUL")
        raw = value.encode("utf-16-le")  # lone surrogates raise UnicodeEncodeError (a ValueError)
        units = list(struct.unpack(f"<{len(raw) // 2}H", raw))
        if len(units) > self.n:
            raise ValueError(f"{value!r} is {len(units)} UTF-16 units, wchar_array[{self.n}] holds {self.n}")
        return units

    @staticmethod
    def _text(units) -> str:
        units = list(units)
        end = units.index(0) if 0 in units else len(units)
        return struct.pack(f"<{end}H", *units[:end]).decode("utf-16-le")

    def check(self, value):
        self._units(value)
        return value

    def pack(self, value, bo):
        units = self._units(value)
        units += [0] * (self.n - len(units))
        return struct.pack(f"{bo}{self.n}H", *units)

    def unpack(self, buf, off, bo):
        return self._text(struct.unpack_from(f"{bo}{self.n}H", buf, off))

    def to_json(self, value):
        return value

    def from_json(self, obj):
        return self.check(obj)

    def to_xml(self, value, elem):
        elem.text = value

    def from_xml(self, elem):
        return self.check(_xml_text(elem))

    def ctype(self, bo):
        return _endian(ctypes.c_uint16, bo) * self.n  # c_wchar is 4 bytes on Linux: use raw 16-bit units

    def to_c(self, value, bo):
        units = self._units(value)
        units += [0] * (self.n - len(units))
        return self.ctype(bo)(*units)

    def from_c(self, cvalue):
        return self._text(cvalue)

    def c_decl(self, name):
        return f"uint16_t {name}[{self.n}]"


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

    def to_c(self, value, bo):
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

    def to_c(self, value, bo):
        arr = self.ctype(bo)()
        for i, v in enumerate(self.check(value)):
            c = self.elem.to_c(v, bo)
            if isinstance(self.elem, CharArray):  # array items need an instance, not bytes
                c = self.elem.ctype(bo).from_buffer_copy(c.ljust(self.elem.n, b"\0"))
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

    def to_c(self, value, bo):
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


class BitField(Spec):
    """`bitfield[uint8_t, 3]`: N bits of a storage int, like `uint8_t x : 3` in C.

    Neighbouring bitfields with the same storage type share one storage unit while they fit
    (little-endian: filled from the lowest bit; big-endian: from the highest bit, like ctypes).
    The Schema does the grouping; this spec only validates values.
    """

    def __init__(self, storage: Scalar, bits: int):
        if isinstance(storage, Pointer) or storage.kind != "int":
            raise TypeError("bitfield storage must be an integer type like uint8_t")
        if not 1 <= bits <= storage.size() * 8:
            raise ValueError(f"bitfield width {bits} does not fit {storage.name}")
        self.storage, self.bits = storage, bits
        self.signed = storage.fmt.islower()
        self.lo, self.hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if self.signed else (0, (1 << bits) - 1)

    def align(self):
        return self.storage.align()

    def size(self):
        return self.storage.size()

    def check(self, value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"bitfield needs int, got {type(value).__name__}")
        if not self.lo <= value <= self.hi:
            raise ValueError(f"{value} out of range for {self.bits}-bit {self.storage.name} [{self.lo}, {self.hi}]")
        return value

    def pack(self, value, bo):
        raise TypeError("bitfields are packed by their containing class")

    unpack = pack

    def to_json(self, value):
        return value

    def from_json(self, obj):
        return self.check(obj)

    def to_xml(self, value, elem):
        elem.text = str(value)

    def from_xml(self, elem):
        return self.check(int(_xml_text(elem).strip()))

    def ctype(self, bo):
        return self.storage.ctype(bo)

    def to_c(self, value, bo):
        return self.check(value)

    def from_c(self, cvalue):
        return cvalue

    def c_decl(self, name):
        return f"{self.storage.c_name} {name} : {self.bits}"


class EnumSpec(Spec):
    """An `enum.Enum` (int-valued) stored as a fixed-width integer."""

    def __init__(self, enum_cls, storage: Scalar):
        if isinstance(storage, Pointer) or storage.kind != "int":
            raise TypeError("enum storage must be an integer type like uint8_t")
        for m in enum_cls:
            if isinstance(m.value, bool) or not isinstance(m.value, int):
                raise TypeError(f"{enum_cls.__name__}.{m.name}: only int-valued enums are supported")
            storage.check(m.value)
        self.cls, self.storage = enum_cls, storage

    def align(self):
        return self.storage.align()

    def size(self):
        return self.storage.size()

    def check(self, value):
        if not isinstance(value, self.cls):
            raise TypeError(f"expected {self.cls.__name__}, got {type(value).__name__}")
        self.storage.check(int(value.value))
        return value

    def _member(self, raw):
        return self.cls(self.storage.check(raw))  # ValueError for unknown values

    def pack(self, value, bo):
        return self.storage.pack(int(self.check(value).value), bo)

    def unpack(self, buf, off, bo):
        return self._member(self.storage.unpack(buf, off, bo))

    def to_json(self, value):
        return int(value.value)

    def from_json(self, obj):
        return self._member(obj)

    def to_xml(self, value, elem):
        elem.text = str(int(value.value))

    def from_xml(self, elem):
        return self._member(int(_xml_text(elem).strip()))

    def ctype(self, bo):
        return self.storage.ctype(bo)

    def to_c(self, value, bo):
        return int(self.check(value).value)

    def from_c(self, cvalue):
        return self._member(cvalue)

    def c_decl(self, name):
        return self.storage.c_decl(name)

    def c_definition(self) -> str:
        lines = [f"enum {self.cls.__name__} {{"]
        lines += [f"    {self.cls.__name__}_{m.name} = {int(m.value)}," for m in self.cls]
        lines.append("};")
        return "\n".join(lines)


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

    def to_c(self, value, bo):
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
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        # bare enum: like a C enum, 32-bit (signed only if some value is negative)
        signed = any(isinstance(m.value, int) and m.value < 0 for m in tp)
        return EnumSpec(tp, resolve_spec(int32_t if signed else uint32_t))
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
wchar_t = Annotated[str, WCharArray(1)]  # one 2-byte UTF-16 unit


class char_array:
    """`char_array[N]`: NUL-padded UTF-8 string in N bytes."""

    def __class_getitem__(cls, n):
        return Annotated[str, CharArray(_positive(n))]


class wchar_array:
    """`wchar_array[N]`: UTF-16 string in N two-byte units (2*N bytes), NUL-padded."""

    def __class_getitem__(cls, n):
        return Annotated[str, WCharArray(_positive(n))]


class bytes_t:
    """`bytes_t[N]`: exactly N raw bytes."""

    def __class_getitem__(cls, n):
        return Annotated[bytes, ByteArray(_positive(n))]


class bitfield:
    """`bitfield[uint8_t, 3]`: a 3-bit field stored in a uint8_t unit (C: `uint8_t x : 3`)."""

    def __class_getitem__(cls, params):
        try:
            storage, bits = params
        except (TypeError, ValueError):
            raise TypeError("bitfield needs two params: bitfield[storage_type, bits]") from None
        st = resolve_spec(storage)
        if not isinstance(st, Scalar):
            raise TypeError("bitfield storage must be an integer type like uint8_t")
        return Annotated[int, BitField(st, _positive(bits))]


class enum_t:
    """`enum_t[Color, uint8_t]`: an int-valued Enum stored as the given integer type."""

    def __class_getitem__(cls, params):
        try:
            enum_cls, storage = params
        except (TypeError, ValueError):
            raise TypeError("enum_t needs two params: enum_t[EnumClass, storage_type]") from None
        if not (isinstance(enum_cls, type) and issubclass(enum_cls, enum.Enum)):
            raise TypeError(f"{enum_cls!r} is not an Enum")
        st = resolve_spec(storage)
        if not isinstance(st, Scalar):
            raise TypeError("enum storage must be an integer type like uint8_t")
        return Annotated[enum_cls, EnumSpec(enum_cls, st)]


class array_t:
    """`array_t[T, N]`: exactly N items of T (scalar, char_array, or firstclass class)."""

    def __class_getitem__(cls, params):
        try:
            elem, n = params
        except (TypeError, ValueError):
            raise TypeError("array_t needs two params: array_t[element_type, length]") from None
        spec = resolve_spec(elem)
        if isinstance(spec, BitField):
            raise TypeError("arrays of bitfields are not supported")
        return Annotated[list, ArrayOf(spec, _positive(n))]
