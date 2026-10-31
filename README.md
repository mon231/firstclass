# firstclass

`@firstclass` works like `@dataclass` (same keyword args) and adds conversion to and from
packed bytes, `ctypes`, a C struct header, JSON and XML. Pure Python 3.10+, no dependencies.

```python
from firstclass import firstclass, uint8_t, uint16_t, int32_t, float32_t, bool_t, char_array, bytes_t, array_t

@firstclass
class Point:
    x: int32_t
    y: int32_t

@firstclass
class Packet:
    kind: uint8_t
    ok: bool_t
    temp: float32_t
    name: char_array[8]        # NUL-padded UTF-8
    mac: bytes_t[6]            # exactly 6 bytes
    samples: array_t[uint16_t, 4]
    origin: Point              # nested

p = Packet(1, True, 21.5, "sensor", b"\x00\x11\x22\x33\x44\x55", [1, 2, 3, 4], Point(10, -20))

raw = p.to_bytes()                 # packed, little-endian
assert Packet.from_bytes(raw) == p
Packet.sizeof()                    # 36
```

## Types

| Marker | C | Bytes |
|---|---|---|
| `uint8_t` `uint16_t` `uint32_t` `uint64_t` | `uintN_t` | 1/2/4/8 |
| `int8_t` `int16_t` `int32_t` `int64_t` | `intN_t` | 1/2/4/8 |
| `float32_t` / `float64_t` | `float` / `double` | 4 / 8 |
| `bool_t` | `bool` | 1 |
| `char_t` | `char` | 1 |
| `char_array[N]` | `char x[N]` | N (UTF-8, NUL-padded) |
| `bytes_t[N]` | `uint8_t x[N]` | N |
| `array_t[T, N]` | `T x[N]` | N * size(T) |
| another `@firstclass` class | `struct Name` | its size |

Values are checked: out of range ints, over-long strings and wrong lengths raise
`ValueError`; wrong Python types raise `TypeError`. Nothing is truncated or coerced silently.

Raw `ctypes` types work as annotations too (packed, same layout as a `_pack_ = 1` Structure):
`ctypes.c_int`, `c_uint64`, `c_size_t`, `c_bool`, `c_float`, `c_double`, `c_char`, `c_uint16 * 4`,
`c_char * 8`. Pointers (`c_void_p`, `POINTER(T)`, `c_char_p`) are stored as the address (an `int`,
0 = NULL) at pointer size; `to_ctypes()` gives a real pointer field you can dereference. The
address only means something inside the same process. Sizes follow the platform
(`c_long` is 4 bytes on Windows, 8 on Linux).

## Methods added to every class

| Method | |
|---|---|
| `obj.to_bytes()` / `Cls.from_bytes(data)` | binary; `from_bytes` needs exactly `sizeof()` bytes |
| `Cls.sizeof()` | size in bytes |
| `obj.to_dict()` / `Cls.from_dict(d)` | JSON-compatible dict (bytes as hex) |
| `obj.to_json(**kw)` / `Cls.from_json(s)` | `kw` goes to `json.dumps` |
| `obj.to_xml(indent=None)` / `Cls.from_xml(s)` | `<Packet><kind>1</kind>...</Packet>`, arrays as `<item>` |
| `obj.to_ctypes()` / `Cls.from_ctypes(c)` / `Cls.ctypes_type()` | `ctypes.Structure` with matching layout |
| `Cls.c_header()` | C source for the struct(s), nested ones first |

## Layout options

```python
@firstclass(byteorder="big")   # default "little"
@firstclass(align=True)        # default False = packed, no padding (like __attribute__((packed)))
@firstclass(frozen=True, slots=True)   # any dataclass option
```

`align=True` inserts natural-alignment padding like a normal C struct (assumes the x86-64 style
rule: alignment equals size). Padding bytes are written as zero and ignored on read.
C headers describe the struct only; C compilers use native byte order, so `byteorder` is noted
in a comment.

## Notes

- Fields with plain types (`str`, `int`) work for JSON/XML only; `to_bytes()` raises
  `TypeError` naming the field.
- Use `from __future__ import annotations` freely; annotations are resolved lazily. Classes
  defined inside functions that refer to local names are not resolvable.
- Don't parse untrusted XML with `from_xml` without extra care (stdlib `xml.etree`).
- Field names that collide with the generated methods (`to_bytes`, ...) raise `TypeError`.

## Example

`python examples/example.py` shows bytes, JSON, XML, ctypes and the C header from one class.

## Develop

```
python -m unittest discover -s tests
```
