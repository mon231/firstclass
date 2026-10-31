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
| `wchar_t` / `wchar_array[N]` | `uint16_t x[N]` | 2 (2*N), UTF-16, NUL-padded |
| `bytes_t[N]` | `uint8_t x[N]` | N |
| `array_t[T, N]` | `T x[N]` | N * size(T) |
| `enum_t[Color, uint8_t]` | `uint8_t x` (+ `enum Color {..}` in header) | storage size |
| `bitfield[uint8_t, 3]` | `uint8_t x : 3` | shares a storage unit |
| another `@firstclass` class | `struct Name` | its size |

Values are checked: out of range ints, over-long strings and wrong lengths raise
`ValueError`; wrong Python types raise `TypeError`. Nothing is truncated or coerced silently.

**Enums.** Int-valued `Enum`/`IntEnum`/`IntFlag` classes. `enum_t[Color, uint8_t]` picks the width;
a bare `Color` annotation is 32-bit like a C `int` (unsigned unless a value is negative). Values
are stored, and written to JSON/XML, as ints; reading an unknown value raises `ValueError`.
`c_header()` emits `enum Color { Color_RED = 1, ... };`.

**Bitfields.** `bitfield[uint8_t, 3]` is 3 bits of a `uint8_t` unit, like `uint8_t x : 3`.
Neighbouring bitfields with the same storage type share a unit while they fit; otherwise a new
unit starts. Little-endian fills from the lowest bit, big-endian from the highest (what ctypes
does). Signed storage (`int8_t`) gives signed fields. Mixing storage types in one run is not
portable across C compilers; keep a run on one type. No arrays of bitfields.

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
| `Cls.unpack_from(buffer, offset=0)` | like `struct.unpack_from`: read one object at `offset`, trailing bytes are fine; any buffer type |
| `Cls.sizeof()` | size in bytes |
| `obj.to_dict()` / `Cls.from_dict(d)` | JSON-compatible dict (bytes as hex) |
| `obj.to_json(**kw)` / `Cls.from_json(s)` | `kw` goes to `json.dumps` |
| `obj.to_xml(indent=None)` / `Cls.from_xml(s)` | `<Packet><kind>1</kind>...</Packet>`, arrays as `<item>` |
| `obj.to_ctypes()` / `Cls.from_ctypes(c)` / `Cls.ctypes_type()` | `ctypes.Structure` with matching layout |
| `Cls.c_header()` | C source for the struct(s), nested ones first |
| `Cls.rust_source()` | Rust `#[repr(C)]` source for the struct(s) and enums |

## Import C/C++ headers

```python
from firstclass import from_header          # also: firstclass.from_header

t = from_header("protocol.h")               # or from_header(text="struct P { ... };")
pkt = t.Packet(1, True, ...)                # structs -> @firstclass classes
t.Color.RED                                 # enums / enum class -> IntEnum
t.constants["MAX_LEN"]                      # integer #defines, consts, enum members
t.skipped                                   # {"name": "reason"} for anything not convertible
```

Understood: `struct`/`class`/`typedef struct`, nested and anonymous-member structs, `enum` and
`enum class : type` (explicit values, implicit increments, constant expressions), bitfields, arrays
(multi-dimensional, sizes from `#define`/enum/`const`/`sizeof`), `char[N]` -> `char_array`,
`uint8_t[N]` -> `bytes_t`, pointers and function pointers (stored as addresses), `typedef`/`using`,
`#define`/`#if`/`#ifdef`/`#elif` (preset macros with `defines={...}`), `namespace`, `extern "C"`,
`#pragma pack(push/pop/N)` and `__attribute__((packed))`. Layout follows C: natural alignment
padding unless the struct is packed; force it with `align=True/False`. Options: `byteorder`,
`long_size` (size of C `long`; default is this machine's).

Skipped with a reason instead of failing: unions, flexible array members, unnamed padding
bitfields, inheritance, virtual members, unknown types (`#include`d headers are not read), `pack(2)`
/`pack(4)`, `aligned(N)`, `long double`, empty structs, anything depending on a skipped struct.
Plain C enums are 32-bit like C unless `: type` is given. C field names that are Python keywords
(`class`, `from`, ...) get a trailing underscore (`class_`).

## Validation

A value that cannot be encoded never becomes an instance. Fields are checked when the object is
created and on every attribute assignment, so the mistake surfaces where it was made, not later in
`to_bytes()`:

```python
Header(kind=300, tag="x")    # ValueError: Header.kind: 300 out of range for uint8_t [0, 255]
Header(kind=1, tag="ab")     # ValueError: Header.tag: 'ab' is 2 bytes, char_array[1] holds 1
h.name = "far too long"      # ValueError, and h.name keeps its old value
```

Wrong type -> `TypeError`, wrong value or size -> `ValueError`. Sequences and bytes-likes are
normalised (`(1, 2)` becomes `[1, 2]`, `bytearray` becomes `bytes`) and copied. The one gap is
mutating a container in place (`obj.nums[0] = 300`), which is not an assignment to the field;
`to_bytes()` still refuses it. Applies to hand-written classes and to classes made by
`from_header` / `from_rust`.

## Comparison

Like a dataclass, `==` / `!=` compare field by field. Unlike a plain dataclass, ordering is on by
default: `<`, `<=`, `>`, `>=` compare fields in declaration order, like tuples, so you can `sorted()`,
`min()` and `max()` instances. Comparing with another class raises `TypeError` (`==` is just `False`).
Switch it off with `@firstclass(order=False)`; `eq=False` turns off both.

## Rust

```python
from firstclass import from_rust

t = from_rust("src/protocol.rs")            # or from_rust(text="...")
t.Header, t.Status, t.constants, t.skipped  # same result object as from_header
print(t.Header.rust_source())               # and back: any firstclass class -> Rust definitions
```

`from_rust` reads simple `#[repr(C)]` structs (C layout with padding) and `#[repr(C, packed)]`
structs (no padding), plus C-like enums with `#[repr(u8)]`, `#[repr(i32)]` or `#[repr(C)]` and
plain discriminants (constant expressions, `as` casts, `const`s allowed). Types: `u8`..`u64`,
`i8`..`i64`, `usize`, `isize`, `f32`, `f64`, `bool`, `c_int` and the other `c_*` names, `[T; N]`
(`N` may be a `const` expression), nested structs, type aliases, tuple structs, raw pointers,
references, `Box`, `Option<&T>` and `fn` pointers (stored as addresses). `[u8; N]` becomes
`bytes_t[N]`, `[c_char; N]` becomes `char_array[N]`, `wchar_t` the 2-byte UTF-16 types. Modules are
flattened and `#[cfg(test)]` items are ignored.

Skipped with a reason in `.skipped`: structs without `#[repr(C)]` (Rust may reorder their fields;
pass `assume_repr_c=True` to read them anyway), `repr(transparent)`, `repr(align(N))`,
`packed(N)` for N > 1, generics, unions, `char`, `u128`, `String`/`Vec`/slices/tuples, enums with
data. Options: `byteorder`, `align`, `assume_repr_c`, `long_size`.

`Cls.rust_source()` writes `#[repr(C)]` / `#[repr(C, packed)]` structs, `#[repr(u8)]` enums,
`[u8; N]` for strings and bytes, `[u16; N]` for wide strings, `*mut c_void` for pointers and
`r#` for keyword names. Rust has no bitfields, so each group of bitfields becomes one integer
field with the member names in a comment.

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

`python examples/example.py` defines a class in Python, then imports `examples/protocol.h` (enums, `enum class`,
bitfields, `#pragma pack` structs) and uses the generated classes.

## Develop

```
python -m unittest discover -s tests
```
