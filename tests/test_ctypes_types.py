"""Raw ctypes types as annotations. Layout is packed (the firstclass default)."""
from __future__ import annotations

import ctypes
import sys
import unittest

from firstclass import firstclass, uint8_t


@firstclass
class Ints:
    a: ctypes.c_int8
    b: ctypes.c_uint8
    c: ctypes.c_int16
    d: ctypes.c_uint16
    e: ctypes.c_int32
    f: ctypes.c_uint32
    g: ctypes.c_int64
    h: ctypes.c_uint64
    i: ctypes.c_int
    j: ctypes.c_uint
    k: ctypes.c_long
    l: ctypes.c_ulong
    m: ctypes.c_longlong
    n: ctypes.c_size_t
    o: ctypes.c_bool
    p: ctypes.c_float
    q: ctypes.c_double
    r: ctypes.c_char
    s: ctypes.c_byte
    t: ctypes.c_ubyte
    u: ctypes.c_short
    v: ctypes.c_ssize_t


class RefInts(ctypes.Structure):
    _pack_ = 1
    if sys.version_info >= (3, 14):
        _layout_ = "ms"  # 3.14 warns when _pack_ is set without it
    _fields_ = [(n, getattr(ctypes, t)) for n, t in zip(
        "abcdefghijklmnopqrstuv",
        ["c_int8", "c_uint8", "c_int16", "c_uint16", "c_int32", "c_uint32", "c_int64", "c_uint64",
         "c_int", "c_uint", "c_long", "c_ulong", "c_longlong", "c_size_t", "c_bool", "c_float",
         "c_double", "c_char", "c_byte", "c_ubyte", "c_short", "c_ssize_t"])]


@firstclass
class WithPointers:
    tag: uint8_t
    raw: ctypes.c_void_p
    ints: ctypes.POINTER(ctypes.c_int)
    name: ctypes.c_char_p
    after: ctypes.c_uint16


class RefPointers(ctypes.Structure):
    _pack_ = 1
    if sys.version_info >= (3, 14):
        _layout_ = "ms"  # 3.14 warns when _pack_ is set without it
    _fields_ = [("tag", ctypes.c_uint8), ("raw", ctypes.c_void_p),
                ("ints", ctypes.POINTER(ctypes.c_int)), ("name", ctypes.c_void_p),
                ("after", ctypes.c_uint16)]


@firstclass
class Arrays:
    words: ctypes.c_uint16 * 3
    text: ctypes.c_char * 5
    raw: ctypes.c_ubyte * 2


class RefArrays(ctypes.Structure):
    _pack_ = 1
    if sys.version_info >= (3, 14):
        _layout_ = "ms"  # 3.14 warns when _pack_ is set without it
    _fields_ = [("words", ctypes.c_uint16 * 3), ("text", ctypes.c_char * 5), ("raw", ctypes.c_ubyte * 2)]


class CtypesIntegers(unittest.TestCase):
    VALUES = (-8, 200, -1000, 60000, -2**31, 2**32 - 1, -2**62, 2**64 - 1, -5, 7, -3, 9, -11, 13,
              True, 1.5, 2.25, "x", -7, 255, -300, -17)

    def test_packed_size_matches_ctypes(self):
        self.assertEqual(Ints.sizeof(), ctypes.sizeof(RefInts))

    def test_bytes_match_reference_struct(self):
        ref = RefInts(*[v.encode() if isinstance(v, str) else v for v in self.VALUES])
        self.assertEqual(Ints(*self.VALUES).to_bytes(), bytes(ref))

    def test_roundtrip_all_formats(self):
        v = Ints(*self.VALUES)
        self.assertEqual(Ints.from_bytes(v.to_bytes()), v)
        self.assertEqual(Ints.from_json(v.to_json()), v)
        self.assertEqual(Ints.from_xml(v.to_xml()), v)
        self.assertEqual(Ints.from_ctypes(v.to_ctypes()), v)
        self.assertEqual(bytes(v.to_ctypes()), v.to_bytes())

    def test_from_foreign_struct(self):
        ref = RefInts(*[x.encode() if isinstance(x, str) else x for x in self.VALUES])
        self.assertEqual(Ints.from_bytes(bytes(ref)), Ints(*self.VALUES))

    def test_range_checked(self):
        bad = list(self.VALUES)
        bad[1] = 256  # c_uint8
        with self.assertRaises(ValueError):
            Ints(*bad).to_bytes()

    def test_c_header(self):
        h = Ints.c_header()
        for line in ("int8_t a;", "uint8_t b;", "int64_t g;", "uint64_t h;", "bool o;", "float p;",
                     "double q;"):
            self.assertIn(line, h)


class CtypesPointers(unittest.TestCase):
    def test_packed_layout_matches_ctypes(self):
        self.assertEqual(WithPointers.sizeof(), ctypes.sizeof(RefPointers))
        self.assertEqual(WithPointers.sizeof(), 1 + 3 * ctypes.sizeof(ctypes.c_void_p) + 2)

    def test_bytes_hold_raw_addresses(self):
        buf = (ctypes.c_int * 3)(10, 20, 30)
        addr = ctypes.addressof(buf)
        w = WithPointers(7, 0, addr, 0, 9)
        ref = RefPointers(7, None, ctypes.cast(addr, ctypes.POINTER(ctypes.c_int)), None, 9)
        self.assertEqual(w.to_bytes(), bytes(ref))
        self.assertEqual(WithPointers.from_bytes(w.to_bytes()), w)

    def test_to_ctypes_pointer_is_dereferenceable(self):
        buf = (ctypes.c_int * 3)(10, 20, 30)
        w = WithPointers(1, 0, ctypes.addressof(buf), 0, 2)
        c = w.to_ctypes()
        self.assertEqual([c.ints[i] for i in range(3)], [10, 20, 30])
        self.assertIsNone(c.raw)
        self.assertEqual(WithPointers.from_ctypes(c), w)

    def test_null_and_void(self):
        w = WithPointers(0, 0, 0, 0, 0)
        self.assertEqual(w.to_bytes(), bytes(RefPointers()))
        c = WithPointers(0, 0xDEAD0000, 0, 0, 0).to_ctypes()
        self.assertEqual(c.raw, 0xDEAD0000)

    def test_char_p_is_address(self):
        s = ctypes.create_string_buffer(b"hi")
        w = WithPointers(0, 0, 0, ctypes.addressof(s), 0)
        self.assertEqual(ctypes.string_at(w.to_ctypes().name), b"hi")

    def test_pointer_json_and_range(self):
        w = WithPointers(1, 5, 6, 7, 8)
        self.assertEqual(WithPointers.from_json(w.to_json()), w)
        with self.assertRaises(ValueError):
            WithPointers(0, -1, 0, 0, 0).to_bytes()
        with self.assertRaises(ValueError):
            WithPointers(0, 2 ** (8 * ctypes.sizeof(ctypes.c_void_p)), 0, 0, 0).to_bytes()
        with self.assertRaises(TypeError):
            WithPointers(0, "p", 0, 0, 0).to_bytes()

    def test_c_header(self):
        h = WithPointers.c_header()
        self.assertIn("void *raw;", h)
        self.assertIn("void *ints;", h)

    def test_swapped_byteorder_bytes_ok_ctypes_refused(self):
        @firstclass(byteorder="big" if sys.byteorder == "little" else "little")
        class P:
            p: ctypes.c_void_p
        P(0x1234).to_bytes()
        with self.assertRaises(TypeError):
            P.ctypes_type()


class CtypesArrays(unittest.TestCase):
    def test_layout_and_bytes(self):
        a = Arrays([1, 2, 3], "abcde", [9, 8])
        ref = RefArrays((ctypes.c_uint16 * 3)(1, 2, 3), b"abcde", (ctypes.c_ubyte * 2)(9, 8))
        self.assertEqual(a.to_bytes(), bytes(ref))
        self.assertEqual(Arrays.from_bytes(bytes(ref)), a)
        self.assertEqual(Arrays.from_ctypes(a.to_ctypes()), a)


if __name__ == "__main__":
    unittest.main()
