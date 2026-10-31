from __future__ import annotations

import ctypes
import enum
import sys
import unittest

from firstclass import (
    array_t, bitfield, enum_t, firstclass, int8_t, uint8_t, uint16_t, uint32_t,
)


class Color(enum.IntEnum):
    RED = 1
    GREEN = 2
    BLUE = 300


class Mode(enum.Enum):
    OFF = 0
    ON = 1


class Perm(enum.IntFlag):
    R = 4
    W = 2
    X = 1


class Signed(enum.IntEnum):
    NEG = -2
    POS = 5


@firstclass
class Enums:
    color: enum_t[Color, uint16_t]
    mode: enum_t[Mode, uint8_t]
    perm: enum_t[Perm, uint8_t]
    bare: Signed
    colors: array_t[enum_t[Color, uint16_t], 2]


@firstclass
class Bits:
    a: bitfield[uint8_t, 3]
    b: bitfield[uint8_t, 5]
    c: bitfield[uint8_t, 4]
    d: bitfield[uint8_t, 4]
    tail: uint8_t


@firstclass(byteorder="big")
class BitsBE:
    a: bitfield[uint16_t, 3]
    b: bitfield[uint16_t, 9]
    c: bitfield[uint16_t, 5]   # 3+9+5 > 16: starts a new unit


class Ref(ctypes.Structure):
    _pack_ = 1
    if sys.version_info >= (3, 14):
        _layout_ = "ms"
    _fields_ = [("a", ctypes.c_uint8, 3), ("b", ctypes.c_uint8, 5), ("c", ctypes.c_uint8, 4),
                ("d", ctypes.c_uint8, 4), ("tail", ctypes.c_uint8)]


class RefBE(ctypes.BigEndianStructure):
    _pack_ = 1
    if sys.version_info >= (3, 14):
        _layout_ = "ms"
    _fields_ = [("a", ctypes.c_uint16, 3), ("b", ctypes.c_uint16, 9), ("c", ctypes.c_uint16, 5)]


@firstclass
class SignedBits:
    x: bitfield[int8_t, 3]
    y: bitfield[int8_t, 5]


class SignedRef(ctypes.Structure):
    _pack_ = 1
    if sys.version_info >= (3, 14):
        _layout_ = "ms"
    _fields_ = [("x", ctypes.c_int8, 3), ("y", ctypes.c_int8, 5)]


@firstclass
class MixedStorage:
    a: bitfield[uint8_t, 4]
    b: bitfield[uint16_t, 4]   # different storage type: new unit
    c: bitfield[uint16_t, 12]


@firstclass(align=True)
class AlignedBits:
    flag: uint8_t
    a: bitfield[uint32_t, 4]
    b: bitfield[uint32_t, 28]
    z: uint8_t


class EnumTests(unittest.TestCase):
    def make(self):
        return Enums(Color.BLUE, Mode.ON, Perm.R | Perm.X, Signed.NEG, [Color.RED, Color.GREEN])

    def test_layout(self):
        e = self.make()
        self.assertEqual(e.to_bytes(), (300).to_bytes(2, "little") + b"\x01\x05"
                         + (-2).to_bytes(4, "little", signed=True) + b"\x01\x00\x02\x00")
        self.assertEqual(Enums.sizeof(), 2 + 1 + 1 + 4 + 4)

    def test_roundtrips(self):
        e = self.make()
        self.assertEqual(Enums.from_bytes(e.to_bytes()), e)
        self.assertEqual(Enums.from_json(e.to_json()), e)
        self.assertEqual(Enums.from_xml(e.to_xml()), e)
        self.assertEqual(Enums.from_ctypes(e.to_ctypes()), e)
        self.assertEqual(bytes(e.to_ctypes()), e.to_bytes())

    def test_types_come_back_as_members(self):
        got = Enums.from_bytes(self.make().to_bytes())
        self.assertIs(got.color, Color.BLUE)
        self.assertIs(got.mode, Mode.ON)
        self.assertEqual(got.perm, Perm.R | Perm.X)
        self.assertIs(got.bare, Signed.NEG)

    def test_json_uses_ints(self):
        d = self.make().to_dict()
        self.assertEqual((d["color"], d["mode"], d["perm"], d["bare"], d["colors"]), (300, 1, 5, -2, [1, 2]))

    def test_unknown_value_rejected(self):
        raw = bytearray(self.make().to_bytes())
        raw[2] = 9  # Mode has no 9
        with self.assertRaisesRegex(ValueError, "Enums.mode"):
            Enums.from_bytes(bytes(raw))
        with self.assertRaises(ValueError):
            Enums.from_dict({**self.make().to_dict(), "color": 7})

    def test_wrong_type_rejected(self):
        e = self.make()
        with self.assertRaises(TypeError):
            e.color = 300  # plain int, not a member
        with self.assertRaises(TypeError):
            Enums(300, Mode.ON, Perm.R, Signed.NEG, [Color.RED, Color.GREEN])

    def test_value_must_fit_storage(self):
        with self.assertRaises(ValueError):
            enum_t[Color, uint8_t]  # BLUE=300 does not fit in a byte

    def test_non_int_enum_rejected(self):
        class S(enum.Enum):
            A = "a"
        with self.assertRaises(TypeError):
            enum_t[S, uint8_t]

    def test_c_header(self):
        h = Enums.c_header()
        self.assertIn("enum Color {\n    Color_RED = 1,\n    Color_GREEN = 2,\n    Color_BLUE = 300,\n};", h)
        self.assertEqual(h.count("enum Color {"), 1)
        self.assertLess(h.index("enum Mode {"), h.index("struct Enums {"))
        self.assertIn("    uint16_t color;", h)
        self.assertIn("    int32_t bare;", h)


class BitfieldTests(unittest.TestCase):
    def test_matches_ctypes_little(self):
        b = Bits(5, 17, 9, 15, 200)
        ref = Ref(5, 17, 9, 15, 200)
        self.assertEqual(Bits.sizeof(), ctypes.sizeof(Ref))
        self.assertEqual(b.to_bytes(), bytes(ref))
        self.assertEqual(Bits.from_bytes(bytes(ref)), b)

    def test_bit_positions(self):
        # a=0b101 in bits 0-2, b=0b10001 in bits 3-7
        self.assertEqual(Bits(5, 17, 0, 0, 0).to_bytes()[0], 0b10001_101)

    def test_matches_ctypes_big(self):
        b = BitsBE(5, 300, 21)
        self.assertEqual(BitsBE.sizeof(), ctypes.sizeof(RefBE))
        self.assertEqual(b.to_bytes(), bytes(RefBE(5, 300, 21)))
        self.assertEqual(BitsBE.from_bytes(bytes(RefBE(5, 300, 21))), b)

    def test_signed(self):
        s = SignedBits(-3, -16)
        self.assertEqual(s.to_bytes(), bytes(SignedRef(-3, -16)))
        self.assertEqual(SignedBits.from_bytes(s.to_bytes()), s)
        self.assertEqual(SignedBits.from_bytes(bytes(SignedRef(3, 15))), SignedBits(3, 15))

    def test_all_formats(self):
        b = Bits(7, 31, 15, 1, 255)
        self.assertEqual(Bits.from_json(b.to_json()), b)
        self.assertEqual(Bits.from_xml(b.to_xml()), b)
        self.assertEqual(Bits.from_ctypes(b.to_ctypes()), b)
        self.assertEqual(bytes(b.to_ctypes()), b.to_bytes())

    def test_range_checked(self):
        with self.assertRaisesRegex(ValueError, "Bits.a"):
            Bits(8, 0, 0, 0, 0).to_bytes()
        with self.assertRaises(ValueError):
            SignedBits(4, 0).to_bytes()
        with self.assertRaises(TypeError):
            Bits(True, 0, 0, 0, 0).to_bytes()

    def test_different_storage_starts_new_unit(self):
        m = MixedStorage(0xA, 0x5, 0xABC)
        self.assertEqual(MixedStorage.sizeof(), 1 + 2)
        self.assertEqual(m.to_bytes(), bytes([0x0A]) + (0x5 | (0xABC << 4)).to_bytes(2, "little"))
        self.assertEqual(MixedStorage.from_bytes(m.to_bytes()), m)

    def test_aligned(self):
        a = AlignedBits(1, 0xF, 0xABCDEF, 2)
        self.assertEqual(AlignedBits.sizeof(), 12)  # 1 + 3 pad + 4 + 1 + 3 pad
        self.assertEqual(AlignedBits.from_bytes(a.to_bytes()), a)
        self.assertEqual(ctypes.sizeof(a.to_ctypes()), 12)
        self.assertEqual(bytes(a.to_ctypes()), a.to_bytes())

    def test_c_header(self):
        h = Bits.c_header()
        self.assertIn("    uint8_t a : 3;\n    uint8_t b : 5;", h)

    def test_bad_definitions(self):
        with self.assertRaises(ValueError):
            bitfield[uint8_t, 9]
        with self.assertRaises(TypeError):
            bitfield[float, 3]
        with self.assertRaises(TypeError):
            array_t[bitfield[uint8_t, 3], 2]
        with self.assertRaises(TypeError):
            bitfield[uint8_t]


if __name__ == "__main__":
    unittest.main()
