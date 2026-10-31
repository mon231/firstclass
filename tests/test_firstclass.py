from __future__ import annotations

import ctypes
import dataclasses
import struct
import unittest

from firstclass import (
    array_t, bool_t, bytes_t, char_array, char_t, firstclass, float32_t, float64_t,
    int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t,
)


@firstclass
class Point:
    x: int16_t
    y: int16_t


@firstclass
class Packet:
    kind: uint8_t
    flag: bool_t
    seq: uint32_t
    temp: float32_t
    name: char_array[6]
    mac: bytes_t[4]
    samples: array_t[uint16_t, 3]
    origin: Point
    corners: array_t[Point, 2]


@firstclass(byteorder="big")
class Big:
    a: uint16_t
    b: int32_t


@firstclass(align=True)
class Aligned:
    a: uint8_t
    b: uint32_t
    c: uint8_t
    d: uint64_t
    e: uint16_t


@firstclass(frozen=True, slots=True)
class Frozen:
    a: uint8_t
    b: uint8_t = 7


@firstclass
class Loose:
    a: uint8_t
    label: str = "x"


def make_packet():
    return Packet(1, True, 0xDEADBEEF, 1.5, "hello", b"\x01\x02\x03\x04", [1, 2, 3],
                  Point(-1, 2), [Point(3, 4), Point(5, -6)])


class Bytes(unittest.TestCase):
    def test_layout_is_packed_little(self):
        p = make_packet()
        expect = (struct.pack("<B?If", 1, True, 0xDEADBEEF, 1.5) + b"hello\0" + b"\x01\x02\x03\x04"
                  + struct.pack("<3H", 1, 2, 3) + struct.pack("<2h", -1, 2)
                  + struct.pack("<2h2h", 3, 4, 5, -6))
        self.assertEqual(p.to_bytes(), expect)
        self.assertEqual(Packet.sizeof(), len(expect))

    def test_roundtrip(self):
        p = make_packet()
        self.assertEqual(Packet.from_bytes(p.to_bytes()), p)

    def test_big_endian(self):
        self.assertEqual(Big(1, -2).to_bytes(), struct.pack(">Hi", 1, -2))
        self.assertEqual(Big.from_bytes(struct.pack(">Hi", 1, -2)), Big(1, -2))

    def test_align_padding(self):
        a = Aligned(1, 2, 3, 4, 5)
        self.assertEqual(a.to_bytes(), struct.pack("<B3xIB7xQH6x", 1, 2, 3, 4, 5))
        self.assertEqual(Aligned.sizeof(), 32)
        self.assertEqual(Aligned.from_bytes(a.to_bytes()), a)

    def test_range_errors(self):
        with self.assertRaisesRegex(ValueError, "Point.x"):
            Point(40000, 0).to_bytes()
        with self.assertRaises(ValueError):
            Frozen(256).to_bytes()
        with self.assertRaises(TypeError):
            Frozen(True).to_bytes()
        with self.assertRaises(TypeError):
            Frozen(1.5).to_bytes()

    def test_char_array_errors(self):
        p = make_packet()
        with self.assertRaises(ValueError):
            p.name = "toolong!"
        p.name = "exact!"
        self.assertEqual(Packet.from_bytes(p.to_bytes()).name, "exact!")

    def test_bytes_exact(self):
        p = make_packet()
        with self.assertRaises(ValueError):
            p.mac = b"\x00"
        self.assertEqual(p.mac, b"\x01\x02\x03\x04")  # the rejected assignment changed nothing

    def test_wrong_length_in(self):
        with self.assertRaises(ValueError):
            Point.from_bytes(b"\x00\x00\x00")

    def test_bool_stored_one_byte(self):
        @firstclass
        class B:
            f: bool_t
        self.assertEqual(B(True).to_bytes(), b"\x01")
        self.assertEqual(B.from_bytes(b"\x00"), B(False))

    def test_all_int_widths(self):
        @firstclass
        class W:
            a: uint8_t
            b: uint16_t
            c: uint32_t
            d: uint64_t
            e: int8_t
            f: int16_t
            g: int32_t
            h: int64_t
            i: float64_t
            j: char_t
        w = W(255, 65535, 2**32 - 1, 2**64 - 1, -128, -32768, -2**31, -2**63, 2.5, "z")
        self.assertEqual(W.from_bytes(w.to_bytes()), w)
        self.assertEqual(W.sizeof(), 1 + 2 + 4 + 8 + 1 + 2 + 4 + 8 + 8 + 1)

    def test_plain_field_no_binary(self):
        with self.assertRaisesRegex(TypeError, "Loose.label"):
            Loose(1).to_bytes()


class Dataclass(unittest.TestCase):
    def test_is_dataclass(self):
        self.assertTrue(dataclasses.is_dataclass(Packet))
        self.assertEqual(Frozen(1).b, 7)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            Frozen(1).a = 2

    def test_slots_frozen_roundtrip(self):
        self.assertEqual(Frozen.from_bytes(Frozen(3, 4).to_bytes()), Frozen(3, 4))

    def test_name_clash(self):
        with self.assertRaises(TypeError):
            @firstclass
            class Bad:
                to_bytes: uint8_t


class Json(unittest.TestCase):
    def test_roundtrip(self):
        p = make_packet()
        self.assertEqual(Packet.from_json(p.to_json()), p)

    def test_shape(self):
        d = make_packet().to_dict()
        self.assertEqual(d["mac"], "01020304")
        self.assertEqual(d["origin"], {"x": -1, "y": 2})
        self.assertEqual(d["corners"][1], {"x": 5, "y": -6})

    def test_validation(self):
        with self.assertRaises(ValueError):
            Point.from_dict({"x": 1, "y": 2, "z": 3})
        with self.assertRaises(ValueError):
            Point.from_dict({"x": 99999, "y": 2})
        with self.assertRaises(TypeError):
            Point.from_dict({"x": "1", "y": 2})

    def test_plain_fields_json(self):
        self.assertEqual(Loose.from_json(Loose(1, "q").to_json()), Loose(1, "q"))


class Xml(unittest.TestCase):
    def test_roundtrip(self):
        p = make_packet()
        self.assertEqual(Packet.from_xml(p.to_xml()), p)
        self.assertEqual(Packet.from_xml(p.to_xml(indent=2)), p)

    def test_shape(self):
        self.assertEqual(Point(1, 2).to_xml(), "<Point><x>1</x><y>2</y></Point>")

    def test_unknown_element(self):
        with self.assertRaises(ValueError):
            Point.from_xml("<Point><x>1</x><y>2</y><z>3</z></Point>")

    def test_plain_fields_xml(self):
        self.assertEqual(Loose.from_xml(Loose(1, "q").to_xml()), Loose(1, "q"))

    def test_empty_char_array(self):
        @firstclass
        class S:
            s: char_array[4]
        self.assertEqual(S.from_xml(S("").to_xml()), S(""))


class Ctypes(unittest.TestCase):
    def test_bytes_match(self):
        p = make_packet()
        c = p.to_ctypes()
        self.assertEqual(ctypes.sizeof(c), Packet.sizeof())
        self.assertEqual(bytes(c), p.to_bytes())

    def test_roundtrip(self):
        p = make_packet()
        self.assertEqual(Packet.from_ctypes(p.to_ctypes()), p)

    def test_from_foreign_buffer(self):
        p = make_packet()
        c = Packet.ctypes_type().from_buffer_copy(p.to_bytes())
        self.assertEqual(Packet.from_ctypes(c), p)

    def test_big_endian(self):
        c = Big(1, -2).to_ctypes()
        self.assertEqual(bytes(c), struct.pack(">Hi", 1, -2))
        self.assertEqual(Big.from_ctypes(c), Big(1, -2))

    def test_aligned_matches_native(self):
        a = Aligned(1, 2, 3, 4, 5)
        c = a.to_ctypes()
        self.assertEqual(ctypes.sizeof(c), Aligned.sizeof())
        self.assertEqual(bytes(c), a.to_bytes())

    def test_array_of_char_arrays(self):
        @firstclass
        class Names:
            n: array_t[char_array[4], 2]
            b: array_t[bytes_t[2], 2]
        v = Names(["ab", "cdef"], [b"\x01\x02", b"\x03\x04"])
        self.assertEqual(Names.from_ctypes(v.to_ctypes()), v)
        self.assertEqual(bytes(v.to_ctypes()), v.to_bytes())


class CHeader(unittest.TestCase):
    def test_header(self):
        h = Packet.c_header()
        self.assertIn("struct Point {\n    int16_t x;\n    int16_t y;\n};", h)
        self.assertLess(h.index("struct Point {"), h.index("struct Packet {"))
        self.assertEqual(h.count("struct Point {"), 1)
        for line in ("    uint8_t kind;", "    bool flag;", "    float temp;", "    char name[6];",
                     "    uint8_t mac[4];", "    uint16_t samples[3];", "    struct Point origin;",
                     "    struct Point corners[2];"):
            self.assertIn(line, h)
        self.assertIn("#pragma pack(push, 1)", h)

    def test_aligned_no_pragma(self):
        self.assertNotIn("#pragma", Aligned.c_header())


class Types(unittest.TestCase):
    def test_bad_sizes(self):
        with self.assertRaises(TypeError):
            char_array[0]
        with self.assertRaises(TypeError):
            array_t[uint8_t]

    def test_bad_byteorder(self):
        with self.assertRaises(ValueError):
            @firstclass(byteorder="middle")
            class X:
                a: uint8_t


if __name__ == "__main__":
    unittest.main()


class UnpackFrom(unittest.TestCase):
    def test_offset_and_trailing_bytes(self):
        a, b = Point(1, -2), Point(300, 4)
        stream = b"\xff\xff" + a.to_bytes() + b.to_bytes() + b"tail"
        self.assertEqual(Point.unpack_from(stream, 2), a)
        self.assertEqual(Point.unpack_from(stream, 2 + Point.sizeof()), b)
        self.assertEqual(Point.unpack_from(a.to_bytes()), a)  # default offset 0

    def test_walk_a_stream(self):
        pts = [Point(i, -i) for i in range(5)]
        stream = b"".join(p.to_bytes() for p in pts)
        got = [Point.unpack_from(stream, i * Point.sizeof()) for i in range(5)]
        self.assertEqual(got, pts)

    def test_buffer_types(self):
        p = make_packet()
        raw = b"\x00" * 3 + p.to_bytes()
        for buf in (raw, bytearray(raw), memoryview(raw)):
            self.assertEqual(Packet.unpack_from(buf, 3), p)
        import array
        self.assertEqual(Point.unpack_from(array.array("h", [7, 8])), Point(7, 8))

    def test_too_short_and_bad_offset(self):
        raw = Point(1, 2).to_bytes()
        with self.assertRaises(ValueError):
            Point.unpack_from(raw, 1)
        with self.assertRaises(ValueError):
            Point.unpack_from(raw, -1)
        with self.assertRaises(ValueError):
            Point.unpack_from(raw, 100)
        with self.assertRaises(ValueError):
            Point.unpack_from(raw, True)

    def test_nested_and_aligned(self):
        a = Aligned(1, 2, 3, 4, 5)
        stream = b"\x00" * 5 + a.to_bytes()
        self.assertEqual(Aligned.unpack_from(stream, 5), a)
