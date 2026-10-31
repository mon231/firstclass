import array
import ctypes
import dataclasses
import json
import struct
import unittest
from typing import ClassVar

from firstclass import (
    array_t, bitfield, bool_t, bytes_t, char_array, char_t, enum_t, firstclass, float32_t, float64_t,
    int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t,
)
import enum


class Level(enum.IntEnum):
    LOW = 0
    HIGH = 9


@firstclass
class Small:
    a: uint8_t
    b: uint8_t = 5


@firstclass
class Empty:
    pass


@firstclass
class WithClassVar:
    x: uint8_t
    count: ClassVar[int] = 0


@firstclass
class LittleInner:
    v: uint16_t


@firstclass(byteorder="big")
class BigInner:
    v: uint16_t


@firstclass
class Mixed:
    little: LittleInner
    big: BigInner


@firstclass
class Base:
    a: uint8_t


@firstclass
class Child(Base):
    b: uint16_t


@firstclass
class HasNoInit:
    a: uint8_t
    computed: uint8_t = dataclasses.field(default=0, init=False)


@firstclass(frozen=True)
class FrozenNoInit:
    a: uint8_t
    computed: uint8_t = dataclasses.field(default=0, init=False)


@firstclass
class Defaults:
    items: array_t[uint8_t, 2] = dataclasses.field(default_factory=lambda: [1, 2])
    name: char_array[4] = "ab"


INT_TYPES = {
    "uint8_t": (uint8_t, 0, 2 ** 8 - 1), "uint16_t": (uint16_t, 0, 2 ** 16 - 1),
    "uint32_t": (uint32_t, 0, 2 ** 32 - 1), "uint64_t": (uint64_t, 0, 2 ** 64 - 1),
    "int8_t": (int8_t, -2 ** 7, 2 ** 7 - 1), "int16_t": (int16_t, -2 ** 15, 2 ** 15 - 1),
    "int32_t": (int32_t, -2 ** 31, 2 ** 31 - 1), "int64_t": (int64_t, -2 ** 63, 2 ** 63 - 1),
}


def single(tp):
    @firstclass
    class One:
        v: tp
    return One


class IntegerBounds(unittest.TestCase):
    def test_every_width_accepts_bounds_and_rejects_outside(self):
        for name, (tp, lo, hi) in INT_TYPES.items():
            with self.subTest(name):
                One = single(tp)
                for ok in (lo, hi, 0 if lo <= 0 <= hi else lo):
                    self.assertEqual(One.from_bytes(One(ok).to_bytes()).v, ok)
                for bad in (lo - 1, hi + 1):
                    with self.assertRaises(ValueError):
                        One(bad).to_bytes()
                    with self.assertRaises(ValueError):
                        One.from_dict({"v": bad})

    def test_non_int_values_rejected(self):
        One = single(uint16_t)
        for bad in (1.0, "1", None, True, b"1", [1]):
            with self.subTest(bad):
                with self.assertRaises(TypeError):
                    One(bad).to_bytes()

    def test_bool_t_is_strict(self):
        One = single(bool_t)
        for bad in (0, 1, "true", None):
            with self.subTest(bad):
                with self.assertRaises(TypeError):
                    One(bad).to_bytes()
        self.assertEqual(One.from_bytes(b"\x02").v, True)  # any non-zero byte reads as True

    def test_endianness_per_width(self):
        for name, (tp, lo, hi) in INT_TYPES.items():
            with self.subTest(name):
                @firstclass(byteorder="big")
                class B:
                    v: tp
                size = B.sizeof()
                self.assertEqual(B(hi).to_bytes(), hi.to_bytes(size, "big", signed=lo < 0))


class Floats(unittest.TestCase):
    def test_float32_precision_and_overflow(self):
        One = single(float32_t)
        self.assertEqual(One.from_bytes(One(0.1).to_bytes()).v, struct.unpack("<f", struct.pack("<f", 0.1))[0])
        with self.assertRaises(ValueError):
            One(1e40).to_bytes()

    def test_float64_special_values(self):
        One = single(float64_t)
        for v in (float("inf"), float("-inf"), 5e-324, 1.7976931348623157e308, -0.0):
            self.assertEqual(One.from_bytes(One(v).to_bytes()).v, v)
        nan = One.from_bytes(One(float("nan")).to_bytes()).v
        self.assertNotEqual(nan, nan)

    def test_int_accepted_for_float(self):
        self.assertEqual(single(float64_t)(3).to_bytes(), struct.pack("<d", 3))

    def test_bool_and_str_rejected_for_float(self):
        for bad in (True, "1.0", None):
            with self.assertRaises(TypeError):
                single(float32_t)(bad).to_bytes()

    def test_xml_float_text_is_exact(self):
        One = single(float64_t)
        self.assertEqual(One.from_xml(One(0.1 + 0.2).to_xml()).v, 0.1 + 0.2)


class Strings(unittest.TestCase):
    def test_multibyte_counts_bytes_not_chars(self):
        One = single(char_array[6])
        self.assertEqual(One.from_bytes(One("héllo").to_bytes()).v, "héllo")  # 6 bytes
        with self.assertRaises(ValueError):
            One("héllo!").to_bytes()
        self.assertEqual(One("€€").to_bytes(), "€€".encode())  # exactly 6

    def test_exact_fit_has_no_terminator(self):
        One = single(char_array[3])
        self.assertEqual(One("abc").to_bytes(), b"abc")
        self.assertEqual(One.from_bytes(b"abc").v, "abc")

    def test_stops_at_first_nul(self):
        self.assertEqual(single(char_array[6]).from_bytes(b"ab\0cd\0").v, "ab")

    def test_nul_inside_value_rejected(self):
        with self.assertRaises(ValueError):
            single(char_array[6])("a\0b").to_bytes()

    def test_invalid_utf8_on_read(self):
        with self.assertRaises(ValueError):
            single(char_array[2]).from_bytes(b"\xff\xfe")

    def test_empty_string(self):
        One = single(char_array[4])
        self.assertEqual(One("").to_bytes(), b"\0\0\0\0")
        self.assertEqual(One.from_bytes(b"\0\0\0\0").v, "")

    def test_char_t(self):
        One = single(char_t)
        self.assertEqual(One("x").to_bytes(), b"x")
        with self.assertRaises(ValueError):
            One("xy").to_bytes()

    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            single(char_array[4])(b"abc").to_bytes()

    def test_xml_escaping(self):
        One = single(char_array[8])
        v = One("<&>\"'")
        self.assertEqual(One.from_xml(v.to_xml()), v)
        self.assertNotIn("<&>", v.to_xml())

    def test_json_unicode(self):
        One = single(char_array[8])
        self.assertEqual(One.from_json(One("é").to_json()), One("é"))


class ByteArrays(unittest.TestCase):
    def test_accepts_bytes_like_returns_bytes(self):
        One = single(bytes_t[3])
        for v in (b"abc", bytearray(b"abc"), memoryview(b"abc")):
            got = One.from_bytes(One(v).to_bytes()).v
            self.assertEqual(got, b"abc")
            self.assertIsInstance(got, bytes)

    def test_wrong_length_or_type(self):
        One = single(bytes_t[3])
        for bad, exc in ((b"ab", ValueError), (b"abcd", ValueError), ("abc", TypeError), ([1, 2, 3], TypeError)):
            with self.assertRaises(exc):
                One(bad).to_bytes()

    def test_json_hex_validation(self):
        One = single(bytes_t[2])
        self.assertEqual(One.from_dict({"v": "0aFF"}).v, b"\x0a\xff")
        for bad in ("0a", "zz00", 5):
            with self.assertRaises((ValueError, TypeError)):
                One.from_dict({"v": bad})


class Arrays(unittest.TestCase):
    def test_tuple_accepted_list_returned(self):
        One = single(array_t[uint8_t, 3])
        self.assertEqual(One.from_bytes(One((1, 2, 3)).to_bytes()).v, [1, 2, 3])

    def test_length_and_item_validation(self):
        One = single(array_t[uint8_t, 3])
        with self.assertRaises(ValueError):
            One([1, 2]).to_bytes()
        with self.assertRaises(ValueError):
            One([1, 2, 300]).to_bytes()
        with self.assertRaises(TypeError):
            One("abc").to_bytes()

    def test_array_of_arrays_of_enums(self):
        One = single(array_t[array_t[enum_t[Level, uint8_t], 2], 2])
        v = One([[Level.LOW, Level.HIGH], [Level.HIGH, Level.LOW]])
        self.assertEqual(v.to_bytes(), bytes([0, 9, 9, 0]))
        self.assertEqual(One.from_bytes(v.to_bytes()), v)
        self.assertEqual(One.from_json(v.to_json()), v)
        self.assertEqual(One.from_xml(v.to_xml()), v)

    def test_array_of_char_arrays_and_bytes(self):
        One = single(array_t[char_array[3], 2])
        self.assertEqual(One(["ab", "c"]).to_bytes(), b"ab\0c\0\0")
        self.assertEqual(One.from_bytes(b"ab\0c\0\0").v, ["ab", "c"])

    def test_array_of_nested_structs(self):
        One = single(array_t[Small, 2])
        v = One([Small(1, 2), Small(3, 4)])
        self.assertEqual(v.to_bytes(), b"\x01\x02\x03\x04")
        self.assertEqual(One.from_json(v.to_json()), v)

    def test_bad_params(self):
        with self.assertRaises(TypeError):
            array_t[uint8_t, 0]
        with self.assertRaises(TypeError):
            array_t[uint8_t, -1]
        with self.assertRaises(TypeError):
            array_t[uint8_t, "3"]
        with self.assertRaises(TypeError):
            bytes_t[0]
        with self.assertRaises(TypeError):
            char_array[True]


class DataclassBehaviour(unittest.TestCase):
    def test_defaults_and_factories(self):
        d = Defaults()
        self.assertEqual((d.items, d.name), ([1, 2], "ab"))
        self.assertIsNot(Defaults().items, Defaults().items)
        self.assertEqual(Defaults.from_bytes(d.to_bytes()), d)

    def test_eq_repr_replace_asdict(self):
        s = Small(1)
        self.assertEqual(s, Small(1, 5))
        self.assertNotEqual(s, Small(2))
        self.assertIn("Small(a=1, b=5)", repr(s))
        self.assertEqual(dataclasses.replace(s, b=9).to_bytes(), b"\x01\x09")
        self.assertEqual(dataclasses.asdict(s), {"a": 1, "b": 5})
        self.assertEqual([f.name for f in dataclasses.fields(Small)], ["a", "b"])

    def test_classvar_is_not_a_field(self):
        self.assertEqual(WithClassVar.sizeof(), 1)
        self.assertEqual(WithClassVar(3).to_bytes(), b"\x03")

    def test_inheritance_adds_fields(self):
        self.assertEqual(Child.sizeof(), 3)
        c = Child(1, 513)
        self.assertEqual(c.to_bytes(), b"\x01\x01\x02")
        self.assertEqual(Child.from_bytes(c.to_bytes()), c)
        self.assertEqual(Base.sizeof(), 1)

    def test_init_false_fields_are_still_read(self):
        h = HasNoInit.from_bytes(b"\x01\x07")
        self.assertEqual((h.a, h.computed), (1, 7))
        f = FrozenNoInit.from_bytes(b"\x01\x07")
        self.assertEqual((f.a, f.computed), (1, 7))
        self.assertEqual(f.to_bytes(), b"\x01\x07")

    def test_empty_class(self):
        self.assertEqual(Empty.sizeof(), 0)
        self.assertEqual(Empty().to_bytes(), b"")
        self.assertEqual(Empty.from_bytes(b""), Empty())
        self.assertEqual(Empty().to_json(), "{}")

    def test_mutation_then_serialise(self):
        s = Small(1)
        s.a = 200
        self.assertEqual(s.to_bytes(), b"\xc8\x05")

    def test_decorator_forms(self):
        self.assertTrue(dataclasses.is_dataclass(firstclass(Small)))
        with self.assertRaises(TypeError):
            firstclass(5)
        with self.assertRaises(ValueError):
            firstclass(byteorder="BIG")(type("X", (), {"__annotations__": {"a": uint8_t}}))

    def test_clash_with_every_generated_name(self):
        for name in ("to_bytes", "from_bytes", "unpack_from", "to_dict", "from_dict", "to_json", "from_json",
                     "to_xml", "from_xml", "to_ctypes", "from_ctypes", "ctypes_type", "c_header", "sizeof"):
            with self.subTest(name):
                cls = type("C", (), {"__annotations__": {name: uint8_t}})
                with self.assertRaises(TypeError):
                    firstclass(cls)

    def test_works_inside_function_with_global_types_only(self):
        @firstclass
        class Local:
            a: uint8_t
            b: array_t[uint8_t, 2]
        self.assertEqual(Local(1, [2, 3]).to_bytes(), b"\x01\x02\x03")


class Nesting(unittest.TestCase):
    def test_inner_keeps_its_own_byteorder(self):
        m = Mixed(LittleInner(0x0102), BigInner(0x0102))
        self.assertEqual(m.to_bytes(), b"\x02\x01\x01\x02")
        self.assertEqual(Mixed.from_bytes(m.to_bytes()), m)

    def test_wrong_nested_type(self):
        with self.assertRaises(TypeError):
            Mixed(BigInner(1), BigInner(2)).to_bytes()
        with self.assertRaises(TypeError):
            Mixed(LittleInner(1), {"v": 1}).to_bytes()

    def test_error_path_names_the_field(self):
        with self.assertRaisesRegex(ValueError, "Mixed.little: LittleInner.v"):
            Mixed(LittleInner(70000), BigInner(1)).to_bytes()

    def test_aligned_outer_with_aligned_inner(self):
        @firstclass(align=True)
        class In:
            a: uint8_t
            b: uint64_t

        @firstclass(align=True)
        class Out:
            x: uint8_t
            inner: In
            y: uint16_t

        class RIn(ctypes.Structure):
            _fields_ = [("a", ctypes.c_uint8), ("b", ctypes.c_uint64)]

        class ROut(ctypes.Structure):
            _fields_ = [("x", ctypes.c_uint8), ("inner", RIn), ("y", ctypes.c_uint16)]

        self.assertEqual(In.sizeof(), 16)
        self.assertEqual(Out.sizeof(), ctypes.sizeof(ROut))
        o = Out(1, In(2, 3), 4)
        self.assertEqual(Out.from_bytes(o.to_bytes()), o)
        self.assertEqual(bytes(o.to_ctypes()), o.to_bytes())

    def test_packed_outer_with_aligned_inner_keeps_inner_padding(self):
        @firstclass(align=True)
        class In:
            a: uint8_t
            b: uint32_t

        @firstclass
        class Out:
            x: uint8_t
            inner: In
        self.assertEqual(Out.sizeof(), 1 + 8)

    def test_deep_nesting_all_formats(self):
        @firstclass
        class L3:
            v: uint8_t

        @firstclass
        class L2:
            c: L3

        @firstclass
        class L1:
            b: L2

        @firstclass
        class L0:
            a: L1
        v = L0(L1(L2(L3(42))))
        self.assertEqual(v.to_bytes(), b"\x2a")
        self.assertEqual(L0.from_json(v.to_json()), v)
        self.assertEqual(L0.from_xml(v.to_xml()), v)
        self.assertEqual(L0.from_ctypes(v.to_ctypes()), v)
        self.assertEqual(json.loads(v.to_json()), {"a": {"b": {"c": {"v": 42}}}})


class DictJsonXml(unittest.TestCase):
    def test_from_dict_defaults_and_missing(self):
        self.assertEqual(Small.from_dict({"a": 1}), Small(1, 5))
        with self.assertRaises(TypeError):
            Small.from_dict({"b": 1})
        with self.assertRaises(TypeError):
            Small.from_dict([("a", 1)])
        with self.assertRaises(TypeError):
            Small.from_dict(None)

    def test_to_json_kwargs(self):
        self.assertEqual(Small(1).to_json(separators=(",", ":")), '{"a":1,"b":5}')
        self.assertEqual(Small(1).to_json(indent=1).count("\n"), 3)

    def test_from_json_garbage(self):
        for bad in ("", "{", "[1]", "null", '{"a": "x"}', '{"a": 1, "zz": 2}'):
            with self.subTest(bad):
                with self.assertRaises((ValueError, TypeError)):
                    Small.from_json(bad)

    def test_dict_key_order_is_field_order(self):
        self.assertEqual(list(Mixed(LittleInner(1), BigInner(2)).to_dict()), ["little", "big"])

    def test_xml_errors(self):
        for bad in ("", "<Small>", "<Small><a>x</a></Small>", "<Small><a>1</a><a>2</a></Small>",
                    "<Small><zz>1</zz></Small>", "<Small><a>300</a></Small>", "<Small><a></a></Small>"):
            with self.subTest(bad):
                with self.assertRaises((ValueError, TypeError)):
                    Small.from_xml(bad)

    def test_xml_missing_element_uses_default(self):
        self.assertEqual(Small.from_xml("<Small><a>3</a></Small>"), Small(3, 5))

    def test_xml_root_name_is_not_checked(self):
        self.assertEqual(Small.from_xml("<Whatever><a>3</a><b>4</b></Whatever>"), Small(3, 4))

    def test_xml_bool_text(self):
        One = single(bool_t)
        self.assertEqual(One.from_xml("<One><v>true</v></One>"), One(True))
        with self.assertRaises(ValueError):
            One.from_xml("<One><v>yes</v></One>")

    def test_xml_indent_forms(self):
        self.assertIn("\n  <a>", Small(1).to_xml(indent=2))
        self.assertEqual(Small.from_xml(Small(1).to_xml(indent=4)), Small(1))

    def test_xml_array_children_must_be_items(self):
        One = single(array_t[uint8_t, 2])
        with self.assertRaises(ValueError):
            One.from_xml("<One><v><x>1</x><x>2</x></v></One>")


class BufferHandling(unittest.TestCase):
    def test_from_bytes_accepts_buffer_types(self):
        raw = Small(1, 2).to_bytes()
        for buf in (raw, bytearray(raw), memoryview(raw), array.array("B", raw)):
            self.assertEqual(Small.from_bytes(buf), Small(1, 2))

    def test_from_bytes_length_must_match(self):
        for bad in (b"", b"\x01", b"\x01\x02\x03"):
            with self.assertRaises(ValueError):
                Small.from_bytes(bad)

    def test_from_bytes_wrong_type(self):
        for bad in ("ab", None, 5):
            with self.assertRaises(TypeError):
                Small.from_bytes(bad)

    def test_unpack_from_big_endian_and_bitfields(self):
        @firstclass(byteorder="big")
        class B:
            a: uint16_t
            bits_lo: bitfield[uint8_t, 4]
            bits_hi: bitfield[uint8_t, 4]
        raw = b"\xff" + B(0x0102, 3, 12).to_bytes() + b"\xee"
        self.assertEqual(B.unpack_from(raw, 1), B(0x0102, 3, 12))

    def test_to_bytes_returns_immutable_bytes(self):
        self.assertIsInstance(Small(1).to_bytes(), bytes)

    def test_padding_is_zero_and_ignored_on_read(self):
        @firstclass(align=True)
        class P:
            a: uint8_t
            b: uint32_t
        raw = P(1, 2).to_bytes()
        self.assertEqual(raw[1:4], b"\0\0\0")
        dirty = bytes([raw[0], 0xAA, 0xBB, 0xCC]) + raw[4:]
        self.assertEqual(P.from_bytes(dirty), P(1, 2))


class CtypesBridge(unittest.TestCase):
    def test_ctypes_type_is_cached_and_structure(self):
        t = Small.ctypes_type()
        self.assertIs(t, Small.ctypes_type())
        self.assertTrue(issubclass(t, ctypes.Structure))
        self.assertEqual(ctypes.sizeof(t), 2)

    def test_big_endian_uses_big_endian_structure(self):
        self.assertTrue(issubclass(BigInner.ctypes_type(), ctypes.BigEndianStructure))
        self.assertTrue(issubclass(LittleInner.ctypes_type(), ctypes.LittleEndianStructure))

    def test_shared_memory_style_use(self):
        buf = bytearray(b"\x07\x08")
        c = Small.ctypes_type().from_buffer(buf)
        self.assertEqual(Small.from_ctypes(c), Small(7, 8))
        c.a = 99
        self.assertEqual(buf[0], 99)

    def test_ctypes_fields_validate(self):
        with self.assertRaises(ValueError):
            Small(300).to_ctypes()

    def test_c_header_for_many_shapes(self):
        @firstclass
        class Shapes:
            e: enum_t[Level, uint8_t]
            arr: array_t[enum_t[Level, uint8_t], 2]
            m: array_t[array_t[uint16_t, 3], 2]
            s: char_array[5]
            sub: Small
            subs: array_t[Small, 2]
            f: float32_t
            d: float64_t
            flag: bool_t
        h = Shapes.c_header()
        for line in ("    uint8_t e;", "    uint8_t arr[2];", "    uint16_t m[2][3];", "    char s[5];",
                     "    struct Small sub;", "    struct Small subs[2];", "    float f;", "    double d;",
                     "    bool flag;"):
            self.assertIn(line, h)
        self.assertEqual(h.count("enum Level {"), 1)
        self.assertEqual(h.count("struct Small {"), 1)


if __name__ == "__main__":
    unittest.main()
