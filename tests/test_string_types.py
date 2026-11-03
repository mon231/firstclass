"""The four string specs: wire format, validation, variable-length rules, bare-str default."""
import ctypes
import struct
import unittest

from firstclass import (
    array_t, bitfield, firstclass, int32_t, NullTerminatedString, NullTerminatedWString, SizedString,
    SizedWString, uint32_t, uint8_t,
)


@firstclass
class NT:
    s: NullTerminatedString


@firstclass
class SZ:
    s: SizedString


@firstclass
class WSZLittle:
    s: SizedWString


@firstclass(byteorder="big")
class WNTBig:
    s: NullTerminatedWString


@firstclass(byteorder="big")
class WSZBig:
    s: SizedWString


@firstclass
class Mixed:
    a: uint32_t
    s: NullTerminatedString
    n: int32_t


@firstclass
class Strings:
    a: array_t[NullTerminatedString, 2]


@firstclass
class Defaults:
    s: str
    n: int
    f: float
    b: bool


@firstclass
class Defaulted:
    s: NullTerminatedString = "x"


class Wire(unittest.TestCase):
    def test_null_terminated_narrow(self):
        self.assertEqual(NT("hi").to_bytes(), b"hi\x00")
        self.assertEqual(NT("").to_bytes(), b"\x00")
        self.assertEqual(NT.from_bytes(b"hi\x00"), NT("hi"))
        self.assertEqual(NT.from_bytes(b"hi\x00extra-wire-is-fine"), NT("hi"))

    def test_sized_narrow(self):
        self.assertEqual(SZ("hi").to_bytes(), struct.pack("<I", 2) + b"hi")
        self.assertEqual(SZ("").to_bytes(), b"\x00\x00\x00\x00")
        self.assertEqual(SZ.from_bytes(struct.pack("<I", 2) + b"hi"), SZ("hi"))
        self.assertEqual(SZ.from_bytes(struct.pack("<I", 2) + b"hi-trailing-is-fine"), SZ("hi"))

    def test_sized_length_is_utf8_bytes(self):
        self.assertEqual(SZ("€").to_bytes(), struct.pack("<I", 3) + "€".encode())

    def test_wide_null_terminated(self):
        self.assertEqual(WNTBig("€").to_bytes(), b"\x20\xac\x00\x00")
        self.assertEqual(WNTBig.from_bytes(b"\x20\xac\x00\x00").s, "€")

    def test_wide_sized_little_endian(self):
        # "€" is one UTF-16 unit (0x20AC); little-endian unit bytes first
        self.assertEqual(WSZLittle("€").to_bytes(), struct.pack("<I", 1) + b"\xac\x20")
        self.assertEqual(WSZLittle.from_bytes(struct.pack("<I", 1) + b"\xac\x20").s, "€")

    def test_wide_sized_big_endian(self):
        self.assertEqual(WSZBig("€").to_bytes(), struct.pack(">I", 1) + b"\x20\xac")
        self.assertEqual(WSZBig.from_bytes(struct.pack(">I", 1) + b"\x20\xac").s, "€")

    def test_mixed_layout_is_sequential(self):
        m = Mixed(7, "abc", -1)
        expect = struct.pack("<I", 7) + b"abc\x00" + struct.pack("<i", -1)
        self.assertEqual(m.to_bytes(), expect)
        self.assertEqual(Mixed.from_bytes(expect), m)

    def test_string_array(self):
        s = Strings(["ab", "c"])
        expect = b"ab\x00" + b"c\x00"
        self.assertEqual(s.to_bytes(), expect)
        self.assertEqual(Strings.from_bytes(expect), s)

    def test_bare_annotations_map_to_defaults(self):
        d = Defaults("hi", 300, 1.5, True)
        self.assertEqual(d.to_bytes(), b"hi\x00" + struct.pack("<i", 300) + struct.pack("<d", 1.5) + b"\x01")
        self.assertEqual(Defaults.from_bytes(d.to_bytes()), d)
        self.assertEqual(Defaulted().s, "x")


class Validation(unittest.TestCase):
    def test_needs_str(self):
        for cls in (NT, SZ):
            with self.assertRaises(TypeError):
                cls(5)
        instance = NT("ok")
        with self.assertRaises(TypeError):
            instance.s = 7
        self.assertEqual(instance.s, "ok")

    def test_no_nul_in_any_string(self):
        for cls in (NT, SZ, WSZLittle, WNTBig):
            with self.assertRaises(ValueError):
                cls("a\0b")
        instance = SZ("ok")
        with self.assertRaises(ValueError):
            instance.s = "a\0b"
        self.assertEqual(instance.s, "ok")

    def test_from_json_xml_validate(self):
        with self.assertRaises(ValueError):
            NT.from_json('{"s": "a\\u0000b"}')
        self.assertEqual(NT.from_json(NT("hu").to_json()), NT("hu"))
        self.assertEqual(SZ.from_xml(SZ("hu").to_xml()), SZ("hu"))

    def test_bare_int_range_follows_int32(self):
        with self.assertRaises(ValueError):
            Defaults("x", 2 ** 31, 0.0, True)

    def test_wide_invalid_surrogate_rejected(self):
        with self.assertRaises(ValueError):
            WSZBig("\ud800")


class VariableLengthRules(unittest.TestCase):
    def test_sizeof_is_refused(self):
        for cls in (NT, SZ, WSZLittle):
            with self.assertRaises(TypeError):
                cls.sizeof()

    def test_ctypes_c_rust_refused(self):
        with self.assertRaises(TypeError):
            NT.ctypes_type()
        with self.assertRaises(TypeError):
            NT("x").to_ctypes()
        with self.assertRaises(TypeError):
            NT.c_header()
        with self.assertRaises(TypeError):
            NT.rust_source()

    def test_align_true_throws_at_class_creation(self):
        with self.assertRaises(TypeError):
            @firstclass(align=True)
            class Bad:
                s: NullTerminatedString

    def test_nested_string_class_throws_at_parent_creation(self):
        with self.assertRaises(TypeError):
            @firstclass
            class Child:
                s: NullTerminatedString
            @firstclass
            class Parent:
                child: Child

    def test_fixed_class_still_needs_exact_bytes(self):
        @firstclass
        class Scalar:
            n: uint8_t
        with self.assertRaises(ValueError):
            Scalar.from_bytes(b"\x01\x02")

    def test_truncated_or_unterminated_buffer(self):
        with self.assertRaises(ValueError):
            NT.from_bytes(b"hi")
        with self.assertRaises(ValueError):
            SZ.from_bytes(b"\x01\x00\x00")  # wants 1 content byte, there is none


class HeaderSide(unittest.TestCase):
    def parse(self, text):
        from firstclass import from_header
        return from_header(text=text)

    def test_std_string_maps_to_null_terminated(self):
        t = self.parse("struct S { std::string name; };")
        s = t.S("hi")
        self.assertEqual(s.to_bytes(), b"hi\x00")
        self.assertEqual(t.S.from_bytes(b"hi\x00"), s)

    def test_std_wstring_maps_to_null_terminated_w(self):
        t = self.parse("struct S { std::wstring name; };")
        s = t.S("hi")
        self.assertEqual(s.to_bytes(), b"h\x00i\x00\x00\x00")
        self.assertEqual(t.S.from_bytes(s.to_bytes()), s)

    def test_char_pointer_pairs_with_uint32_len(self):
        import dataclasses
        t = self.parse("struct S { uint32_t name_len; char *name; };")
        self.assertEqual(len(dataclasses.fields(t.S)), 1)  # name_len is consumed
        s = t.S("hi")
        self.assertEqual(s.to_bytes(), struct.pack("<I", 2) + b"hi")
        self.assertEqual(t.S.from_bytes(s.to_bytes()), s)

    def test_wchar_pointer_pairs_to_sized_w(self):
        t = self.parse("struct S { uint32_t n_len; wchar_t *n; };")
        self.assertEqual(t.S("ab").to_bytes(), struct.pack("<I", 2) + b"a\x00b\x00")

    def test_length_name_spellings(self):
        t = self.parse("struct S { uint32_t name_size; char *name; };")
        self.assertEqual(t.S("x").to_bytes(), b"\x01\x00\x00\x00x")

    def test_unpaired_char_pointer_is_skipped(self):
        t = self.parse("struct S { uint8_t a; char *note; };")
        self.assertIn("S", t.skipped)

    def test_non_string_pointer_keeps_address(self):
        t = self.parse("struct S { uint32_t v_len; int *values; };")
        s = t.S(3, 0x1234)
        raw = s.to_bytes()  # uint32 + natural padding + the address; the address is not a string field
        self.assertEqual(len(raw), 8 + ctypes_size())
        self.assertEqual(t.S.from_bytes(raw), s)

    def test_char16_pointer_pairs_too(self):
        t = self.parse("struct S { uint32_t n_len; char16_t *n; };")
        self.assertEqual(t.skipped, {})
        s = t.S("hi")
        self.assertEqual(s.to_bytes(), b"\x02\x00\x00\x00" + "hi".encode("utf-16-le"))
        self.assertEqual(t.S.from_bytes(s.to_bytes()), s)

    def test_length_array_does_not_pair(self):
        t = self.parse("struct S { uint32_t n_len[2]; char *n; };")
        self.assertIn("S", t.skipped)


def ctypes_size():
    import ctypes
    return ctypes.sizeof(ctypes.c_void_p)


class VariableExtras(unittest.TestCase):
    def test_bitfields_in_variable_class_match_the_fixed_layout(self):
        @firstclass
        class FixedLittle:
            a: bitfield[uint8_t, 3]
            b: bitfield[uint8_t, 3]
            c: bitfield[uint8_t, 3]  # 3+3+3 > 8: capacity starts a second unit

        @firstclass
        class FlexLittle:
            a: bitfield[uint8_t, 3]
            b: bitfield[uint8_t, 3]
            c: bitfield[uint8_t, 3]
            s: NullTerminatedString

        @firstclass(byteorder="big")
        class FixedBig:
            a: bitfield[uint8_t, 3]
            b: bitfield[uint8_t, 3]
            c: bitfield[uint8_t, 3]

        @firstclass(byteorder="big")
        class FlexBig:
            a: bitfield[uint8_t, 3]
            b: bitfield[uint8_t, 3]
            c: bitfield[uint8_t, 3]
            s: NullTerminatedString

        flex = FlexLittle(7, 7, 3, "z")
        self.assertEqual(flex.to_bytes(), FixedLittle(7, 7, 3).to_bytes() + b"z\x00")
        self.assertEqual(FlexLittle.from_bytes(flex.to_bytes()), flex)
        big = FlexBig(7, 7, 3, "z")
        self.assertEqual(big.to_bytes(), FixedBig(7, 7, 3).to_bytes() + b"z\x00")
        self.assertEqual(FlexBig.from_bytes(big.to_bytes()), big)

    def test_unpack_from_reads_one_object_at_the_offset(self):
        @firstclass
        class V:
            n: uint8_t
            s: NullTerminatedString
        self.assertEqual(V.unpack_from(b"\xff\x01z\x00", 1), V(1, "z"))

    def test_from_ctypes_is_refused(self):
        with self.assertRaises(TypeError):
            NT.from_ctypes(ctypes.c_void_p())

    def test_forward_reference_resolves_later(self):
        mod = self._module()
        exec(compile(
            "from firstclass import firstclass, int32_t, uint8_t\n"
            "@firstclass\n"
            "class A:\n"
            "    n: uint8_t\n"
            "    b: 'B'\n"  # created before B exists: deferred, checked when hints resolve
            "@firstclass\n"
            "class B:\n"
            "    x: int32_t\n",
            "<forward-ref>", "exec"), mod.__dict__)
        a = mod.A(1, mod.B(-2))
        self.assertEqual(a.to_bytes(), b"\x01" + b"\xfe\xff\xff\xff")
        self.assertEqual(mod.A.from_bytes(a.to_bytes()), a)

    def test_deferred_resolution_still_throws_at_first_use(self):
        mod = self._module()
        exec(compile(
            "from firstclass import firstclass\n"
            "@firstclass\n"
            "class A:\n"
            "    p: 'Missing'\n",
            "<forward-ref>", "exec"), mod.__dict__)
        A = mod.A
        with self.assertRaises(NameError):
            A(1)  # construction resolves the hints: deferred checks
        mod.Missing = object
        with self.assertRaises(TypeError):
            A(1)  # hints resolve now: the creation checks run, no silent field

    def _module(self):
        import sys, types
        name = "fwd_varstrings_" + self.id()
        mod = types.ModuleType(name)
        mod.__dict__["__name__"] = name
        sys.modules[name] = mod
        self.addCleanup(sys.modules.pop, name, None)
        return mod


if __name__ == "__main__":
    unittest.main()
