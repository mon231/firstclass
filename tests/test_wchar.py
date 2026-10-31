import ctypes
import struct
import unittest

from firstclass import array_t, firstclass, from_header, uint8_t, wchar_array, wchar_t


def single(tp, **kw):
    @firstclass(**kw)
    class One:
        v: tp
    return One


class Basics(unittest.TestCase):
    def test_two_bytes_per_unit(self):
        One = single(wchar_array[3])
        self.assertEqual(One.sizeof(), 6)
        self.assertEqual(One("abc").to_bytes(), "abc".encode("utf-16-le"))
        self.assertEqual(One.from_bytes("abc".encode("utf-16-le")).v, "abc")

    def test_nul_padded_to_full_width(self):
        One = single(wchar_array[4])
        self.assertEqual(One("ab").to_bytes(), b"a\0b\0\0\0\0\0")
        self.assertEqual(One.from_bytes(b"a\0b\0\0\0\0\0").v, "ab")
        self.assertEqual(One("").to_bytes(), bytes(8))
        self.assertEqual(One.from_bytes(bytes(8)).v, "")

    def test_exact_fit_has_no_terminator(self):
        One = single(wchar_array[2])
        self.assertEqual(One.from_bytes(One("xy").to_bytes()).v, "xy")

    def test_stops_at_first_nul_unit(self):
        One = single(wchar_array[4])
        self.assertEqual(One.from_bytes("ab".encode("utf-16-le") + bytes(2) + "c".encode("utf-16-le")).v, "ab")

    def test_non_ascii_bmp(self):
        One = single(wchar_array[4])
        text = "hé€中"
        self.assertEqual(One.from_bytes(One(text).to_bytes()).v, text)
        self.assertEqual(One(text).to_bytes(), text.encode("utf-16-le"))

    def test_surrogate_pairs_count_as_two_units(self):
        One = single(wchar_array[4])
        smile = "\U0001F600"
        self.assertEqual(len(smile.encode("utf-16-le")), 4)
        self.assertEqual(One.from_bytes(One(smile + "ab").to_bytes()).v, smile + "ab")
        with self.assertRaises(ValueError):
            One(smile + smile + "a").to_bytes()  # 5 units
        self.assertEqual(One.from_bytes(One(smile + smile).to_bytes()).v, smile + smile)

    def test_too_long_counts_units_not_chars(self):
        One = single(wchar_array[3])
        with self.assertRaisesRegex(ValueError, "4 UTF-16 units"):
            One("abcd").to_bytes()

    def test_bad_values(self):
        One = single(wchar_array[4])
        with self.assertRaises(ValueError):
            One("a\0b").to_bytes()
        with self.assertRaises(ValueError):
            One("\ud800").to_bytes()  # lone surrogate cannot be encoded
        for bad in (b"ab", 5, None, ["a"]):
            with self.assertRaises(TypeError):
                One(bad).to_bytes()

    def test_invalid_utf16_on_read(self):
        One = single(wchar_array[2])
        with self.assertRaises(ValueError):
            One.from_bytes(struct.pack("<2H", 0xD800, 0x0041))  # high surrogate not followed by a low one

    def test_bad_sizes(self):
        for bad in (0, -1, "3", None, True):
            with self.assertRaises(TypeError):
                wchar_array[bad]


class WideChar(unittest.TestCase):
    def test_single_unit(self):
        One = single(wchar_t)
        self.assertEqual(One.sizeof(), 2)
        self.assertEqual(One("€").to_bytes(), b"\xac\x20")
        self.assertEqual(One.from_bytes(b"\xac\x20").v, "€")

    def test_single_unit_rejects_longer_values(self):
        One = single(wchar_t)
        for bad in ("ab", "\U0001F600"):
            with self.assertRaises(ValueError):
                One(bad).to_bytes()

    def test_empty_single_unit_is_nul(self):
        One = single(wchar_t)
        self.assertEqual(One("").to_bytes(), bytes(2))


class ByteOrder(unittest.TestCase):
    def test_big_endian_units(self):
        One = single(wchar_array[2], byteorder="big")
        self.assertEqual(One("ab").to_bytes(), "ab".encode("utf-16-be"))
        self.assertEqual(One.from_bytes("ab".encode("utf-16-be")).v, "ab")

    def test_big_endian_surrogates(self):
        One = single(wchar_array[2], byteorder="big")
        smile = "\U0001F600"
        self.assertEqual(One(smile).to_bytes(), smile.encode("utf-16-be"))
        self.assertEqual(One.from_bytes(smile.encode("utf-16-be")).v, smile)

    def test_big_endian_differs_from_little(self):
        little, big = single(wchar_array[1]), single(wchar_array[1], byteorder="big")
        self.assertEqual(little("Ă").to_bytes(), b"\x02\x01")
        self.assertEqual(big("Ă").to_bytes(), b"\x01\x02")


class Layout(unittest.TestCase):
    def test_alignment_is_two(self):
        @firstclass(align=True)
        class A:
            a: uint8_t
            w: wchar_array[2]
            b: uint8_t
        self.assertEqual(A.sizeof(), 1 + 1 + 4 + 1 + 1)  # pad before w, tail pad to 2

        class Ref(ctypes.Structure):
            _fields_ = [("a", ctypes.c_uint8), ("w", ctypes.c_uint16 * 2), ("b", ctypes.c_uint8)]
        self.assertEqual(A.sizeof(), ctypes.sizeof(Ref))
        a = A(1, "xy", 2)
        self.assertEqual(A.from_bytes(a.to_bytes()), a)

    def test_in_arrays_and_nested(self):
        @firstclass
        class Inner:
            name: wchar_array[3]

        @firstclass
        class Outer:
            names: array_t[wchar_array[3], 2]
            inner: Inner
            items: array_t[Inner, 2]
        o = Outer(["ab", "c"], Inner("xyz"), [Inner("1"), Inner("22")])
        self.assertEqual(Outer.sizeof(), 6 * 2 + 6 + 12)
        self.assertEqual(Outer.from_bytes(o.to_bytes()), o)


class Formats(unittest.TestCase):
    def setUp(self):
        self.One = single(wchar_array[4])
        self.v = self.One("hé\U0001F600")

    def test_json_xml(self):
        self.assertEqual(self.One.from_json(self.v.to_json()), self.v)
        self.assertEqual(self.One.from_xml(self.v.to_xml()), self.v)
        self.assertEqual(self.One.from_dict(self.v.to_dict()), self.v)

    def test_json_validates(self):
        with self.assertRaises(ValueError):
            self.One.from_dict({"v": "too long!"})
        with self.assertRaises(TypeError):
            self.One.from_dict({"v": 5})

    def test_ctypes_matches_bytes_little_and_big(self):
        for kw in ({}, {"byteorder": "big"}):
            with self.subTest(kw):
                One = single(wchar_array[4], **kw)
                v = One("hé\U0001F600")
                c = v.to_ctypes()
                self.assertEqual(bytes(c), v.to_bytes())
                self.assertEqual(One.from_ctypes(c), v)

    def test_ctypes_array_of_wchar_arrays(self):
        @firstclass
        class Names:
            n: array_t[wchar_array[3], 2]
        v = Names(["ab", "€"])
        self.assertEqual(Names.from_ctypes(v.to_ctypes()), v)
        self.assertEqual(bytes(v.to_ctypes()), v.to_bytes())

    def test_c_header_uses_portable_uint16(self):
        self.assertIn("uint16_t v[4];", self.One.c_header())

    def test_rust_source_uses_u16(self):
        self.assertIn("pub v: [u16; 4],", self.One.rust_source())

    def test_unpack_from_and_compare(self):
        raw = b"\xff" + self.v.to_bytes()
        self.assertEqual(self.One.unpack_from(raw, 1), self.v)
        self.assertTrue(self.One("a") < self.One("b"))


class FromHeader(unittest.TestCase):
    def test_wchar_t_and_char16_t(self):
        t = from_header(text="""
            struct S {
                wchar_t name[4];
                char16_t title[2];
                wchar_t single;
                wchar_t grid[2][3];
                wchar_t *ptr;
            };
        """, align=False)
        self.assertEqual(t.skipped, {})
        self.assertEqual(t.S.sizeof(), 8 + 4 + 2 + 12 + ctypes.sizeof(ctypes.c_void_p))
        s = t.S("ab", "xy", "z", ["abc", "de"], 0)
        self.assertEqual(t.S.from_bytes(s.to_bytes()), s)
        self.assertEqual(s.to_bytes()[:4], "ab".encode("utf-16-le"))

    def test_wchar_alignment_in_c_layout(self):
        t = from_header(text="struct S { uint8_t a; wchar_t w; uint8_t b; };")
        self.assertEqual(t.S.sizeof(), 6)

    def test_wchar_in_typedef_and_macro_size(self):
        t = from_header(text="""
            #define LEN 3
            typedef wchar_t wname_t[LEN];
            struct S { wname_t a; };
        """, align=False)
        self.assertEqual(t.S.sizeof(), 6)


if __name__ == "__main__":
    unittest.main()
