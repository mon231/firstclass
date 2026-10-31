import ctypes
import enum
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import firstclass
from firstclass import (
    array_t, bitfield, bytes_t, char_array, enum_t, firstclass as fc, float32_t, float64_t, from_header, from_rust,
    int16_t, int32_t, uint8_t, uint16_t, uint32_t, uint64_t, wchar_array,
)

PTR = ctypes.sizeof(ctypes.c_void_p)


class PackedAndAligned(unittest.TestCase):
    SRC = """
    #[repr(C, packed)]
    #[derive(Clone, Copy)]
    pub struct Header {
        pub magic: u32,
        pub version: u8,
        pub length: u16,
        pub stamp: u64,
        pub ratio: f32,
        pub precise: f64,
        pub ok: bool,
        pub delta: i16,
        pub signed: i8,
        pub big: i64,
    }

    #[repr(C)]
    struct Padded { a: u8, b: u32, c: u8, d: u64, e: u16 }

    #[repr(C, packed(1))]
    struct PackedOne { a: u8, b: u32 }
    """

    def setUp(self):
        self.t = from_rust(text=self.SRC)

    def test_all_parsed(self):
        self.assertEqual(self.t.skipped, {})
        self.assertEqual(sorted(self.t.structs), ["Header", "PackedOne", "Padded"])

    def test_packed_layout_matches_struct_module(self):
        h = self.t.Header(0xCAFE, 3, 513, 2 ** 40, 1.5, 2.25, True, -7, -1, -2 ** 40)
        self.assertEqual(h.to_bytes(), struct.pack("<IBHQfd?hbq", 0xCAFE, 3, 513, 2 ** 40, 1.5, 2.25, True, -7, -1, -2 ** 40))
        self.assertEqual(self.t.Header.from_bytes(h.to_bytes()), h)

    def test_repr_c_has_c_padding(self):
        class Ref(ctypes.Structure):
            _fields_ = [("a", ctypes.c_uint8), ("b", ctypes.c_uint32), ("c", ctypes.c_uint8),
                        ("d", ctypes.c_uint64), ("e", ctypes.c_uint16)]
        self.assertEqual(self.t.Padded.sizeof(), ctypes.sizeof(Ref))
        self.assertEqual(self.t.Padded.sizeof(), 32)

    def test_packed_one(self):
        self.assertEqual(self.t.PackedOne.sizeof(), 5)

    def test_options(self):
        self.assertEqual(from_rust(text=self.SRC, align=False).Padded.sizeof(), 16)
        self.assertEqual(from_rust(text=self.SRC, align=True).PackedOne.sizeof(), 8)
        be = from_rust(text=self.SRC, byteorder="big")
        self.assertEqual(be.PackedOne(1, 2).to_bytes(), struct.pack(">BI", 1, 2))

    def test_generated_classes_are_full_firstclass(self):
        t = self.t
        h = t.Header(1, 2, 3, 4, 0.5, 0.25, False, 5, 6, 7)
        self.assertEqual(t.Header.from_json(h.to_json()), h)
        self.assertEqual(t.Header.from_xml(h.to_xml()), h)
        self.assertEqual(t.Header.from_ctypes(h.to_ctypes()), h)
        self.assertEqual(t.Header.unpack_from(b"\0\0" + h.to_bytes(), 2), h)
        self.assertTrue(t.PackedOne(1, 2) < t.PackedOne(1, 3))


class TypesAndArrays(unittest.TestCase):
    def test_array_forms(self):
        t = from_rust(text="""
            const N: usize = 4;
            const WIDE: usize = N * 2;
            type Mac = [u8; 6];
            #[repr(C, packed)]
            struct S {
                raw: [u8; N],
                name: [c_char; WIDE],
                mac: Mac,
                words: [u16; 3],
                grid: [[i16; 3]; 2],
                signed: [i8; 2],
                chars: [[c_char; 4]; 2],
                calc: [u8; (N + 1) * 2 - 1],
                cast: [u8; 3 as usize],
            }
        """)
        self.assertEqual(t.skipped, {})
        self.assertEqual(t.S.sizeof(), 4 + 8 + 6 + 6 + 12 + 2 + 8 + 9 + 3)
        s = t.S(b"\x01\x02\x03\x04", "name", b"\x00\x11\x22\x33\x44\x55", [1, 2, 3], [[1, 2, 3], [4, 5, 6]],
                [-1, 1], ["ab", "cd"], b"x" * 9, b"abc")
        self.assertEqual(t.S.from_bytes(s.to_bytes()), s)

    def test_c_ffi_type_names(self):
        t = from_rust(text="""
            use std::os::raw::*;
            #[repr(C, packed)]
            struct S {
                a: c_char, b: c_schar, c: c_uchar, d: c_short, e: c_ushort, f: c_int, g: c_uint,
                h: c_longlong, i: c_ulonglong, j: c_float, k: c_double,
                l: libc::c_int, m: core::ffi::c_uint, n: std::os::raw::c_short,
            }
        """)
        self.assertEqual(t.skipped, {})
        self.assertEqual(t.S.sizeof(), 1 + 1 + 1 + 2 + 2 + 4 + 4 + 8 + 8 + 4 + 8 + 4 + 4 + 2)

    def test_usize_isize_and_long_follow_the_platform(self):
        t = from_rust(text="#[repr(C, packed)] struct S { a: usize, b: isize, c: c_long, d: c_ulong }")
        self.assertEqual(t.S.sizeof(), 2 * PTR + 2 * ctypes.sizeof(ctypes.c_long))
        t4 = from_rust(text="#[repr(C, packed)] struct S { c: c_long, d: c_ulong }", long_size=4)
        self.assertEqual(t4.S.sizeof(), 8)

    def test_pointers_references_and_fn_pointers(self):
        t = from_rust(text="""
            #[repr(C)]
            struct Node<'a>;
            #[repr(C)]
            struct S {
                a: *const u8,
                b: *mut S,
                c: &'static u8,
                d: &mut u16,
                e: Option<&u8>,
                f: Option<Box<S>>,
                g: Box<u8>,
                h: extern "C" fn(i32) -> i32,
                i: unsafe extern "C" fn(*mut u8),
                j: fn(),
                k: Option<extern "C" fn()>,
                l: std::ptr::NonNull<u8>,
                m: [*mut u8; 2],
            }
        """)
        self.assertEqual(t.skipped.get("S"), None, t.skipped)
        self.assertEqual(t.S.sizeof(), 13 * PTR + PTR)  # twelve pointers + the array of two = 14 slots
        s = t.S(*([0x10] * 12), [1, 2])
        self.assertEqual(t.S.from_bytes(s.to_bytes()), s)

    def test_nested_structs_and_arrays_of_structs(self):
        t = from_rust(text="""
            #[repr(C, packed)] struct Point { x: i16, y: i16 }
            #[repr(C, packed)] struct Line { a: Point, b: self::Point, pts: [crate::Point; 3] }
        """)
        self.assertEqual(t.skipped, {})
        self.assertEqual(t.Line.sizeof(), 4 * 5)
        line = t.Line(t.Point(1, 2), t.Point(3, 4), [t.Point(5, 6), t.Point(7, 8), t.Point(9, 10)])
        self.assertEqual(t.Line.from_bytes(line.to_bytes()), line)

    def test_struct_used_before_definition(self):
        t = from_rust(text="#[repr(C)] struct A { b: B } #[repr(C)] struct B { x: u16 }")
        self.assertEqual(t.A.sizeof(), 2)

    def test_type_alias_chain(self):
        t = from_rust(text="""
            type Byte = u8;
            type Id = u32;
            type Ids = [Id; 2];
            type Ptr = *mut Byte;
            #[repr(C, packed)] struct S { b: Byte, ids: Ids, p: Ptr }
        """)
        self.assertEqual(t.S.sizeof(), 1 + 8 + PTR)

    def test_wchar(self):
        t = from_rust(text="""
            #[repr(C, packed)]
            struct S { w: wchar_t, name: [wchar_t; 4], lib: [libc::wchar_t; 2] }
        """)
        self.assertEqual(t.S.sizeof(), 2 + 8 + 4)
        s = t.S("a", "hé", "xy")
        self.assertEqual(t.S.from_bytes(s.to_bytes()), s)

    def test_tuple_structs(self):
        t = from_rust(text="#[repr(C, packed)] struct Pair(u8, u16); #[repr(C)] struct Wrap(Pair);")
        self.assertEqual(list(t.Pair.__dataclass_fields__), ["f0", "f1"])
        self.assertEqual(t.Pair(1, 2).to_bytes(), b"\x01\x02\x00")
        self.assertEqual(t.Wrap.sizeof(), 3)

    def test_keyword_field_names(self):
        t = from_rust(text="#[repr(C, packed)] struct S { r#type: u8, r#match: u8, from: u8, class: u8 }")
        self.assertEqual(list(t.S.__dataclass_fields__), ["type", "match", "from_", "class_"])


class Enums(unittest.TestCase):
    SRC = """
    const BASE: u8 = 10;
    #[repr(u8)] #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum Status { Ok = 0, Busy, Failed = 0x80 }
    #[repr(i16)] enum Signed { Neg = -2, Pos = 5, Next }
    #[repr(u32)] enum Big { A = 1 << 31, B = 0xFFFF_FFFF }
    #[repr(C)] enum CLike { X, Y = BASE as isize, Z }
    #[repr(u16)] enum Expr { A = 1 + 2, B = (3 << 2) | 1, C = BASE as u16 * 2 }
    #[repr(u8)] enum Attrs {
        #[allow(dead_code)] First = 1,
        /// documented
        Second,
    }
    #[repr(C, packed)]
    struct Uses { s: Status, g: Signed, b: Big, c: CLike, tail: u8 }
    """

    def setUp(self):
        self.t = from_rust(text=self.SRC)

    def values(self, enum_cls):
        return {m.name: m.value for m in enum_cls}

    def test_no_skips(self):
        self.assertEqual(self.t.skipped, {})

    def test_values(self):
        t = self.t
        self.assertEqual(self.values(t.Status), {"Ok": 0, "Busy": 1, "Failed": 128})
        self.assertEqual(self.values(t.Signed), {"Neg": -2, "Pos": 5, "Next": 6})
        self.assertEqual(self.values(t.Big), {"A": 2 ** 31, "B": 2 ** 32 - 1})
        self.assertEqual(self.values(t.CLike), {"X": 0, "Y": 10, "Z": 11})
        self.assertEqual(self.values(t.Expr), {"A": 3, "B": 13, "C": 20})
        self.assertEqual(self.values(t.Attrs), {"First": 1, "Second": 2})
        self.assertTrue(issubclass(t.Status, enum.IntEnum))

    def test_repr_picks_the_storage_size(self):
        self.assertEqual(self.t.Uses.sizeof(), 1 + 2 + 4 + 4 + 1)

    def test_roundtrip_and_bytes(self):
        t = self.t
        u = t.Uses(t.Status.Failed, t.Signed.Neg, t.Big.B, t.CLike.Z, 9)
        self.assertEqual(u.to_bytes(), struct.pack("<BhIIB", 0x80, -2, 2 ** 32 - 1, 11, 9))
        back = t.Uses.from_bytes(u.to_bytes())
        self.assertEqual(back, u)
        self.assertIs(back.s, t.Status.Failed)
        self.assertEqual(t.Uses.from_json(u.to_json()), u)

    def test_unknown_discriminant_is_rejected(self):
        bad = bytearray(struct.pack("<BhIIB", 0, 5, 1, 0, 0))
        bad[0] = 77
        with self.assertRaises(ValueError):
            self.t.Uses.from_bytes(bytes(bad))

    def test_constants_are_qualified(self):
        c = self.t.constants
        self.assertEqual((c["BASE"], c["Status::Busy"], c["CLike::Z"]), (10, 1, 11))

    def test_value_must_fit_repr(self):
        t = from_rust(text="#[repr(u8)] enum E { A = 300 } #[repr(i8)] enum F { A = 200 }")
        self.assertEqual(sorted(t.skipped), ["E", "F"])
        self.assertIn("out of range", t.skipped["E"])


class Skipping(unittest.TestCase):
    def test_reasons(self):
        t = from_rust(text="""
            struct NoRepr { a: u8 }
            #[repr(Rust)] struct Explicit { a: u8 }
            #[repr(packed)] struct PackedOnly { a: u8 }
            #[repr(transparent)] struct Transparent(u8);
            #[repr(C, align(16))] struct Aligned { a: u8 }
            #[repr(C, packed(2))] struct Two { a: u8, b: u32 }
            #[repr(C)] struct Generic<T> { a: T }
            #[repr(C)] struct Lifetime<'a> { a: &'a u8 }
            #[repr(C)] union U { a: u8, b: u16 }
            #[repr(C)] struct HasChar { c: char }
            #[repr(C)] struct Wide { a: u128 }
            #[repr(C)] struct HasString { s: String }
            #[repr(C)] struct HasVec { v: Vec<u8> }
            #[repr(C)] struct HasSlice { s: [u8] }
            #[repr(C)] struct HasTuple { t: (u8, u8) }
            #[repr(C)] struct Unknown { m: Missing }
            #[repr(C)] struct Dep { u: Unknown }
            #[repr(C)] struct BadLen { a: [u8; UNKNOWN] }
            #[repr(C)] struct Empty {}
            #[repr(C)] struct Unit;
            #[repr(C)] struct Generics2 { a: Vec<u8>, b: u8 }
            #[repr(C)] struct Fine { a: u8 }
        """)
        want = {
            "NoRepr": "layout unspecified", "Explicit": "layout unspecified", "PackedOnly": "layout unspecified",
            "Transparent": "transparent", "Aligned": "align(16)", "Two": "packed(2)", "Generic": "generic",
            "Lifetime": "generic", "U": "union", "HasChar": "char", "Wide": "128-bit", "HasString": "String",
            "HasVec": "Vec", "HasSlice": "slices", "HasTuple": "tuples", "Unknown": "unknown type Missing",
            "Dep": "depends on skipped", "BadLen": "array length", "Empty": "empty", "Unit": "empty",
            "Generics2": "Vec",
        }
        for name, word in want.items():
            self.assertIn(name, t.skipped, name)
            self.assertIn(word, t.skipped[name], name)
        self.assertEqual(list(t.structs), ["Fine"])

    def test_assume_repr_c(self):
        t = from_rust(text="struct A { a: u8, b: u32 } enum E { X, Y } #[repr(packed)] struct P { a: u8, b: u32 }",
                      assume_repr_c=True)
        self.assertEqual(t.skipped, {})
        self.assertEqual((t.A.sizeof(), t.P.sizeof()), (8, 5))

    def test_enums_with_data_and_generics(self):
        t = from_rust(text="""
            #[repr(u8)] enum Data { A(u8), B }
            #[repr(u8)] enum Struct { A { x: u8 }, B }
            #[repr(u8)] enum Generic<T> { A }
            #[repr(C)] enum Ok { A, B }
        """)
        self.assertIn("data", t.skipped["Data"])
        self.assertIn("data", t.skipped["Struct"])
        self.assertIn("generic", t.skipped["Generic"])
        self.assertEqual(list(t.enums), ["Ok"])

    def test_one_bad_field_skips_only_that_struct(self):
        t = from_rust(text="#[repr(C)] struct Bad { a: u8, s: String } #[repr(C)] struct Good { a: u8 }")
        self.assertEqual(list(t.structs), ["Good"])
        self.assertIn("Bad.s", t.skipped["Bad"])


class SourceShapes(unittest.TestCase):
    def test_comments_attributes_visibility_and_noise(self):
        t = from_rust(text="""
            //! crate docs
            #![allow(dead_code)]
            use std::collections::HashMap;
            /* block /* nested */ comment with struct Hidden { a: u8 } */
            // struct AlsoHidden { a: u8 }
            /// doc comment
            #[derive(Debug, Clone, Copy, PartialEq)]
            #[repr(C)]
            pub(crate) struct Real {
                /// field doc
                #[allow(unused)]
                pub a: u8,
                pub(crate) b: u16, // trailing
                c: [u8; 2],
            }
            impl Real {
                pub const SIZE: usize = 5;
                pub fn new() -> Self { Real { a: 0, b: 0, c: [0; 2] } }
            }
            impl Default for Real { fn default() -> Self { Self::new() } }
            fn helper(x: u8) -> u8 { x + 1 }
            static GLOBAL: u8 = 3;
            extern "C" { fn external(x: i32) -> i32; }
            macro_rules! make { ($x:expr) => { struct Fake { a: u8 } }; }
            trait T { fn f(&self); }
            let_not_valid_but_ignored!();
            const fn const_helper() -> usize { 3 }
        """)
        self.assertEqual(list(t.structs), ["Real"])
        self.assertEqual(t.Real.sizeof(), 6)  # C layout: u8, pad, u16, [u8;2]

    def test_strings_and_chars_do_not_confuse_the_tokenizer(self):
        t = from_rust(text="""
            const MSG: &str = "struct Hidden { a: u8 } // not a comment /* nor this */";
            const QUOTE: char = '"';
            const LIFETIME_FN: u8 = 1;
            #[repr(C)] struct S<'a, 'b> { r: &'a u8 }
            #[repr(C)] struct After { a: u8 }
            const BYTE: u8 = b'A';
        """)
        self.assertEqual(sorted(t.structs), ["After"])
        self.assertEqual(t.constants["BYTE"], 65)
        self.assertEqual(t.constants["LIFETIME_FN"], 1)

    def test_modules_are_flattened(self):
        t = from_rust(text="""
            pub mod net {
                pub const LEN: usize = 3;
                #[repr(C, packed)] pub struct Pkt { pub data: [u8; LEN] }
                pub mod inner { #[repr(C)] pub struct Deep { pub a: u8 } }
            }
            #[cfg(test)]
            mod tests { #[repr(C)] struct InTests { a: u8 } }
            #[cfg(test)]
            #[repr(C)] struct AlsoTestOnly { a: u8 }
            #[repr(C)] struct Uses { p: net::Pkt, d: net::inner::Deep }
        """)
        self.assertEqual(sorted(t.structs), ["Deep", "Pkt", "Uses"])
        self.assertEqual(t.Uses.sizeof(), 4)

    def test_number_literal_forms(self):
        t = from_rust(text="""
            const A: usize = 0x10;
            const B: usize = 0b101;
            const C: usize = 0o17;
            const D: usize = 1_000;
            const E: usize = 10usize;
            const F: usize = 0xFFu8 as usize;
            const G: u32 = 1u32 << 4;
            const H: usize = 010;
            const I: usize = A + B * 2;
        """)
        c = t.constants
        self.assertEqual((c["A"], c["B"], c["C"], c["D"], c["E"], c["F"], c["G"], c["H"], c["I"]),
                         (16, 5, 15, 1000, 10, 255, 16, 10, 26))

    def test_from_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "lib.rs"
            path.write_text("#[repr(C)] pub struct P { a: u8, b: u16 }", encoding="utf-8")
            self.assertEqual(from_rust(path).P.sizeof(), 4)
            self.assertEqual(from_rust(str(path)).P.sizeof(), 4)

    def test_args_validated(self):
        with self.assertRaises(TypeError):
            from_rust()
        with self.assertRaises(TypeError):
            from_rust("a.rs", text="")
        with self.assertRaises(ValueError):
            from_rust(text="", byteorder="middle")
        with self.assertRaises(FileNotFoundError):
            from_rust("definitely/missing.rs")
        self.assertEqual(dict(from_rust(text="")), {})

    def test_both_spellings(self):
        self.assertIs(firstclass.from_rust, from_rust)
        self.assertIs(firstclass.firstclass.from_rust, from_rust)

    def test_garbage_does_not_hang_or_crash(self):
        t = from_rust(text="}}} ;;; ( ) [ ] # ! ; struct enum < > :: -> ; pub pub ; "
                           "#[repr(C)] struct Ok { a: u8 }")
        self.assertIn("Ok", t.structs)


class RustSourceOutput(unittest.TestCase):
    def make(self):
        class Color(enum.IntEnum):
            RED = 1
            GREEN = 2

        @fc
        class Inner:
            x: int16_t

        @fc
        class Msg:
            color: enum_t[Color, uint8_t]
            name: char_array[8]
            wname: wchar_array[4]
            raw: bytes_t[4]
            grid: array_t[array_t[uint16_t, 3], 2]
            inner: Inner
            many: array_t[Inner, 2]
            f: float32_t
            d: float64_t
            big: uint64_t
            p: ctypes.c_void_p
            a: bitfield[uint8_t, 3]
            b: bitfield[uint8_t, 5]
            type: uint8_t
        return Msg

    def test_shapes(self):
        src = self.make().rust_source()
        for line in ("use std::os::raw::c_void;", "#[repr(C, packed)]", "#[derive(Clone, Copy)]", "pub struct Msg {",
                     "pub struct Inner {", "    pub x: i16,", "    pub color: Color,", "    pub name: [u8; 8],",
                     "    pub wname: [u16; 4],", "    pub raw: [u8; 4],", "    pub grid: [[u16; 3]; 2],",
                     "    pub inner: Inner,", "    pub many: [Inner; 2],", "    pub f: f32,", "    pub d: f64,",
                     "    pub big: u64,", "    pub p: *mut c_void,", "    pub bits_0: u8, // a: 3 bits, b: 5 bits",
                     "    pub r#type: u8,", "#[repr(u8)]", "pub enum Color {", "    RED = 1,", "    GREEN = 2,"):
            self.assertIn(line, src)
        self.assertEqual(src.count("pub struct Inner"), 1)
        self.assertLess(src.index("pub struct Inner"), src.index("pub struct Msg"))
        self.assertLess(src.index("pub enum Color"), src.index("pub struct Msg"))

    def test_aligned_is_plain_repr_c(self):
        @fc(align=True)
        class A:
            a: uint8_t
            b: uint32_t
        src = A.rust_source()
        self.assertIn("#[repr(C)]\n", src)
        self.assertNotIn("packed", src)
        self.assertNotIn("c_void", src)

    def test_enum_aliases_become_comments_and_signed_repr(self):
        class E(enum.IntEnum):
            A = -1
            B = 1
            C = 1  # alias of B
        @fc
        class S:
            e: enum_t[E, int32_t]
        src = S.rust_source()
        self.assertIn("#[repr(i32)]", src)
        self.assertIn("    A = -1,", src)
        self.assertEqual(src.count("= 1,"), 1)

    def test_rust_keywords_are_escaped(self):
        @fc
        class K:
            match: uint8_t
            self: uint8_t
        src = K.rust_source()
        self.assertIn("pub r#match: u8,", src)
        self.assertIn("pub self_: u8,", src)

    def test_roundtrip_through_from_rust(self):
        Msg = self.make()
        again = from_rust(text=Msg.rust_source())
        self.assertEqual(again.skipped, {})
        self.assertEqual(again.Msg.sizeof(), Msg.sizeof())
        self.assertEqual(again.Inner.sizeof(), 2)
        self.assertEqual({m.name: m.value for m in again.Color}, {"RED": 1, "GREEN": 2})

    def test_roundtrip_aligned_and_from_header_source(self):
        t = from_header(text="""
            enum class M : uint16_t { A = 1, B };
            struct P { uint8_t a; uint32_t b; };
            struct S { P p; M m; uint64_t big; wchar_t w[3]; double d[2]; };
        """)
        again = from_rust(text=t.S.rust_source())
        self.assertEqual(again.skipped, {})
        for name, cls in again.structs.items():
            self.assertEqual(cls.sizeof(), t.structs[name].sizeof(), name)

    def test_c_and_rust_agree_on_sizes(self):
        Msg = self.make()
        self.assertEqual(from_header(text=Msg.c_header()).Msg.sizeof(), from_rust(text=Msg.rust_source()).Msg.sizeof())


if __name__ == "__main__":
    unittest.main()
