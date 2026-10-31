import ctypes
import enum
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import firstclass
from firstclass import from_header


class PackedBasics(unittest.TestCase):
    H = """
    #include <stdint.h>
    // line comment
    /* block
       comment */
    #pragma pack(push, 1)
    typedef struct Header {
        uint8_t  kind;      /* trailing */
        uint16_t length;
        int32_t  value;
        uint64_t stamp;
        float    ratio;
        double   precise;
        bool     ok;
        char     tag;
    } Header_t;
    #pragma pack(pop)
    """

    def test_struct_and_alias(self):
        t = from_header(text=self.H)
        self.assertIs(t.Header, t.Header_t)
        self.assertEqual(t.Header.sizeof(), 1 + 2 + 4 + 8 + 4 + 8 + 1 + 1)
        self.assertEqual(t.skipped, {})

    def test_bytes_match_struct_module(self):
        t = from_header(text=self.H)
        h = t.Header(7, 513, -5, 2 ** 40, 1.5, 2.25, True, "z")
        self.assertEqual(h.to_bytes(), struct.pack("<BHiQfd?c", 7, 513, -5, 2 ** 40, 1.5, 2.25, True, b"z"))
        self.assertEqual(t.Header.from_bytes(h.to_bytes()), h)

    def test_big_endian_option(self):
        t = from_header(text=self.H, byteorder="big")
        h = t.Header(1, 2, 3, 4, 0.5, 0.25, False, "q")
        self.assertEqual(h.to_bytes(), struct.pack(">BHiQfd?c", 1, 2, 3, 4, 0.5, 0.25, False, b"q"))

    def test_packed_attribute(self):
        t = from_header(text="""
            struct __attribute__((packed)) A { uint8_t a; uint32_t b; };
            struct B { uint8_t a; uint32_t b; } __attribute__((packed));
            struct C { uint8_t a; uint32_t b; };
        """)
        self.assertEqual((t.A.sizeof(), t.B.sizeof(), t.C.sizeof()), (5, 5, 8))

    def test_pack_push_pop_nesting(self):
        t = from_header(text="""
            struct Before { uint8_t a; uint32_t b; };
            #pragma pack(push, 1)
            struct Inside { uint8_t a; uint32_t b; };
            #pragma pack(push, 4)
            struct Four { uint8_t a; uint32_t b; };
            #pragma pack(pop)
            struct Inside2 { uint8_t a; uint32_t b; };
            #pragma pack(pop)
            struct After { uint8_t a; uint32_t b; };
            #pragma pack(1)
            struct One { uint8_t a; uint32_t b; };
            #pragma pack()
            struct Reset { uint8_t a; uint32_t b; };
        """)
        sizes = {n: c.sizeof() for n, c in t.structs.items()}
        self.assertEqual(sizes, {"Before": 8, "Inside": 5, "Inside2": 5, "After": 8, "One": 5, "Reset": 8})
        self.assertIn("Four", t.skipped)
        self.assertIn("pack(4)", t.skipped["Four"])


class AlignedDefault(unittest.TestCase):
    H = "struct Mixed { uint8_t a; uint32_t b; uint8_t c; uint64_t d; uint16_t e; };"

    def test_padding_like_c(self):
        t = from_header(text=self.H)
        self.assertEqual(t.Mixed.sizeof(), 32)
        m = t.Mixed(1, 2, 3, 4, 5)
        self.assertEqual(m.to_bytes(), struct.pack("<B3xIB7xQH6x", 1, 2, 3, 4, 5))

    def test_matches_ctypes_native(self):
        t = from_header(text=self.H)

        class Ref(ctypes.Structure):
            _fields_ = [("a", ctypes.c_uint8), ("b", ctypes.c_uint32), ("c", ctypes.c_uint8),
                        ("d", ctypes.c_uint64), ("e", ctypes.c_uint16)]
        self.assertEqual(t.Mixed.sizeof(), ctypes.sizeof(Ref))

    def test_align_override(self):
        t = from_header(text=self.H, align=False)
        self.assertEqual(t.Mixed.sizeof(), 16)
        t = from_header(text="#pragma pack(1)\nstruct S { uint8_t a; uint32_t b; };", align=True)
        self.assertEqual(t.S.sizeof(), 8)


class Enums(unittest.TestCase):
    H = """
    #define BASE 10
    enum Color { RED, GREEN, BLUE = 7, ALPHA };
    enum Flags { F_A = 1 << 0, F_B = 1 << 1, F_C = F_B << 1, F_ALL = F_A | F_B | F_C };
    enum Neg { LOW = -3, MID, HIGH = BASE * 2 + 1 };
    enum class Mode : uint8_t { Off, On = 0x10, Auto = 'A' };
    enum class Wide : uint16_t { X = 1000 };
    typedef enum { T_ONE = 1, T_TWO } Typed;
    struct Uses {
        enum Color color;
        Typed t;
        enum Mode m;
        Wide w;
        uint8_t after;
    };
    """

    def setUp(self):
        self.t = from_header(text=self.H)

    def test_no_skips(self):
        self.assertEqual(self.t.skipped, {})

    def test_values(self):
        t = self.t
        self.assertEqual({m.name: m.value for m in t.Color}, {"RED": 0, "GREEN": 1, "BLUE": 7, "ALPHA": 8})
        self.assertEqual({m.name: m.value for m in t.Flags}, {"F_A": 1, "F_B": 2, "F_C": 4, "F_ALL": 7})
        self.assertEqual({m.name: m.value for m in t.Neg}, {"LOW": -3, "MID": -2, "HIGH": 21})
        self.assertEqual({m.name: m.value for m in t.Mode}, {"Off": 0, "On": 16, "Auto": 65})
        self.assertTrue(issubclass(t.Color, enum.IntEnum))

    def test_constants(self):
        c = self.t.constants
        self.assertEqual((c["BLUE"], c["F_ALL"], c["Mode::On"], c["T_TWO"]), (7, 7, 16, 2))

    def test_struct_with_enum_fields(self):
        t = self.t
        u = t.Uses(t.Color.BLUE, t.Typed.T_TWO, t.Mode.On, t.Wide.X, 9)
        self.assertEqual(t.Uses.from_bytes(u.to_bytes()), u)
        self.assertIs(t.Uses.from_bytes(u.to_bytes()).m, t.Mode.On)

    def test_enum_sizes(self):
        # C enums are int-sized (4 bytes); enum class uses its underlying type
        t = from_header(text=self.H, align=False)
        self.assertEqual(t.Uses.sizeof(), 4 + 4 + 1 + 2 + 1)

    def test_roundtrip_formats(self):
        t = self.t
        u = t.Uses(t.Color.GREEN, t.Typed.T_ONE, t.Mode.Auto, t.Wide.X, 1)
        self.assertEqual(t.Uses.from_json(u.to_json()), u)
        self.assertEqual(t.Uses.from_xml(u.to_xml()), u)

    def test_negative_enum_is_signed(self):
        t = from_header(text="enum Neg { A = -1, B }; struct S { enum Neg n; };")
        self.assertEqual(t.S(t.Neg.A).to_bytes(), b"\xff\xff\xff\xff")

    def test_header_roundtrip_enum(self):
        again = from_header(text=self.t.Uses.c_header())
        self.assertEqual(again.Uses.sizeof(), self.t.Uses.sizeof())


class Arrays(unittest.TestCase):
    H = """
    #define NAME_LEN 8
    #define ROWS 2
    #define COLS (ROWS + 1)
    enum { COUNT = 3 };
    typedef uint8_t mac_t[6];
    typedef struct { int16_t x, y; } Vec;
    #pragma pack(push, 1)
    struct Data {
        char name[NAME_LEN];
        uint8_t raw[4];
        unsigned char raw2[2];
        mac_t mac;
        int16_t grid[ROWS][COLS];
        uint32_t counts[COUNT];
        char names[2][4];
        Vec points[2];
        uint8_t blobs[2][3];
    };
    #pragma pack(pop)
    """

    def setUp(self):
        self.t = from_header(text=self.H)

    def test_sizes(self):
        self.assertEqual(self.t.skipped, {})
        self.assertEqual(self.t.Data.sizeof(), 8 + 4 + 2 + 6 + 2 * 3 * 2 + 12 + 8 + 8 + 6)

    def test_roundtrips(self):
        t = self.t
        d = t.Data(
            name="widget", raw=b"\x01\x02\x03\x04", raw2=b"ab", mac=b"\x00\x11\x22\x33\x44\x55",
            grid=[[1, 2, 3], [4, 5, 6]], counts=[7, 8, 9], names=["ab", "cd"],
            points=[t.Vec(1, 2), t.Vec(3, 4)], blobs=[b"abc", b"def"])
        raw = d.to_bytes()
        self.assertEqual(raw[:8], b"widget\0\0")
        self.assertEqual(t.Data.from_bytes(raw), d)
        self.assertEqual(t.Data.from_json(d.to_json()), d)
        self.assertEqual(t.Data.from_xml(d.to_xml()), d)
        self.assertEqual(t.Data.from_ctypes(d.to_ctypes()), d)

    def test_matches_ctypes(self):
        class Vec(ctypes.Structure):
            _pack_ = 1
            if sys.version_info >= (3, 14):
                _layout_ = "ms"
            _fields_ = [("x", ctypes.c_int16), ("y", ctypes.c_int16)]

        class Ref(ctypes.Structure):
            _pack_ = 1
            if sys.version_info >= (3, 14):
                _layout_ = "ms"
            _fields_ = [("name", ctypes.c_char * 8), ("raw", ctypes.c_uint8 * 4), ("raw2", ctypes.c_ubyte * 2),
                        ("mac", ctypes.c_uint8 * 6), ("grid", (ctypes.c_int16 * 3) * 2),
                        ("counts", ctypes.c_uint32 * 3), ("names", (ctypes.c_char * 4) * 2),
                        ("points", Vec * 2), ("blobs", (ctypes.c_uint8 * 3) * 2)]
        self.assertEqual(self.t.Data.sizeof(), ctypes.sizeof(Ref))


class Bitfields(unittest.TestCase):
    H = """
    #pragma pack(1)
    struct Flags {
        uint8_t  ready : 1;
        uint8_t  mode  : 3;
        uint8_t  level : 4;
        uint16_t big   : 12;
        uint16_t rest  : 4;
        int8_t   sgn   : 3;
    };
    """

    def test_bitfields(self):
        t = from_header(text=self.H)
        f = t.Flags(1, 5, 9, 0xABC, 3, -2)
        self.assertEqual(t.Flags.sizeof(), 1 + 2 + 1)
        self.assertEqual(t.Flags.from_bytes(f.to_bytes()), f)
        self.assertEqual(f.to_bytes()[0], 1 | (5 << 1) | (9 << 4))

    def test_matches_ctypes(self):
        t = from_header(text=self.H)

        class Ref(ctypes.Structure):
            _pack_ = 1
            if sys.version_info >= (3, 14):
                _layout_ = "ms"
            _fields_ = [("ready", ctypes.c_uint8, 1), ("mode", ctypes.c_uint8, 3), ("level", ctypes.c_uint8, 4),
                        ("big", ctypes.c_uint16, 12), ("rest", ctypes.c_uint16, 4), ("sgn", ctypes.c_int8, 3)]
        f = t.Flags(1, 5, 9, 0xABC, 3, -2)
        self.assertEqual(f.to_bytes(), bytes(Ref(1, 5, 9, 0xABC, 3, -2)))

    def test_unnamed_padding_bitfield_skipped(self):
        t = from_header(text="struct P { uint8_t a : 3; uint8_t : 5; uint8_t b; };")
        self.assertIn("P", t.skipped)


class Preprocessor(unittest.TestCase):
    def test_include_guard_and_ifdef(self):
        text = """
        #ifndef MY_H
        #define MY_H
        #include <stdint.h>
        #ifdef USE_BIG
        struct S { uint64_t v; };
        #else
        struct S { uint8_t v; };
        #endif
        #if defined(EXTRA) && EXTRA > 1
        struct Extra { uint8_t e; };
        #elif 1
        struct Other { uint8_t o; };
        #endif
        #if 0
        struct Never { uint8_t n; };
        #endif
        #endif
        """
        t = from_header(text=text)
        self.assertEqual((t.S.sizeof(), "Extra" in t, "Other" in t, "Never" in t), (1, False, True, False))
        t = from_header(text=text, defines={"USE_BIG": 1, "EXTRA": 2})
        self.assertEqual((t.S.sizeof(), "Extra" in t, "Other" in t), (8, True, False))

    def test_macro_types_and_function_macros(self):
        t = from_header(text="""
            #define U8 uint8_t
            #define ARR(n) [n]
            #define SZ 3
            #define BYTES (SZ * 2)
            struct S { U8 a; U8 b[BYTES]; };
        """)
        self.assertEqual(t.S.sizeof(), 1 + 6)

    def test_multiline_define_and_sizeof_and_ternary(self):
        t = from_header(text="""
            #define TOTAL \\
                (4 + \\
                 4)
            const int K = sizeof(uint32_t) * 2;
            struct S { uint8_t a[TOTAL]; uint8_t b[K > 7 ? 2 : 3]; uint8_t c[sizeof(long long)]; };
        """)
        self.assertEqual(t.S.sizeof(), 8 + 2 + 8)
        self.assertEqual(t.constants["K"], 8)

    def test_comments_do_not_leak(self):
        t = from_header(text="""
            /* struct Hidden { uint8_t x; }; */
            // struct Hidden2 { uint8_t x; };
            struct Real { uint8_t a; /* uint8_t b; */ // uint8_t c;
            };
            const char *s = "struct Hidden3 { int x; };";
        """)
        self.assertEqual(list(t.structs), ["Real"])


class NestedAndPointers(unittest.TestCase):
    H = """
    struct Inner { uint16_t a; uint8_t b; };
    struct Node;
    typedef void (*callback_t)(int, void *);
    struct Outer {
        struct Inner first;
        struct Inner list[2];
        struct { uint8_t p; uint8_t q; } anon;
        struct Node *next;
        void *user;
        char *text;
        callback_t cb;
        void (*handler)(int);
        size_t len;
        long big;
        unsigned long ubig;
        long long ll;
        short s;
        unsigned u;
    };
    struct Node { int value; struct Node *next; };
    """

    def setUp(self):
        self.t = from_header(text=self.H)

    def test_everything_parsed(self):
        self.assertEqual(self.t.skipped, {})
        self.assertIn("Outer_anon", self.t.structs)

    def test_layout_matches_ctypes(self):
        class Inner(ctypes.Structure):
            _fields_ = [("a", ctypes.c_uint16), ("b", ctypes.c_uint8)]

        class Anon(ctypes.Structure):
            _fields_ = [("p", ctypes.c_uint8), ("q", ctypes.c_uint8)]

        class Ref(ctypes.Structure):
            _fields_ = [("first", Inner), ("list", Inner * 2), ("anon", Anon), ("next", ctypes.c_void_p),
                        ("user", ctypes.c_void_p), ("text", ctypes.c_void_p), ("cb", ctypes.c_void_p),
                        ("handler", ctypes.c_void_p), ("len", ctypes.c_size_t), ("big", ctypes.c_long),
                        ("ubig", ctypes.c_ulong), ("ll", ctypes.c_longlong), ("s", ctypes.c_short),
                        ("u", ctypes.c_uint)]
        self.assertEqual(self.t.Outer.sizeof(), ctypes.sizeof(Ref))
        self.assertEqual(self.t.Inner.sizeof(), ctypes.sizeof(Inner))

    def test_roundtrip(self):
        t = self.t
        o = t.Outer(t.Inner(1, 2), [t.Inner(3, 4), t.Inner(5, 6)], t.Outer_anon(7, 8), 0x1000, 0, 0x2000,
                    0x3000, 0x4000, 99, -5, 6, -7, -8, 9)
        self.assertEqual(t.Outer.from_bytes(o.to_bytes()), o)
        self.assertEqual(t.Outer.from_ctypes(o.to_ctypes()), o)

    def test_long_size_option(self):
        text = "#pragma pack(1)\nstruct L { long a; unsigned long b; };"
        self.assertEqual(from_header(text=text, long_size=4).L.sizeof(), 8)
        self.assertEqual(from_header(text=text, long_size=8).L.sizeof(), 16)

    def test_self_referential_pointer_ok(self):
        n = self.t.Node(5, 0)
        self.assertEqual(self.t.Node.from_bytes(n.to_bytes()), n)


class CppFeatures(unittest.TestCase):
    H = """
    #pragma once
    extern "C" {
    namespace net {
        constexpr int kMax = 4;
        using byte_t = uint8_t;
        enum class Kind : uint8_t { A = 1, B };
        struct Pkt {
            Kind kind;
            byte_t data[kMax];
            int32_t id = 0;
            Pkt() : id(1) {}
            int size() const { return 4; }
            static int counter;
            static const int kLimit = 3;
        };
        class Cls {
        public:
            uint16_t a;
        private:
            uint16_t b;
        };
    }
    }
    """

    def test_parsed(self):
        t = from_header(text=self.H, align=False)
        self.assertEqual(t.skipped, {})
        self.assertEqual(t.Pkt.sizeof(), 1 + 4 + 4)
        self.assertEqual(t.Cls.sizeof(), 4)
        self.assertEqual(t.constants["kMax"], 4)
        p = t.Pkt(t.Kind.B, b"abcd", 5)
        self.assertEqual(t.Pkt.from_bytes(p.to_bytes()), p)

    def test_virtual_and_inheritance_skipped(self):
        t = from_header(text="""
            struct Base { uint8_t a; };
            struct Derived : Base { uint8_t b; };
            struct V { virtual void f(); uint8_t a; };
            struct Good { uint8_t a; };
        """)
        self.assertEqual(list(t.structs), ["Base", "Good"])
        self.assertIn("inheritance", t.skipped["Derived"])
        self.assertIn("virtual", t.skipped["V"])


class SkipsGracefully(unittest.TestCase):
    def test_reasons_and_survivors(self):
        t = from_header(text="""
            union U { int a; float b; };
            struct HasUnion { union U u; uint8_t x; };
            struct Flex { uint8_t n; uint8_t data[]; };
            struct Unknown { missing_t m; };
            struct Dep { struct Unknown u; };
            struct Fine { uint8_t a; };
            #pragma pack(2)
            struct Two { uint8_t a; uint32_t b; };
            #pragma pack()
            struct LongDouble { long double d; };
            struct Empty {};
            struct Dup { uint8_t a; uint8_t a; };
            struct OkOne { uint32_t v; };
            enum Big { HUGE_VAL_ = 1 << 40 };
            enum Broken { X = unknown_const };
            struct UsesBig { enum Big b; };
        """)
        for name, word in [("U", "union"), ("HasUnion", "union"), ("Flex", "flexible"),
                           ("Unknown", "unknown type missing_t"), ("Dep", "depends on skipped"),
                           ("Two", "pack(2)"), ("LongDouble", "long double"), ("Empty", "empty"),
                           ("Dup", "duplicate"), ("Broken", "cannot evaluate")]:
            self.assertIn(name, t.skipped, name)
            self.assertIn(word, t.skipped[name], name)
        self.assertEqual(sorted(t.structs), ["Fine", "OkOne", "UsesBig"])
        self.assertEqual(t.UsesBig.sizeof(), 8)  # an enum with a value above 32 bits widens to 64 bit
        self.assertEqual(t.OkOne.sizeof(), 4)

    def test_empty_and_garbage_input(self):
        self.assertEqual(dict(from_header(text="")), {})
        t = from_header(text="int f(int a) { return a; }\nstatic inline void g(void) {}\nint x = 3;\n"
                             "struct S { uint8_t a; };\n}}} ;;; struct")
        self.assertEqual(list(t.structs), ["S"])


class Api(unittest.TestCase):
    H = "struct P { uint8_t a; uint16_t b; };"

    def test_from_file(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("x.h", "x.hpp"):
                path = Path(d) / name
                path.write_text(self.H, encoding="utf-8")
                self.assertEqual(from_header(path).P.sizeof(), 4)
                self.assertEqual(from_header(str(path)).P.sizeof(), 4)

    def test_both_spellings(self):
        self.assertIs(firstclass.from_header, from_header)
        self.assertIs(firstclass.firstclass.from_header, from_header)

    def test_args_validated(self):
        with self.assertRaises(TypeError):
            from_header()
        with self.assertRaises(TypeError):
            from_header("a.h", text="x")
        with self.assertRaises(ValueError):
            from_header(text="", byteorder="middle")
        with self.assertRaises(FileNotFoundError):
            from_header("definitely/missing.h")

    def test_result_helpers(self):
        t = from_header(text=self.H)
        self.assertIs(t["P"], t.P)
        with self.assertRaises(AttributeError):
            t.Nope

    def test_generated_classes_are_full_firstclass(self):
        t = from_header(text="struct P { int32_t x; char name[4]; };")
        p = t.P(1, "ab")
        self.assertEqual(t.P.from_json(p.to_json()), p)
        self.assertEqual(t.P.unpack_from(b"\0" + p.to_bytes(), 1), p)
        self.assertIn("struct P {", t.P.c_header())

    def test_c_header_roundtrip_is_stable(self):
        t = from_header(text=NestedAndPointers.H)
        again = from_header(text=t.Outer.c_header(), align=True)
        self.assertEqual(sorted(again.structs), ["Inner", "Outer", "Outer_anon"])  # pointer targets are not emitted
        for name, cls in again.structs.items():
            self.assertEqual(cls.sizeof(), t.structs[name].sizeof(), name)


if __name__ == "__main__":
    unittest.main()
