import ctypes
import unittest

from firstclass import from_header


def consts(src, **kw):
    return from_header(text=src, **kw).constants


class ConstantExpressions(unittest.TestCase):
    CASES = {
        "2 + 3 * 4": 14, "(2 + 3) * 4": 20, "10 - 4 - 3": 3, "2 * 3 % 4": 2,
        "1 << 4 | 1": 17, "1 << (4 | 1)": 32, "0xF0 >> 4": 15, "0xFF & 0x0F ^ 0x3": 12,
        "0xF0 | 0x0F": 255, "6 ^ 3": 5, "~0": -1, "~5 & 0xFF": 250, "-5 + 2": -3, "- -3": 3, "+4": 4,
        "!0": 1, "!5": 0, "!!7": 1,
        "3 > 2": 1, "3 < 2": 0, "3 >= 3": 1, "3 <= 2": 0, "3 == 3": 1, "3 != 3": 0,
        "1 && 0": 0, "1 && 2": 1, "0 || 0": 0, "0 || 5": 1, "1 || 0 && 0": 1,
        "1 ? 10 : 20": 10, "0 ? 10 : 20": 20, "0 ? 1 : 0 ? 2 : 3": 3, "1 ? 2 ? 4 : 5 : 6": 4,
        "7 / 2": 3, "-7 / 2": -3, "7 / -2": -3, "-7 / -2": 3, "7 % 3": 1, "-7 % 3": -1, "7 % -3": 1,
        "0x10": 16, "010": 8, "0b101": 5, "10UL": 10, "0x10u": 16, "5LL": 5, "0XfF": 255, "0": 0,
        "'a'": 97, "'\\n'": 10, "'\\0'": 0, "'A' + 1": 66, "'\\\\'": 92,
        "sizeof(uint8_t)": 1, "sizeof(uint64_t)": 8, "sizeof(int)": 4, "sizeof(short)": 2,
        "sizeof(long long)": 8, "sizeof(char)": 1, "sizeof(unsigned char)": 1, "sizeof(double)": 8,
        "sizeof(float)": 4, "sizeof(void *)": ctypes.sizeof(ctypes.c_void_p),
        "(int)5": 5, "(uint8_t)3 + 1": 4, "(unsigned long)2 * 3": 6, "((2))": 2, "(((1 + 1)) * 2)": 4,
        "1 + 2 == 3 && 4 > 3": 1, "1 << 2 + 1": 8, "100 / 10 / 5": 2, "2 * (3 + 4) - 1": 13,
    }

    def test_table(self):
        for expr, want in self.CASES.items():
            with self.subTest(expr):
                self.assertEqual(consts(f"const int X = {expr};").get("X"), want)

    def test_references_to_other_constants(self):
        c = consts("""
            #define A 4
            const int B = A * 2;
            constexpr int C = B + A;
            enum { D = C * 2, E };
            const int F = E + D;
        """)
        self.assertEqual((c["B"], c["C"], c["D"], c["E"], c["F"]), (8, 12, 24, 25, 49))

    def test_invalid_expressions_are_ignored_not_fatal(self):
        c = consts("""
            const int A = 1 / 0;
            const int B = 5 %% 0;
            const int C = unknown_name + 1;
            const int E = 1.5;
            const int OK = 3;
        """)
        self.assertEqual(c, {"OK": 3})

    def test_integer_defines_become_constants(self):
        c = consts("""
            #define A 3
            #define B (A * 2 + 1)
            #define HEX 0xFF
            #define NAME "text"
            #define EMPTY
            #define FUNC(x) ((x) + 1)
            #define USES_ENUM (SOME + 1)
            #define UNKNOWN (nothing_defined + 1)
            #define GONE 9
            #undef GONE
            enum { SOME = 4 };
        """, defines={"FROM_USER": 5})
        self.assertEqual({k: c[k] for k in ("A", "B", "HEX", "USES_ENUM", "FROM_USER")},
                         {"A": 3, "B": 7, "HEX": 255, "USES_ENUM": 5, "FROM_USER": 5})
        for absent in ("NAME", "EMPTY", "FUNC", "UNKNOWN", "GONE"):
            self.assertNotIn(absent, c)

    def test_only_const_declarations_become_constants(self):
        c = consts("int plain = 5; static int s = 6; const int yes = 7; void f(int a = 3);")
        self.assertEqual(c, {"yes": 7})

    def test_constant_types_and_qualifiers(self):
        c = consts("static const unsigned long A = 1; constexpr uint8_t B = 2; const volatile int C = 3;")
        self.assertEqual(c, {"A": 1, "B": 2, "C": 3})

    def test_scoped_enum_members_are_qualified(self):
        c = consts("enum class K : int { A = 1, B }; enum Plain { P = 7 }; const int X = K::B + P;")
        self.assertEqual((c["K::A"], c["K::B"], c["P"], c["X"]), (1, 2, 7, 9))
        self.assertNotIn("A", c)

    def test_enum_values(self):
        t = from_header(text="""
            enum E { A = 5, B, C = -1, D, E_ = 'x', F = 1 << 3, G = F | A, H = sizeof(int), };
        """)
        self.assertEqual({m.name: m.value for m in t.E},
                         {"A": 5, "B": 6, "C": -1, "D": 0, "E_": 120, "F": 8, "G": 13, "H": 4})

    def test_enum_trailing_comma_empty_and_anonymous(self):
        t = from_header(text="enum Empty {}; enum { ANON_A = 3, ANON_B }; enum One { O, };")
        self.assertEqual(list(t.enums), ["Empty", "One"])
        self.assertEqual(t.constants["ANON_B"], 4)


class PreprocessorEdges(unittest.TestCase):
    def names(self, src, **kw):
        return sorted(from_header(text=src, **kw).structs)

    def test_undef_and_redefine(self):
        src = """
        #define X 1
        #undef X
        #ifdef X
        struct WasDefined { uint8_t a; };
        #endif
        #define Y 1
        #define Y 2
        struct Sized { uint8_t a[Y]; };
        """
        t = from_header(text=src)
        self.assertEqual(list(t.structs), ["Sized"])
        self.assertEqual(t.Sized.sizeof(), 2)

    def test_elif_chain_takes_first_true_only(self):
        src = """
        #if V == 1
        struct One { uint8_t a; };
        #elif V == 2
        struct Two { uint8_t a; };
        #elif V == 2
        struct TwoAgain { uint8_t a; };
        #else
        struct Other { uint8_t a; };
        #endif
        """
        self.assertEqual(self.names(src, defines={"V": 1}), ["One"])
        self.assertEqual(self.names(src, defines={"V": 2}), ["Two"])
        self.assertEqual(self.names(src, defines={"V": 3}), ["Other"])
        self.assertEqual(self.names(src), ["Other"])

    def test_nested_conditionals(self):
        src = """
        #ifdef A
          #ifdef B
          struct AB { uint8_t x; };
          #else
          struct AnotB { uint8_t x; };
          #endif
        #else
          #ifdef B
          struct BnotA { uint8_t x; };
          #else
          struct Neither { uint8_t x; };
          #endif
        #endif
        """
        self.assertEqual(self.names(src, defines={"A": 1, "B": 1}), ["AB"])
        self.assertEqual(self.names(src, defines={"A": 1}), ["AnotB"])
        self.assertEqual(self.names(src, defines={"B": 1}), ["BnotA"])
        self.assertEqual(self.names(src), ["Neither"])

    def test_else_inside_inactive_block_stays_inactive(self):
        src = """
        #if 0
          #if 1
          struct Hidden1 { uint8_t x; };
          #else
          struct Hidden2 { uint8_t x; };
          #endif
        #endif
        struct Shown { uint8_t x; };
        """
        self.assertEqual(self.names(src), ["Shown"])

    def test_defined_forms_and_operators(self):
        src = """
        #define FOO
        #if defined FOO && !defined(BAR)
        struct A { uint8_t x; };
        #endif
        #if defined(BAR) || (defined(FOO) && 2 * 3 == 6)
        struct B { uint8_t x; };
        #endif
        #if !(1 > 2)
        struct C { uint8_t x; };
        #endif
        """
        self.assertEqual(self.names(src), ["A", "B", "C"])

    def test_macro_value_in_if(self):
        src = """
        #define VERSION 3
        #if VERSION >= 2 && VERSION < 5
        struct New { uint8_t x; };
        #else
        struct Old { uint8_t x; };
        #endif
        """
        self.assertEqual(self.names(src), ["New"])

    def test_unknown_identifier_in_if_is_zero(self):
        self.assertEqual(self.names("#if NOT_DEFINED\nstruct A { uint8_t x; };\n#else\nstruct B { uint8_t x; };\n#endif"), ["B"])

    def test_self_referential_and_chained_macros(self):
        src = """
        #define LOOP LOOP
        #define A B
        #define B 3
        struct S { uint8_t a[A]; uint8_t LOOP_; };
        """
        self.assertEqual(from_header(text=src).S.sizeof(), 3 + 1)

    def test_empty_and_function_like_macros_do_not_break(self):
        src = """
        #define EMPTY
        #define EXPORT __declspec(dllexport)
        #define MAX(a, b) ((a) > (b) ? (a) : (b))
        #define BIT(n) (1 << (n))
        struct EMPTY Tagged { uint8_t a; };
        """
        t = from_header(text=src)
        self.assertEqual(list(t.structs), ["Tagged"])

    def test_defines_argument_types(self):
        src = "struct S { uint8_t a[N]; TYPE b; };"
        t = from_header(text=src, defines={"N": 2, "TYPE": "uint16_t"}, align=False)
        self.assertEqual(t.S.sizeof(), 4)

    def test_line_continuation_inside_declaration(self):
        src = "struct S { uint8_t \\\n a; \\\n uint16_t b; };"
        self.assertEqual(from_header(text=src, align=False).S.sizeof(), 3)

    def test_ignored_directives(self):
        src = """
        #include <stdint.h>
        #include "other.h"
        #pragma once
        #pragma warning(disable: 4200)
        #error not evaluated when inactive
        #line 10
        struct S { uint8_t a; };
        """
        self.assertEqual(self.names(src.replace("#error not evaluated when inactive\n", "")), ["S"])

    def test_pack_inside_inactive_block_is_ignored(self):
        src = """
        #if 0
        #pragma pack(push, 1)
        #endif
        struct S { uint8_t a; uint32_t b; };
        """
        self.assertEqual(from_header(text=src).S.sizeof(), 8)

    def test_pack_push_without_number_keeps_current(self):
        src = """
        #pragma pack(1)
        #pragma pack(push)
        struct A { uint8_t a; uint32_t b; };
        #pragma pack(pop)
        struct B { uint8_t a; uint32_t b; };
        """
        t = from_header(text=src)
        self.assertEqual((t.A.sizeof(), t.B.sizeof()), (5, 5))

    def test_pop_on_empty_stack_resets(self):
        t = from_header(text="#pragma pack(pop)\nstruct A { uint8_t a; uint32_t b; };")
        self.assertEqual(t.A.sizeof(), 8)

    def test_pack_8_and_16_mean_natural(self):
        t = from_header(text="#pragma pack(8)\nstruct A { uint8_t a; uint64_t b; };")
        self.assertEqual(t.A.sizeof(), 16)

    def test_comment_markers_inside_strings_and_chars(self):
        src = '''
        const char *s = "/* not a comment */";
        const int Q = '/';
        struct S { uint8_t a; };  // trailing
        '''
        t = from_header(text=src)
        self.assertEqual(list(t.structs), ["S"])
        self.assertEqual(t.constants["Q"], 47)

    def test_windows_line_endings_and_tabs(self):
        src = "struct S {\r\n\tuint8_t a;\r\n\tuint16_t b;\r\n};\r\n"
        self.assertEqual(from_header(text=src, align=False).S.sizeof(), 3)


class DeclarationShapes(unittest.TestCase):
    def test_multiple_declarators(self):
        t = from_header(text="struct S { uint8_t a, b[2], *c; uint16_t d, e; };", align=False)
        self.assertEqual(t.S.sizeof(), 1 + 2 + ctypes.sizeof(ctypes.c_void_p) + 4)
        self.assertEqual([f for f in t.S.__dataclass_fields__], ["a", "b", "c", "d", "e"])

    def test_typedef_chain_and_pointer_typedef(self):
        t = from_header(text="""
            typedef uint8_t u8;
            typedef u8 byte;
            typedef byte *byte_ptr;
            typedef byte buf4[4];
            struct S { byte a; byte_ptr p; buf4 b; };
        """, align=False)
        self.assertEqual(t.skipped, {})
        self.assertEqual(t.S.sizeof(), 1 + ctypes.sizeof(ctypes.c_void_p) + 4)

    def test_forward_typedef_then_definition(self):
        t = from_header(text="""
            typedef struct Node Node;
            struct Node { int value; Node *next; };
            typedef struct Pair Pair_t;
            struct Pair { Node *a; Node *b; };
        """)
        self.assertEqual(t.skipped, {})
        self.assertIs(t.Pair, t.Pair_t)
        self.assertEqual(t.Node.sizeof(), 16 if ctypes.sizeof(ctypes.c_void_p) == 8 else 8)

    def test_struct_used_before_definition_by_value_is_resolved(self):
        t = from_header(text="struct Outer { struct Inner i; }; struct Inner { uint16_t a; };")
        self.assertEqual(t.skipped, {})
        self.assertEqual(t.Outer.sizeof(), 2)

    def test_variables_after_definition_are_ignored(self):
        t = from_header(text="struct S { uint8_t a; } s1, *s2; struct T { uint8_t b; } t1;")
        self.assertEqual(sorted(t.structs), ["S", "T"])

    def test_function_declarations_and_bodies_ignored(self):
        t = from_header(text="""
            int add(int a, int b);
            static inline int twice(int x) { return x * 2; }
            void (*global_cb)(int);
            extern int counter;
            struct S { uint8_t a; };
            struct S *make(void);
            struct S make2(struct S in) { struct S out = in; return out; }
            struct T { uint8_t b; };
        """)
        self.assertEqual(sorted(t.structs), ["S", "T"])

    def test_cv_qualifiers_and_attributes_on_fields(self):
        t = from_header(text="""
            struct S {
                const uint8_t a;
                volatile uint16_t b;
                uint32_t c __attribute__((aligned_unused_marker));
                __extension__ uint8_t d;
                uint8_t * const e;
            };
        """, align=False)
        self.assertEqual(t.skipped, {})
        self.assertEqual(t.S.sizeof(), 1 + 2 + 4 + 1 + ctypes.sizeof(ctypes.c_void_p))

    def test_builtin_type_spellings(self):
        spellings = {
            "char": None, "signed char": 1, "unsigned char": 1, "short": 2, "short int": 2,
            "unsigned short": 2, "unsigned short int": 2, "signed short": 2, "int": 4, "signed": 4,
            "signed int": 4, "unsigned": 4, "unsigned int": 4, "long long": 8, "long long int": 8,
            "unsigned long long": 8, "unsigned long long int": 8, "long int": ctypes.sizeof(ctypes.c_long),
            "long unsigned int": ctypes.sizeof(ctypes.c_long), "float": 4, "double": 8, "_Bool": 1, "bool": 1,
            "int8_t": 1, "uint16_t": 2, "int32_t": 4, "uint64_t": 8, "size_t": ctypes.sizeof(ctypes.c_size_t),
            "ssize_t": ctypes.sizeof(ctypes.c_void_p), "intptr_t": ctypes.sizeof(ctypes.c_void_p),
            "uintptr_t": ctypes.sizeof(ctypes.c_void_p), "ptrdiff_t": ctypes.sizeof(ctypes.c_void_p),
        }
        for spelling, size in spellings.items():
            with self.subTest(spelling):
                t = from_header(text=f"struct S {{ {spelling} v; }};")
                self.assertEqual(t.skipped, {})
                self.assertEqual(t.S.sizeof(), size or 1)

    def test_signedness_of_spellings(self):
        t = from_header(text="struct S { signed char a; unsigned char b; short c; unsigned short d; };")
        s = t.S(-1, 255, -1, 65535)
        self.assertEqual(t.S.from_bytes(s.to_bytes()), s)
        with self.assertRaises(ValueError):
            t.S(128, 0, 0, 0).to_bytes()

    def test_char_array_vs_byte_array_vs_int_array(self):
        t = from_header(text="struct S { char a[4]; unsigned char b[4]; signed char c[4]; int16_t d[2]; };",
                        align=False)
        s = t.S("hey", b"\x01\x02\x03\x04", [1, -2, 3, -4], [5, -6])
        self.assertEqual(t.S.from_bytes(s.to_bytes()), s)

    def test_default_member_initialisers_and_nsdmi(self):
        t = from_header(text="struct S { uint8_t a = 1; uint16_t b{2}; uint8_t c[2] = {1, 2}; uint8_t d; };",
                        align=False)
        self.assertEqual(t.skipped, {})
        self.assertEqual(t.S.sizeof(), 1 + 2 + 2 + 1)

    def test_access_specifiers_and_static_members(self):
        t = from_header(text="""
            class C {
            public:
                static const int K = 3;
                static int counter;
                uint8_t a;
            protected:
                uint16_t b;
            private:
                uint8_t c;
            };
        """, align=False)
        self.assertEqual(t.C.sizeof(), 4)

    def test_methods_operators_and_constructors_ignored(self):
        t = from_header(text="""
            struct S {
                S() = default;
                explicit S(int x) : a(x) {}
                ~S();
                uint8_t a;
                bool operator==(const S &o) const { return a == o.a; }
                uint8_t get() const;
                template <typename T> T cast() const { return (T)a; }
                uint8_t b;
            };
        """, align=False)
        self.assertEqual(t.skipped, {})
        self.assertEqual(t.S.sizeof(), 2)

    def test_nested_inline_struct_variants(self):
        t = from_header(text="""
            struct Outer {
                struct Named { uint8_t a; } n1;
                struct { uint16_t b; } anon;
                struct Named n2;
                enum { IN_A, IN_B } mode;
                enum Kind { K1 = 1 } kind;
            };
        """, align=False)
        self.assertEqual(t.skipped, {})
        self.assertEqual(sorted(t.structs), ["Named", "Outer", "Outer_anon"])
        self.assertEqual(t.Outer.sizeof(), 1 + 2 + 1 + 4 + 4)

    def test_anonymous_union_or_struct_member_skips_outer(self):
        t = from_header(text="struct S { uint8_t a; union { uint8_t b; uint16_t c; }; };")
        self.assertIn("S", t.skipped)

    def test_c_style_void_and_misc_tokens_do_not_hang(self):
        junk = "; ; } } ( ) [ ] ; struct union enum typedef template <> :: -> ... ; \"unterminated\n;"
        t = from_header(text=junk + "struct Ok { uint8_t a; };")
        self.assertEqual(list(t.structs), ["Ok"])

    def test_deeply_nested_braces_and_parens(self):
        src = "struct S { uint8_t a[((((1 + 1)) * ((2))))]; };"
        self.assertEqual(from_header(text=src).S.sizeof(), 4)

    def test_extern_c_and_nested_namespaces(self):
        t = from_header(text="""
            #ifdef __cplusplus
            extern "C" {
            #endif
            namespace a { namespace b { struct Deep { uint8_t x; }; } }
            namespace { struct Anon { uint8_t y; }; }
            #ifdef __cplusplus
            }
            #endif
            struct Outside { uint8_t z; };
        """)
        self.assertEqual(sorted(t.structs), ["Anon", "Deep", "Outside"])


class BuildSemantics(unittest.TestCase):
    def test_same_name_struct_in_different_ifdef_branches_last_active_wins(self):
        src = "struct S { uint8_t a; }; struct S { uint16_t a; };"
        self.assertEqual(from_header(text=src).S.sizeof(), 2)

    def test_skip_does_not_poison_independent_structs(self):
        t = from_header(text="struct Bad { missing_t m; }; struct Good { uint8_t a; }; struct Dep { struct Bad b; };")
        self.assertEqual(sorted(t.structs), ["Good"])
        self.assertEqual(sorted(t.skipped), ["Bad", "Dep"])

    def test_by_value_self_containment_is_skipped(self):
        t = from_header(text="struct Loop { struct Loop inner; uint8_t a; }; struct Fine { uint8_t a; };")
        self.assertIn("contains itself", t.skipped["Loop"])
        self.assertEqual(list(t.structs), ["Fine"])

    def test_mutual_by_value_cycle_is_skipped(self):
        t = from_header(text="struct A { struct B b; }; struct B { struct A a; };")
        self.assertEqual(sorted(t.skipped), ["A", "B"])

    def test_array_size_errors_skip_only_that_struct(self):
        t = from_header(text="struct Bad { uint8_t a[UNKNOWN_SIZE]; }; struct Zero { uint8_t a[0]; }; "
                             "struct Good { uint8_t a[1]; };")
        self.assertEqual(sorted(t.skipped), ["Bad", "Zero"])
        self.assertEqual(list(t.structs), ["Good"])

    def test_bitfield_errors(self):
        t = from_header(text="""
            struct TooWide { uint8_t a : 9; };
            struct OnFloat { float f : 3; };
            struct OnPtr { void *p : 3; };
            struct Fine { uint8_t a : 8; };
        """)
        self.assertEqual(sorted(t.skipped), ["OnFloat", "OnPtr", "TooWide"])
        self.assertEqual(list(t.structs), ["Fine"])

    def test_field_named_like_python_keywords_and_builtins(self):
        t = from_header(text="struct S { uint8_t class; uint8_t from; uint8_t type; uint8_t id; };")
        self.assertEqual(t.skipped, {})
        self.assertEqual(list(t.S.__dataclass_fields__), ["class_", "from_", "type", "id"])  # keywords get a "_"
        s = t.S(1, 2, 3, 4)
        self.assertEqual(t.S.from_bytes(s.to_bytes()), s)
        self.assertEqual(t.S.from_dict(s.to_dict()), s)

    def test_field_named_like_generated_method_is_skipped(self):
        t = from_header(text="struct S { uint8_t sizeof; }; struct T { uint8_t to_bytes; };")
        self.assertEqual(sorted(t.skipped), ["S", "T"])

    def test_result_is_stable_across_calls(self):
        src = "struct S { uint8_t a; };"
        a, b = from_header(text=src), from_header(text=src)
        self.assertIsNot(a.S, b.S)
        self.assertEqual(a.S.sizeof(), b.S.sizeof())

    def test_all_formats_on_generated_class(self):
        t = from_header(text="""
            enum class M : uint8_t { A = 1, B };
            struct P { uint8_t x; };
            struct S { char name[4]; uint8_t raw[2]; int16_t v[2]; M m; P p; uint8_t flag : 1; uint8_t rest : 7; };
        """, align=False)
        s = t.S("ab", b"\x01\x02", [1, -1], t.M.B, t.P(7), 1, 5)
        for roundtrip in (lambda x: t.S.from_bytes(x.to_bytes()), lambda x: t.S.from_json(x.to_json()),
                          lambda x: t.S.from_xml(x.to_xml()), lambda x: t.S.from_ctypes(x.to_ctypes()),
                          lambda x: t.S.unpack_from(b"\0" + x.to_bytes(), 1)):
            self.assertEqual(roundtrip(s), s)

    def test_big_endian_generated_class_matches_manual(self):
        from firstclass import firstclass, uint16_t, uint32_t
        t = from_header(text="struct S { uint16_t a; uint32_t b; };", byteorder="big", align=False)

        @firstclass(byteorder="big")
        class Manual:
            a: uint16_t
            b: uint32_t
        self.assertEqual(t.S(1, 2).to_bytes(), Manual(1, 2).to_bytes())


if __name__ == "__main__":
    unittest.main()
