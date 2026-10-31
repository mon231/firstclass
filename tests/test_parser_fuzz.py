"""The C and Rust parsers must survive any input: never raise, never hang."""
import faulthandler
import random
import re
import time
import unittest

from firstclass import from_header, from_rust

C_SAMPLE = """
#ifndef GUARD_H
#define GUARD_H
#include <stdint.h>
#define N 4
#define SIZE (N * 2)
enum Color { RED, GREEN = 5, BLUE = RED + 7 };
enum class Mode : uint8_t { Off, On = 0x10 };
#pragma pack(push, 1)
typedef struct Header {
    uint32_t magic; uint8_t kind : 3, flag : 5; char name[SIZE]; uint8_t raw[N];
    enum Color color; Mode mode; struct { int16_t x, y; } pos; void (*cb)(int); int *ptr;
} Header_t;
#pragma pack(pop)
namespace ns { struct Pair { uint16_t a; uint16_t b = 3; Pair() {} int size() const { return 4; } }; }
union U { int a; float b; };
struct __attribute__((packed)) Tail { uint8_t a; uint32_t b; wchar_t w[2]; };
#if defined(FOO) || 0
struct Hidden { uint8_t x; };
#endif
#endif
"""

RUST_SAMPLE = """
use std::os::raw::c_int;
pub const N: usize = 4;
#[repr(u8)] #[derive(Clone, Copy)] pub enum Status { Ok = 0, Busy, Failed = 0x80 }
#[repr(C, packed)]
pub struct Header {
    pub magic: u32, pub version: u8, pub status: Status, pub name: [c_char; N * 2],
    pub grid: [[u16; 3]; N], pub ptr: *mut Header, pub cb: Option<extern "C" fn(c_int) -> c_int>,
}
#[repr(C)] struct Padded { a: u8, b: u32, c: [u8; 3] }
impl Padded { pub fn new() -> Self { Padded { a: 0, b: 0, c: [0; 3] } } }
mod inner { #[repr(C)] pub struct Deep<'a> { pub r: &'a u8 } }
struct Pair(u8, u16);
"""

C_VOCAB = ("struct union enum class typedef uint8_t uint32_t int char unsigned long const volatile static "
           "; { } ( ) [ ] , = : * & < > :: -> ... 0 1 3 255 0x10 'a' a b c N Foo "
           "#define #undef #if #ifdef #ifndef #elif #else #endif #include #pragma pack(push,1) pack(pop) "
           "__attribute__((packed)) namespace template virtual sizeof extern \"C\" public: using operator "
           "bool wchar_t size_t void double float short signed + - * / % << >> | & ^ ~ ! ? == != && ||").split()
RUST_VOCAB = ("struct enum union pub mod impl type const static fn unsafe extern use crate self as "
              "#[repr(C)] #[repr(u8)] #[repr(C,packed)] #[derive(Clone)] #![allow(x)] u8 u16 u32 u64 i32 usize f32 bool "
              "c_int c_char [ ] ( ) { } < > :: -> => , ; : = * & ' 'a 0 1 3 255 0x10 b'a' N Foo Option Box "
              "mut dyn where \"str\" /*c*/ + - << >> | ! ? .. r#type").split()


def mutate(rng, source, vocab):
    tokens = re.findall(r"\S+", source)
    for _ in range(rng.randint(1, 6)):
        if not tokens:
            break
        i = rng.randrange(len(tokens))
        action = rng.choice(["delete", "duplicate", "swap", "insert", "replace", "truncate"])
        if action == "delete":
            del tokens[i]
        elif action == "duplicate":
            tokens.insert(i, tokens[i])
        elif action == "swap":
            j = rng.randrange(len(tokens))
            tokens[i], tokens[j] = tokens[j], tokens[i]
        elif action == "insert":
            tokens.insert(i, rng.choice(vocab))
        elif action == "replace":
            tokens[i] = rng.choice(vocab)
        else:
            tokens = tokens[:i]
    sep = rng.choice([" ", "\n"])
    return sep.join(tokens)


def soup(rng, vocab):
    return rng.choice([" ", "\n"]).join(rng.choice(vocab) for _ in range(rng.randint(0, 60)))


class Watchdog(unittest.TestCase):
    def setUp(self):
        faulthandler.dump_traceback_later(240, exit=True)  # a hang would otherwise block the whole run

    def tearDown(self):
        faulthandler.cancel_dump_traceback_later()

    def survive(self, parse, text, label):
        start = time.perf_counter()
        try:
            result = parse(text)
        except Exception as e:  # noqa: BLE001 - the point of the test
            self.fail(f"{label}: {type(e).__name__}: {e}\n--- input ---\n{text}")
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5, f"{label}: took {elapsed:.1f}s\n{text}")
        # whatever came out must be usable
        for cls in result.structs.values():
            cls.sizeof()
        for name, reason in result.skipped.items():
            self.assertIsInstance(reason, str)
            self.assertTrue(reason, name)
        return result


class HeaderParser(Watchdog):
    def test_samples_parse_cleanly(self):
        t = from_header(text=C_SAMPLE)
        self.assertEqual(sorted(t.structs), ["Header", "Header_pos", "Pair", "Tail"])
        self.assertIn("U", t.skipped)

    def test_mutated_valid_sources(self):
        for seed in range(400):
            rng = random.Random(seed)
            text = mutate(rng, C_SAMPLE, C_VOCAB)
            self.survive(lambda t: from_header(text=t), text, f"mutation seed {seed}")

    def test_random_token_soup(self):
        for seed in range(400):
            rng = random.Random(seed + 10_000)
            self.survive(lambda t: from_header(text=t), soup(rng, C_VOCAB), f"soup seed {seed}")

    def test_random_characters(self):
        for seed in range(100):
            rng = random.Random(seed + 20_000)
            text = "".join(rng.choice("abc{}()[];,:*#=<>\"' \n\\/0x9") for _ in range(rng.randint(0, 200)))
            self.survive(lambda t: from_header(text=t), text, f"chars seed {seed}")

    def test_pathological_shapes(self):
        shapes = {
            "deep braces": "struct S " + "{ " * 200 + "}" * 200 + ";",
            "deep parens": "struct S { uint8_t a[" + "(" * 300 + "1" + ")" * 300 + "]; };",
            "long macro chain": "".join(f"#define M{i} M{i + 1}\n" for i in range(300)) + "#define M300 2\nstruct S { char a[M0]; };",
            "self macro": "#define A A\nstruct S { char a[A]; };",
            "mutual macros": "#define A B\n#define B A\nstruct S { char a[A]; };",
            "huge array": "struct S { char a[99999999999]; };",
            "huge expr": "struct S { char a[1 << 70]; };",
            "many fields": "struct S { " + "uint8_t f%d; " * 1 % 0 + "".join(f"uint8_t g{i}; " for i in range(2000)) + "};",
            "unterminated comment": "struct S { uint8_t a; /* never closed",
            "unterminated string": 'const char *s = "abc\nstruct S { uint8_t a; };',
            "lone hash": "#\n# \n#\n#define\n#define 3\n#if\n#elif\n#else\n#endif\n#endif\n",
            "empty struct names": "struct { uint8_t a; }; struct ; enum ; typedef ;",
            "typedef loop": "typedef A B; typedef B A; struct S { A a; };",
            "struct loop": "struct A { struct B b; }; struct B { struct C c; }; struct C { struct A a; };",
            "nul bytes": "struct S { uint8_t a; };\x00struct T { uint8_t b; };",
            "non-ascii": "struct S { uint8_t é; }; // €\nstruct T { uint8_t b; };",
            "pack stack": "#pragma pack(pop)\n" * 50 + "#pragma pack(push,1)\n" * 50 + "struct S { uint8_t a; uint32_t b; };",
            "huge enum": "enum E { " + ", ".join(f"V{i}" for i in range(3000)) + " };",
            "bit widths": "struct S { uint8_t a : 0; uint8_t b : -1; uint8_t c : 99; uint8_t d : N; };",
        }
        for label, text in shapes.items():
            with self.subTest(label):
                self.survive(lambda t: from_header(text=t), text, label)

    def test_huge_array_is_skipped_not_allocated(self):
        t = from_header(text="struct S { char a[99999999999]; }; struct Ok { uint8_t a; };")
        self.assertIn("Ok", t.structs)

    def test_error_messages_do_not_leak_internals(self):
        t = from_header(text="struct S { missing_t m; };")
        self.assertEqual(t.skipped["S"], "S.m: unknown type missing_t")


class RustParser(Watchdog):
    def test_sample_parses_cleanly(self):
        t = from_rust(text=RUST_SAMPLE)
        self.assertIn("Header", t.structs)
        self.assertIn("Padded", t.structs)
        self.assertIn("Pair", t.skipped)
        self.assertIn("Deep", t.skipped)

    def test_mutated_valid_sources(self):
        for seed in range(400):
            rng = random.Random(seed)
            self.survive(lambda t: from_rust(text=t), mutate(rng, RUST_SAMPLE, RUST_VOCAB), f"mutation seed {seed}")

    def test_random_token_soup(self):
        for seed in range(400):
            rng = random.Random(seed + 10_000)
            self.survive(lambda t: from_rust(text=t), soup(rng, RUST_VOCAB), f"soup seed {seed}")

    def test_random_characters(self):
        for seed in range(100):
            rng = random.Random(seed + 20_000)
            text = "".join(rng.choice("abc{}()[];,:*#=<>\"' \n\\/0x9&!") for _ in range(rng.randint(0, 200)))
            self.survive(lambda t: from_rust(text=t), text, f"chars seed {seed}")

    def test_pathological_shapes(self):
        shapes = {
            "deep brackets": "#[repr(C)] struct S { a: " + "[" * 100 + "u8" + "; 1]" * 100 + " }",
            "deep generics": "#[repr(C)] struct S { a: " + "Option<" * 50 + "&u8" + ">" * 50 + " }",
            "unterminated block comment": "#[repr(C)] struct S { a: u8 } /* never closed",
            "unterminated string": '#[repr(C)] struct S { a: u8 } const X: &str = "abc',
            "nested comments": "/* a /* b /* c */ */ */ #[repr(C)] struct S { a: u8 }",
            "lifetimes": "#[repr(C)] struct S<'a, 'b: 'a> { r: &'a &'b u8 }",
            "raw strings": 'const X: &str = r#"struct Hidden { a: u8 }"#; #[repr(C)] struct S { a: u8 }',
            "many fields": "#[repr(C)] struct S { " + "".join(f"f{i}: u8, " for i in range(2000)) + "}",
            "const cycle": "const A: usize = B; const B: usize = A; #[repr(C)] struct S { a: [u8; A] }",
            "huge array": "#[repr(C)] struct S { a: [u8; 99999999999] } #[repr(C)] struct Ok { a: u8 }",
            "inner attribute in enum body": "#[repr(u8)] enum E { #![x] A, #!b B } #[repr(C)] struct Ok { a: u8 }",
            "attr soup": "#[a] #[b(c)] #![d] #[e = \"f\"] #[repr(C)] struct S { a: u8 }",
            "empty items": "struct; enum; mod; impl; fn; const; type; use;",
            "non-ascii": "#[repr(C)] struct S { a: u8 } // €\n#[repr(C)] struct T { b: u8 }",
            "nul bytes": "#[repr(C)] struct S { a: u8 }\x00#[repr(C)] struct T { b: u8 }",
        }
        for label, text in shapes.items():
            with self.subTest(label):
                self.survive(lambda t: from_rust(text=t), text, label)

    def test_huge_array_is_skipped_not_allocated(self):
        t = from_rust(text="#[repr(C)] struct S { a: [u8; 99999999999] } #[repr(C)] struct Ok { a: u8 }")
        self.assertIn("Ok", t.structs)


if __name__ == "__main__":
    unittest.main()
