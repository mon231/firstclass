"""Layouts checked against the real C compiler and rustc (skipped when they are not installed)."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gen import Generator  # noqa: E402

from firstclass import (  # noqa: E402
    array_t, bitfield, bool_t, bytes_t, char_array, enum_t, firstclass, float32_t, float64_t, int8_t, int16_t,
    int32_t, uint8_t, uint16_t, uint32_t, uint64_t, wchar_array,
)
import ctypes  # noqa: E402
import enum  # noqa: E402

SEEDS = range(40)


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=180)


def find_c_compiler():
    for name in ("gcc", "cc", "clang"):
        path = shutil.which(name)
        if path:
            return path
    return None


def ms_bitfield_flag(cc):
    """firstclass lays bitfields out like MSVC / ctypes; `-mms-bitfields` makes gcc and clang do the same."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "t.c"), "w") as f:
            f.write("int main(void) { return 0; }\n")
        r = run([cc, "-mms-bitfields", "t.c", "-o", "t.out"], d)
        return ["-mms-bitfields"] if r.returncode == 0 else None


CC = find_c_compiler()
MS_FLAG = ms_bitfield_flag(CC) if CC else None
RUSTC = shutil.which("rustc")


def schemas_of(cls):
    """Every schema reachable from cls, dependencies first."""
    seen, order = set(), []

    def walk(sc):
        if sc.cls in seen:
            return
        seen.add(sc.cls)
        for dep in sc._nested_schemas():
            walk(dep)
        order.append(sc)
    walk(cls.__firstclass__)
    return order


def expected_layout(cls, rust=False):
    """{(struct, field): offset} plus {struct: size} as firstclass computes them."""
    sizes, offsets = {}, {}
    for sc in schemas_of(cls):
        items, total, _ = sc.layout()
        sizes[sc.cls.__name__] = total
        groups = 0
        for item in items:
            if item[0] == "field":
                offsets[(sc.cls.__name__, sc.fields[item[1]][0].name)] = item[1:][1]
            elif rust:  # bitfield groups are plain integer fields in the Rust output
                offsets[(sc.cls.__name__, f"bits_{groups}")] = item[2]
                groups += 1
    return sizes, offsets


def c_program(cls):
    lines = ["#include <stdio.h>", "#include <stddef.h>", cls.c_header(), "int main(void) {"]
    sizes, offsets = expected_layout(cls)
    for name in sizes:
        lines.append(f'  printf("S {name} %zu\\n", sizeof(struct {name}));')
    for (name, field) in offsets:
        lines.append(f'  printf("O {name}.{field} %zu\\n", offsetof(struct {name}, {field}));')
    lines += ["  return 0;", "}", ""]
    return "\n".join(lines)


def parse_report(text):
    sizes, offsets = {}, {}
    for line in text.splitlines():
        kind, key, value = line.split()
        if kind == "S":
            sizes[key] = int(value)
        else:
            name, field = key.split(".")
            offsets[(name, field)] = int(value)
    return sizes, offsets


def compile_and_run_c(source, extra=()):
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "t.c"), "w") as f:
            f.write(source)
        exe = os.path.join(d, "t.exe")
        r = run([CC, "-std=gnu11", "-w", *extra, "t.c", "-o", exe], d)
        if r.returncode:
            raise AssertionError(f"C compile failed:\n{r.stderr}\n{source}")
        out = run([exe], d)
        if out.returncode:
            raise AssertionError(f"C program failed:\n{out.stderr}")
        return out.stdout


@unittest.skipUnless(CC, "no C compiler found")
class CompilerLayout(unittest.TestCase):
    def check(self, cls, extra=()):
        want_sizes, want_offsets = expected_layout(cls)
        got_sizes, got_offsets = parse_report(compile_and_run_c(c_program(cls), extra))
        self.assertEqual(got_sizes, want_sizes)
        self.assertEqual(got_offsets, want_offsets)

    def test_random_classes_without_bitfields(self):
        for seed in SEEDS:
            g = Generator(seed, bitfields=False, byteorder="little")
            cls = g.build_many(4)[-1].cls
            with self.subTest(seed=seed, cls=cls.__name__):
                self.check(cls)

    @unittest.skipUnless(MS_FLAG, "compiler cannot do MS bitfield layout")
    def test_random_classes_with_bitfields(self):
        for seed in SEEDS:
            g = Generator(seed, bitfields=True, byteorder="little")
            cls = g.build_many(4)[-1].cls
            with self.subTest(seed=seed, cls=cls.__name__):
                self.check(cls, MS_FLAG)

    @unittest.skipUnless(MS_FLAG, "compiler cannot do MS bitfield layout")
    def test_bitfield_corner_cases(self):
        @firstclass
        class Packed12:
            a: bitfield[uint16_t, 12]
            b: bitfield[uint16_t, 12]   # does not fit the first unit: new unit
            c: uint8_t

        @firstclass(align=True)
        class Aligned12:
            c0: uint8_t
            a: bitfield[uint32_t, 4]
            b: bitfield[uint32_t, 28]
            z: uint8_t

        @firstclass
        class Fill:
            a: bitfield[uint8_t, 8]
            b: bitfield[uint8_t, 1]
            c: bitfield[int8_t, 7]
            d: uint16_t
        for cls in (Packed12, Aligned12, Fill):
            with self.subTest(cls=cls.__name__):
                self.check(cls, MS_FLAG)

    def test_known_structures(self):
        class Kind(enum.IntEnum):
            A = 1
            B = 2

        @firstclass
        class Inner:
            x: int16_t
            y: int16_t

        @firstclass(align=True)
        class Mixed:
            tag: uint8_t
            n: uint32_t
            flag: bool_t
            big: uint64_t
            f: float32_t
            d: float64_t
            name: char_array[5]
            wide: wchar_array[3]
            raw: bytes_t[3]
            kind: enum_t[Kind, uint8_t]
            ptr: ctypes.c_void_p
            grid: array_t[array_t[int16_t, 3], 2]
            inner: Inner
            many: array_t[Inner, 3]
        self.check(Mixed)

    @unittest.skipUnless(MS_FLAG, "compiler cannot do MS bitfield layout")
    def test_bytes_match_what_the_compiler_stores(self):
        @firstclass
        class Frame:
            magic: uint32_t
            ready: bitfield[uint8_t, 1]
            error: bitfield[uint8_t, 1]
            prio: bitfield[uint8_t, 6]
            temp: int16_t
            name: char_array[4]
            ratio: float32_t
        source = "\n".join([
            "#include <stdio.h>", "#include <string.h>", Frame.c_header(), "int main(void) {",
            "  struct Frame f; memset(&f, 0, sizeof f);",
            "  f.magic = 0xFEEDFACEu; f.ready = 1; f.error = 0; f.prio = 42; f.temp = -5;",
            '  memcpy(f.name, "abc", 3); f.ratio = 1.5f;',
            "  const unsigned char *p = (const unsigned char *)&f;",
            '  for (size_t i = 0; i < sizeof f; i++) printf("%02x", p[i]);',
            '  printf("\\n"); return 0;', "}", ""])
        got = compile_and_run_c(source, MS_FLAG).strip()
        want = Frame(0xFEEDFACE, 1, 0, 42, -5, "abc", 1.5).to_bytes().hex()
        self.assertEqual(got, want)

    def test_header_import_matches_the_compiler(self):
        """from_header must see the same sizes the compiler does for real-looking headers."""
        from firstclass import from_header
        header = """
            #include <stdint.h>
            #include <stdbool.h>
            enum Mode { M_A, M_B = 7 };
            struct Inner { uint8_t a; uint32_t b; };
            #pragma pack(push, 1)
            struct Wire { uint8_t kind; uint32_t id; struct Inner in; char tag[3]; uint64_t stamp; };
            #pragma pack(pop)
            struct Native { uint8_t kind; double d; struct Inner in; enum Mode m; bool ok; uint16_t w[3]; };
        """
        types = from_header(text=header)
        source = "\n".join(["#include <stdio.h>", "#include <stddef.h>", header, "int main(void) {",
                            '  printf("Wire %zu\\n", sizeof(struct Wire));',
                            '  printf("Native %zu\\n", sizeof(struct Native));',
                            '  printf("Inner %zu\\n", sizeof(struct Inner));', "  return 0;", "}", ""])
        got = {k: int(v) for k, v in (line.split() for line in compile_and_run_c(source).splitlines())}
        self.assertEqual(got, {"Wire": types.Wire.sizeof(), "Native": types.Native.sizeof(),
                               "Inner": types.Inner.sizeof()})


def rust_program(classes):
    src = ["#![allow(warnings)]", "use std::mem::{size_of, offset_of};"]
    for i, cls in enumerate(classes):
        src.append(f"mod m{i} {{\n{cls.rust_source()}\n}}")
    src.append("fn main() {")
    for i, cls in enumerate(classes):
        sizes, offsets = expected_layout(cls, rust=True)
        for name in sizes:
            src.append(f'    println!("S {i}.{name} {{}}", size_of::<m{i}::{name}>());')
        for (name, field) in offsets:
            src.append(f'    println!("O {i}.{name}.{field} {{}}", offset_of!(m{i}::{name}, {field}));')
    src += ["}", ""]
    return "\n".join(src)


@unittest.skipUnless(RUSTC, "rustc not found")
class RustLayout(unittest.TestCase):
    def compile_and_run(self, source):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "t.rs"), "w") as f:
                f.write(source)
            exe = os.path.join(d, "t.exe")
            r = run([RUSTC, "--edition", "2021", "-A", "warnings", "t.rs", "-o", exe], d)
            if r.returncode:
                raise AssertionError(f"rustc failed:\n{r.stderr}\n{source}")
            return run([exe], d).stdout

    def check_many(self, classes):
        got_sizes, got_offsets = {}, {}
        for line in self.compile_and_run(rust_program(classes)).splitlines():
            kind, key, value = line.split()
            parts = key.split(".")
            if kind == "S":
                got_sizes[(int(parts[0]), parts[1])] = int(value)
            else:
                got_offsets[(int(parts[0]), parts[1], parts[2])] = int(value)
        want_sizes, want_offsets = {}, {}
        for i, cls in enumerate(classes):
            sizes, offsets = expected_layout(cls, rust=True)
            want_sizes.update({(i, n): v for n, v in sizes.items()})
            want_offsets.update({(i, n, f): v for (n, f), v in offsets.items()})
        self.assertEqual(got_sizes, want_sizes)
        self.assertEqual(got_offsets, want_offsets)

    def test_random_classes_in_one_compile(self):
        classes = []
        for seed in SEEDS:
            classes.append(Generator(seed, byteorder="little").build_many(4)[-1].cls)
        self.check_many(classes)

    def test_known_structures(self):
        class Kind(enum.IntEnum):
            A = 1
            B = 2

        @firstclass
        class Inner:
            x: int16_t

        @firstclass(align=True)
        class Aligned:
            a: uint8_t
            b: uint32_t
            c: uint8_t
            d: uint64_t
            e: uint16_t
            kind: enum_t[Kind, uint16_t]
            inner: Inner
            wide: wchar_array[3]

        @firstclass
        class Packed:
            a: uint8_t
            b: uint32_t
            name: char_array[3]
            bits_a: bitfield[uint8_t, 3]
            bits_b: bitfield[uint8_t, 5]
            ptr: ctypes.c_void_p
        self.check_many([Aligned, Packed])

    def test_bytes_match_what_rust_stores(self):
        @firstclass
        class Frame:
            magic: uint32_t
            temp: int16_t
            name: char_array[4]
            ratio: float32_t
            samples: array_t[uint16_t, 2]
        main = "\n".join([
            "fn main() {",
            '    let f = m0::Frame { magic: 0xFEEDFACE, temp: -5, name: *b"abc\\0", ratio: 1.5, samples: [1, 258] };',
            "    let bytes = unsafe { std::slice::from_raw_parts(&f as *const _ as *const u8, std::mem::size_of::<m0::Frame>()) };",
            '    for b in bytes { print!("{:02x}", b); }', '    println!();', "}", ""])
        source = f"#![allow(warnings)]\nmod m0 {{\n{Frame.rust_source()}\n}}\n{main}"
        got = self.compile_and_run(source).strip()
        self.assertEqual(got, Frame(0xFEEDFACE, -5, "abc", 1.5, [1, 258]).to_bytes().hex())

    def test_from_rust_matches_rustc_sizes(self):
        from firstclass import from_rust
        src = """
            #[repr(C)] pub struct Inner { pub a: u8, pub b: u32 }
            #[repr(C, packed)] pub struct Wire { pub kind: u8, pub id: u32, pub inner: Inner, pub stamp: u64 }
            #[repr(u8)] pub enum Mode { A = 1, B }
            #[repr(C)] pub struct Native { pub kind: u8, pub d: f64, pub inner: Inner, pub m: Mode, pub w: [u16; 3] }
        """
        types = from_rust(text=src)
        main = "fn main() {\n" + "".join(
            f'    println!("{n} {{}}", std::mem::size_of::<{n}>());\n' for n in ("Inner", "Wire", "Native")) + "}\n"
        got = {k: int(v) for k, v in (line.split() for line in self.compile_and_run(src + main).splitlines())}
        self.assertEqual(got, {n: types.structs[n].sizeof() for n in ("Inner", "Wire", "Native")})


if __name__ == "__main__":
    unittest.main()
