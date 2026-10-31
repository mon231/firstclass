"""Large inputs finish quickly and correctly."""
import faulthandler
import time
import unittest

from firstclass import (
    array_t, bytes_t, char_array, firstclass, from_header, from_rust, uint8_t, uint16_t, uint32_t,
)


class Scale(unittest.TestCase):
    def setUp(self):
        faulthandler.dump_traceback_later(240, exit=True)

    def tearDown(self):
        faulthandler.cancel_dump_traceback_later()

    def timed(self, seconds, fn):
        start = time.perf_counter()
        result = fn()
        self.assertLess(time.perf_counter() - start, seconds)
        return result

    def test_many_fields(self):
        n = 600
        ann = {f"f{i}": uint16_t for i in range(n)}
        cls = firstclass(type("Wide", (), {"__annotations__": ann}))
        value = cls(**{f"f{i}": i for i in range(n)})
        raw = self.timed(5, value.to_bytes)
        self.assertEqual(len(raw), 2 * n)
        self.assertEqual(self.timed(5, lambda: cls.from_bytes(raw)), value)
        self.assertEqual(cls.from_json(value.to_json()), value)
        self.assertEqual(cls.from_xml(value.to_xml()), value)
        self.assertEqual(cls.from_ctypes(value.to_ctypes()), value)

    def test_big_arrays(self):
        @firstclass
        class Big:
            data: array_t[uint8_t, 100_000]
            words: array_t[uint32_t, 20_000]
            blob: bytes_t[500_000]
        value = Big(list(range(256)) * 390 + [0] * 160, list(range(20_000)), bytes(range(256)) * 1953 + bytes(32))
        raw = self.timed(10, value.to_bytes)
        self.assertEqual(len(raw), 100_000 + 80_000 + 500_000)
        self.assertEqual(self.timed(10, lambda: Big.from_bytes(raw)), value)

    def test_deep_nesting(self):
        cls = firstclass(type("L0", (), {"__annotations__": {"v": uint8_t}}))
        value = cls(7)
        for depth in range(1, 40):
            cls = firstclass(type(f"L{depth}", (), {"__annotations__": {"inner": cls, "tag": uint8_t}}))
            value = cls(value, depth)
        raw = value.to_bytes()
        self.assertEqual(len(raw), 40)
        self.assertEqual(cls.from_bytes(raw), value)
        self.assertEqual(cls.from_json(value.to_json()), value)
        self.assertEqual(cls.from_xml(value.to_xml()), value)

    def test_large_c_header(self):
        count = 2000
        text = "\n".join(f"struct S{i} {{ uint8_t a; uint32_t b; char name[{i % 16 + 1}]; }};" for i in range(count))
        t = self.timed(30, lambda: from_header(text=text))
        self.assertEqual(len(t.structs), count)
        self.assertEqual(t.skipped, {})

    def test_large_c_header_with_many_macros_and_enums(self):
        lines = [f"#define C{i} {i}" for i in range(3000)]
        lines.append("enum Big { " + ", ".join(f"E{i} = C{i} * 2" for i in range(3000)) + " };")
        lines.append("struct S { uint8_t a[C100]; enum Big b; };")
        t = self.timed(30, lambda: from_header(text="\n".join(lines)))
        self.assertEqual(t.S.sizeof(), 104)  # uint8_t[100] already 4-aligned, then the 4-byte enum
        self.assertEqual(t.constants["E2999"], 5998)

    def test_large_rust_source(self):
        count = 1500
        text = "\n".join(f"#[repr(C)] pub struct S{i} {{ pub a: u8, pub b: u32, pub n: [u8; {i % 16 + 1}] }}"
                         for i in range(count))
        t = self.timed(30, lambda: from_rust(text=text))
        self.assertEqual(len(t.structs), count)

    def test_long_single_line_inputs(self):
        t = self.timed(10, lambda: from_header(text="struct S { " + "uint8_t a; " * 1 + " " * 500_000 + "};"))
        self.assertEqual(t.S.sizeof(), 1)

    def test_many_instances(self):
        @firstclass
        class P:
            a: uint8_t
            b: uint16_t
            name: char_array[4]
        start = time.perf_counter()
        items = [P(i % 256, i % 65536, "ab") for i in range(20_000)]
        blob = b"".join(p.to_bytes() for p in items)
        back = [P.unpack_from(blob, i * P.sizeof()) for i in range(20_000)]
        self.assertEqual(back, items)
        self.assertLess(time.perf_counter() - start, 30)


if __name__ == "__main__":
    unittest.main()
