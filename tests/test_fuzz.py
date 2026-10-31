"""Randomised end-to-end checks: every conversion path must agree for any generated class."""
import os
import random
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from gen import Generator  # noqa: E402

from firstclass import _specs as S  # noqa: E402
from firstclass import from_header, from_rust  # noqa: E402

SEEDS = range(80)
INSTANCES = 6


def last_built(seed, **kw):
    g = Generator(seed, **kw)
    built = g.build_many(4)
    return g, built


class RoundTrips(unittest.TestCase):
    def test_every_format_roundtrips(self):
        for seed in SEEDS:
            g, builds = last_built(seed)
            rng = random.Random(seed * 7919)
            for b in builds:
                cls = b.cls
                for _ in range(INSTANCES):
                    inst = b.make(rng)
                    with self.subTest(seed=seed, cls=cls.__name__):
                        raw = inst.to_bytes()
                        self.assertEqual(len(raw), cls.sizeof())
                        self.assertEqual(cls.from_bytes(raw), inst)
                        self.assertEqual(cls.unpack_from(b"\xaa\xbb" + raw + b"\xcc", 2), inst)
                        self.assertEqual(cls.from_dict(inst.to_dict()), inst)
                        self.assertEqual(cls.from_json(inst.to_json()), inst)
                        self.assertEqual(cls.from_xml(inst.to_xml()), inst)
                        self.assertEqual(cls.from_xml(inst.to_xml(indent=2)), inst)

    def test_ctypes_agrees_with_the_bytes(self):
        for seed in SEEDS:
            g, builds = last_built(seed)
            rng = random.Random(seed)
            for b in builds:
                if not b.ctypes_parity:
                    continue
                for _ in range(INSTANCES):
                    inst = b.make(rng)
                    with self.subTest(seed=seed, cls=b.cls.__name__):
                        c = inst.to_ctypes()
                        self.assertEqual(bytes(c), inst.to_bytes())
                        self.assertEqual(b.cls.from_ctypes(c), inst)


FMT = {"uint8_t": "B", "uint16_t": "H", "uint32_t": "I", "uint64_t": "Q", "int8_t": "b", "int16_t": "h",
       "int32_t": "i", "int64_t": "q", "float32_t": "f", "float64_t": "d", "bool_t": "?"}


def oracle(buf, offset, spec, value, bo):
    """Re-encode one value with the struct module only (an independent second implementation)."""
    if isinstance(spec, S.EnumSpec):
        struct.pack_into(bo + FMT[spec.storage.name], buf, offset, int(value))
    elif isinstance(spec, S.Pointer):
        struct.pack_into(bo + ("Q" if S._PTR_SIZE == 8 else "I"), buf, offset, value)
    elif isinstance(spec, S.Scalar):
        struct.pack_into(bo + FMT[spec.name], buf, offset, value)
    elif isinstance(spec, S.CharArray):
        buf[offset:offset + spec.n] = value.encode("utf-8").ljust(spec.n, bytes(1))
    elif isinstance(spec, S.WCharArray):
        raw = value.encode("utf-16-be" if bo == ">" else "utf-16-le")
        buf[offset:offset + 2 * spec.n] = raw.ljust(2 * spec.n, bytes(1))
    elif isinstance(spec, S.ByteArray):
        buf[offset:offset + spec.n] = value
    elif isinstance(spec, S.ArrayOf):
        for i, item in enumerate(value):
            oracle(buf, offset + i * spec.elem.size(), spec.elem, item, bo)
    elif isinstance(spec, S.Nested):
        buf[offset:offset + spec.size()] = oracle_object(value)
    else:
        raise AssertionError(f"oracle cannot encode {spec!r}")


def oracle_object(inst):
    schema = type(inst).__firstclass__
    items, total, _ = schema.layout()
    buf = bytearray(total)
    for item in items:
        _, index, offset = item
        field, spec = schema.fields[index]
        oracle(buf, offset, spec, getattr(inst, field.name), schema.bo)
    return bytes(buf)


class Oracle(unittest.TestCase):
    def test_struct_module_agrees_on_every_value(self):
        checked = 0
        for seed in SEEDS:
            g, builds = last_built(seed, bitfields=False)  # bitfield groups have no struct-module equivalent
            rng = random.Random(seed * 13)
            for b in builds:
                for _ in range(INSTANCES):
                    inst = b.make(rng)
                    with self.subTest(seed=seed, cls=b.cls.__name__):
                        self.assertEqual(inst.to_bytes(), oracle_object(inst))
                        checked += 1
        self.assertGreater(checked, 1000)


class CrossLanguage(unittest.TestCase):
    def check_foreign(self, make_types, label):
        for seed in SEEDS:
            g, builds = last_built(seed)
            rng = random.Random(seed + 1)
            for b in builds:
                types = make_types(b, g.byteorder)
                with self.subTest(seed=seed, cls=b.cls.__name__, via=label):
                    self.assertEqual(types.skipped, {})
                    foreign = types.structs[b.cls.__name__]
                    self.assertEqual(foreign.sizeof(), b.cls.sizeof())
                    for _ in range(INSTANCES):
                        raw = b.make(rng).to_bytes()
                        self.assertEqual(foreign.from_bytes(raw).to_bytes(), raw)

    def test_c_header_reimport_is_byte_compatible(self):
        self.check_foreign(lambda b, bo: from_header(text=b.cls.c_header(), byteorder=bo), "c_header")

    def test_rust_source_reimport_is_byte_compatible(self):
        self.check_foreign(lambda b, bo: from_rust(text=b.cls.rust_source(), byteorder=bo), "rust")

    def test_c_and_rust_agree(self):
        for seed in SEEDS:
            g, builds = last_built(seed)
            for b in builds:
                with self.subTest(seed=seed, cls=b.cls.__name__):
                    c = from_header(text=b.cls.c_header(), byteorder=g.byteorder)
                    rs = from_rust(text=b.cls.rust_source(), byteorder=g.byteorder)
                    self.assertEqual(c.structs[b.cls.__name__].sizeof(), rs.structs[b.cls.__name__].sizeof())


class Decoder(unittest.TestCase):
    def test_arbitrary_bytes_decode_or_raise_value_error_and_are_stable(self):
        for seed in SEEDS:
            g, builds = last_built(seed)
            rng = random.Random(seed * 31)
            for b in builds:
                cls = b.cls
                for _ in range(30):
                    raw = rng.randbytes(cls.sizeof())
                    with self.subTest(seed=seed, cls=cls.__name__, raw=raw.hex()):
                        try:
                            first = cls.from_bytes(raw)
                        except ValueError:
                            continue  # e.g. a byte pattern that is not a valid enum value or UTF-8
                        again = first.to_bytes()  # compare bytes: NaN != NaN
                        self.assertEqual(cls.from_bytes(again).to_bytes(), again)

    def test_truncated_and_oversized_input_is_a_value_error(self):
        for seed in SEEDS:
            g, builds = last_built(seed)
            for b in builds:
                size = b.cls.sizeof()
                for bad in (bytes(max(size - 1, 0)) if size else b"x", bytes(size + 1)):
                    with self.subTest(seed=seed, cls=b.cls.__name__, size=len(bad)):
                        with self.assertRaises(ValueError):
                            b.cls.from_bytes(bad)


class Ordering(unittest.TestCase):
    def test_comparisons_never_raise_for_generated_classes(self):
        for seed in SEEDS:
            g, builds = last_built(seed)
            rng = random.Random(seed)
            for b in builds:
                x, y = b.make(rng), b.make(rng)
                with self.subTest(seed=seed, cls=b.cls.__name__):
                    self.assertTrue(x <= x and x >= x and not x < x and not x > x)
                    self.assertEqual(x < y, not x >= y)
                    self.assertEqual(sorted([x, y])[0] <= sorted([x, y])[1], True)


class Mutation(unittest.TestCase):
    def test_assigning_a_wrong_type_is_always_rejected(self):
        for seed in SEEDS:
            g, builds = last_built(seed)
            rng = random.Random(seed)
            for b in builds:
                inst = b.make(rng)
                for name in list(b.cls.__dataclass_fields__):
                    before = getattr(inst, name)
                    with self.subTest(seed=seed, cls=b.cls.__name__, field=name):
                        with self.assertRaises((TypeError, ValueError)):
                            setattr(inst, name, object())
                        self.assertEqual(getattr(inst, name), before)


if __name__ == "__main__":
    unittest.main()
