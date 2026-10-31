import dataclasses
import enum
import unittest

from firstclass import (
    array_t, bitfield, bytes_t, char_array, enum_t, firstclass, float64_t, from_header, uint8_t, uint16_t,
)


class Level(enum.IntEnum):
    LOW = 1
    HIGH = 5


@firstclass
class Version:
    major: uint8_t
    minor: uint8_t
    patch: uint16_t = 0


@firstclass
class Rich:
    name: char_array[4]
    level: enum_t[Level, uint8_t]
    items: array_t[uint8_t, 2]
    blob: bytes_t[2]
    ratio: float64_t
    bits: bitfield[uint8_t, 3]
    inner: Version


def rich(**kw):
    base = dict(name="a", level=Level.LOW, items=[1, 2], blob=b"\x00\x01", ratio=1.0, bits=1,
                inner=Version(1, 0))
    base.update(kw)
    return Rich(**base)


class Equality(unittest.TestCase):
    def test_eq_and_ne(self):
        self.assertEqual(Version(1, 2, 3), Version(1, 2, 3))
        self.assertNotEqual(Version(1, 2, 3), Version(1, 2, 4))
        self.assertTrue(Version(1, 2) == Version(1, 2, 0))
        self.assertFalse(Version(1, 2) != Version(1, 2))

    def test_equality_survives_a_bytes_roundtrip(self):
        r = rich()
        self.assertEqual(Rich.from_bytes(r.to_bytes()), r)

    def test_different_classes_are_never_equal(self):
        @firstclass
        class Other:
            major: uint8_t
            minor: uint8_t
            patch: uint16_t = 0
        self.assertNotEqual(Version(1, 2), Other(1, 2))
        self.assertFalse(Version(1, 2) == (1, 2, 0))

    def test_not_hashable_unless_frozen(self):
        with self.assertRaises(TypeError):
            hash(Version(1, 2))

        @firstclass(frozen=True)
        class Frozen:
            a: uint8_t
        self.assertEqual(hash(Frozen(1)), hash(Frozen(1)))
        self.assertEqual({Frozen(1), Frozen(1), Frozen(2)}, {Frozen(1), Frozen(2)})


class Ordering(unittest.TestCase):
    def test_all_four_operators(self):
        a, b = Version(1, 2, 3), Version(1, 3, 0)
        self.assertTrue(a < b)
        self.assertTrue(a <= b)
        self.assertTrue(b > a)
        self.assertTrue(b >= a)
        self.assertFalse(a > b)
        self.assertFalse(a >= b)
        self.assertFalse(b < a)
        self.assertFalse(b <= a)

    def test_equal_values(self):
        a, b = Version(1, 2, 3), Version(1, 2, 3)
        self.assertTrue(a <= b)
        self.assertTrue(a >= b)
        self.assertFalse(a < b)
        self.assertFalse(a > b)

    def test_lexicographic_in_field_order(self):
        self.assertTrue(Version(2, 0, 0) > Version(1, 255, 65535))
        self.assertTrue(Version(1, 1, 0) > Version(1, 0, 65535))
        self.assertTrue(Version(0, 0, 1) > Version(0, 0, 0))

    def test_sorted_min_max(self):
        vs = [Version(1, 2, 0), Version(0, 9, 9), Version(1, 0, 5), Version(1, 2, 1)]
        self.assertEqual(sorted(vs), [Version(0, 9, 9), Version(1, 0, 5), Version(1, 2, 0), Version(1, 2, 1)])
        self.assertEqual(max(vs), Version(1, 2, 1))
        self.assertEqual(min(vs), Version(0, 9, 9))
        self.assertEqual(sorted(vs, reverse=True)[0], Version(1, 2, 1))

    def test_ordering_matches_after_roundtrip(self):
        vs = [Version(3, 1), Version(1, 4), Version(2, 2)]
        back = [Version.from_bytes(v.to_bytes()) for v in vs]
        self.assertEqual(sorted(back), sorted(vs))

    def test_comparing_with_other_types_raises(self):
        with self.assertRaises(TypeError):
            Version(1, 2) < (1, 2, 0)
        with self.assertRaises(TypeError):
            Version(1, 2) >= 5

        @firstclass
        class Other:
            major: uint8_t
            minor: uint8_t
            patch: uint16_t = 0
        with self.assertRaises(TypeError):
            Version(1, 2) < Other(1, 3)

    def test_every_field_kind_participates(self):
        base = rich()
        self.assertTrue(rich(name="b") > base)
        self.assertTrue(rich(level=Level.HIGH) > base)
        self.assertTrue(rich(items=[1, 3]) > base)
        self.assertTrue(rich(blob=b"\x00\x02") > base)
        self.assertTrue(rich(ratio=1.5) > base)
        self.assertTrue(rich(bits=2) > base)
        self.assertTrue(rich(inner=Version(1, 1)) > base)
        self.assertTrue(base < rich(inner=Version(2, 0)))

    def test_first_differing_field_decides(self):
        self.assertTrue(rich(name="b", ratio=0.0) > rich(name="a", ratio=99.0))

    def test_nested_classes_are_orderable(self):
        @firstclass
        class Pair:
            lo: Version
            hi: Version
        self.assertTrue(Pair(Version(1, 0), Version(9, 9)) < Pair(Version(1, 1), Version(0, 0)))

    def test_works_for_every_byte_order_and_alignment(self):
        for kw in ({}, {"byteorder": "big"}, {"align": True}):
            with self.subTest(kw):
                @firstclass(**kw)
                class V:
                    a: uint8_t
                    b: uint16_t
                self.assertTrue(V(1, 2) < V(1, 3))
                self.assertTrue(V(2, 0) > V(1, 9))


class Options(unittest.TestCase):
    def test_order_false_disables_ordering_only(self):
        @firstclass(order=False)
        class NoOrder:
            a: uint8_t
        self.assertEqual(NoOrder(1), NoOrder(1))
        with self.assertRaises(TypeError):
            NoOrder(1) < NoOrder(2)

    def test_eq_false_means_no_ordering_either(self):
        @firstclass(eq=False)
        class NoEq:
            a: uint8_t
        x, y = NoEq(1), NoEq(1)
        self.assertNotEqual(x, y)  # identity comparison
        self.assertEqual(x, x)
        with self.assertRaises(TypeError):
            x < y

    def test_eq_false_with_explicit_order_is_a_dataclass_error(self):
        with self.assertRaises(ValueError):
            @firstclass(eq=False, order=True)
            class Bad:
                a: uint8_t

    def test_order_true_explicit_is_fine(self):
        @firstclass(order=True)
        class Explicit:
            a: uint8_t
        self.assertTrue(Explicit(1) < Explicit(2))

    def test_field_compare_false_is_ignored(self):
        @firstclass
        class Skip:
            a: uint8_t
            noise: uint8_t = dataclasses.field(default=0, compare=False)
        self.assertEqual(Skip(1, 5), Skip(1, 9))
        self.assertFalse(Skip(1, 9) < Skip(1, 5))
        self.assertTrue(Skip(1, 0) < Skip(2, 0))

    def test_functools_total_ordering_not_needed(self):
        for name in ("__lt__", "__le__", "__gt__", "__ge__", "__eq__"):
            self.assertIn(name, vars(Version))


class HeaderGenerated(unittest.TestCase):
    def test_generated_classes_compare(self):
        t = from_header(text="""
            enum class Lvl : uint8_t { A = 1, B };
            struct Ver { uint8_t major; uint8_t minor; };
            struct Msg { struct Ver ver; Lvl lvl; char tag[4]; };
        """)
        a = t.Msg(t.Ver(1, 0), t.Lvl.A, "x")
        b = t.Msg(t.Ver(1, 0), t.Lvl.B, "a")
        self.assertTrue(a < b)
        self.assertTrue(t.Ver(1, 9) < t.Ver(2, 0))
        self.assertEqual(sorted([b, a]), [a, b])
        self.assertEqual(t.Msg.from_bytes(a.to_bytes()), a)

    def test_generated_classes_from_two_parses_are_different_types(self):
        src = "struct Ver { uint8_t major; };"
        a, b = from_header(text=src), from_header(text=src)
        self.assertNotEqual(a.Ver(1), b.Ver(1))
        with self.assertRaises(TypeError):
            a.Ver(1) < b.Ver(2)


if __name__ == "__main__":
    unittest.main()
