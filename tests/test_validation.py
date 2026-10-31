"""Values that cannot be encoded must be rejected when the instance is created or assigned,
never later at to_bytes() / conversion time."""
import copy
import ctypes
import dataclasses
import enum
import pickle
import unittest

from firstclass import (
    array_t, bitfield, bool_t, bytes_t, char_array, char_t, enum_t, firstclass, float32_t, float64_t, from_header,
    from_rust, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, wchar_array, wchar_t,
)


class Level(enum.IntEnum):
    LOW = 1
    HIGH = 5


@firstclass
class Inner:
    v: uint8_t


@firstclass
class Everything:
    u8: uint8_t
    i16: int16_t
    flag: bool_t
    f32: float32_t
    name: char_array[4]
    wide: wchar_array[3]
    blob: bytes_t[2]
    nums: array_t[uint8_t, 2]
    level: enum_t[Level, uint8_t]
    bits: bitfield[uint8_t, 3]
    inner: Inner
    ptr: ctypes.c_void_p


GOOD = dict(u8=1, i16=-2, flag=True, f32=1.5, name="abcd", wide="xyz", blob=b"\x00\x01", nums=[1, 2],
            level=Level.LOW, bits=7, inner=Inner(1), ptr=0)


def make(**override):
    return Everything(**{**GOOD, **override})


class CreationTime(unittest.TestCase):
    def test_good_baseline(self):
        make().to_bytes()

    def test_every_field_kind_rejects_bad_values_in_the_constructor(self):
        bad = {
            "u8": [256, -1, 1.5, "1", True, None],
            "i16": [40000, -40000, 1.0],
            "flag": [1, 0, "yes", None],
            "f32": [1e40, "1.0", True],
            "name": ["abcde", "ab\0", b"abc", 5],
            "wide": ["abcd", "a\0", b"ab", None],
            "blob": [b"\x00", b"\x00\x01\x02", "ab", [0, 1]],
            "nums": [[1], [1, 2, 3], [1, 256], "ab", 5],
            "level": [1, 5, Level, "LOW", None],
            "bits": [8, -1, True, 2.0],
            "inner": [Inner, {"v": 1}, 1, None],
            "ptr": [-1, 2 ** 64, "p", None],
        }
        for field, values in bad.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises((TypeError, ValueError)) as cm:
                        make(**{field: value})
                    self.assertIn(f"Everything.{field}", str(cm.exception))

    def test_error_types(self):
        with self.assertRaises(ValueError):
            make(u8=256)
        with self.assertRaises(TypeError):
            make(u8="1")

    def test_the_header_tag_example(self):
        t = from_header(text="struct Header { uint8_t kind; char tag; };")
        t.Header(1, "x")
        with self.assertRaises(ValueError):
            t.Header(1, "ab")  # two characters cannot fit a C char
        with self.assertRaises(ValueError):
            t.Header(1, "€")  # three UTF-8 bytes in a one-byte field

    def test_char_t_alone(self):
        @firstclass
        class C:
            c: char_t
        C("x")
        for bad in ("xy", "éé"):
            with self.assertRaises(ValueError):
                C(bad)

    def test_keyword_arguments_and_positional_both_checked(self):
        with self.assertRaises(ValueError):
            Inner(256)
        with self.assertRaises(ValueError):
            Inner(v=256)

    def test_missing_argument_is_still_a_normal_type_error(self):
        with self.assertRaises(TypeError):
            Inner()

    def test_first_bad_field_is_reported(self):
        with self.assertRaisesRegex(ValueError, "Everything.u8"):
            make(u8=999, i16=99999)

    def test_nested_errors_name_the_innermost_field(self):
        with self.assertRaisesRegex(ValueError, "Inner.v"):
            make(inner=Inner(300))

    def test_no_partial_object_escapes(self):
        created = []

        @firstclass
        class Tracker:
            a: uint8_t

            def __post_init__(self):
                created.append(self)
        with self.assertRaises(ValueError):
            Tracker(300)
        self.assertEqual(created, [])  # validation runs before the user's own __post_init__


class Assignment(unittest.TestCase):
    def test_assignment_is_checked_and_failed_assignment_changes_nothing(self):
        e = make()
        for field, bad in (("u8", 300), ("name", "toolong"), ("blob", b"\x00"), ("nums", [1]),
                           ("level", 5), ("inner", None), ("flag", 1), ("f32", 1e40), ("bits", 9)):
            with self.subTest(field):
                before = getattr(e, field)
                with self.assertRaises((TypeError, ValueError)):
                    setattr(e, field, bad)
                self.assertEqual(getattr(e, field), before)

    def test_valid_assignment_works_and_encodes(self):
        e = make()
        e.u8 = 200
        e.name = "hi"
        e.level = Level.HIGH
        self.assertEqual(Everything.from_bytes(e.to_bytes()), e)
        self.assertEqual(e.u8, 200)

    def test_non_field_attributes_are_free(self):
        e = make()
        e.anything = "goes"
        self.assertEqual(e.anything, "goes")

    def test_augmented_assignment(self):
        i = Inner(250)
        i.v += 5
        self.assertEqual(i.v, 255)
        with self.assertRaises(ValueError):
            i.v += 1
        self.assertEqual(i.v, 255)

    def test_setattr_builtin_and_dunder_dict_paths(self):
        e = make()
        with self.assertRaises(ValueError):
            setattr(e, "u8", -1)
        e.__dict__["u8"] = 999  # bypasses validation, but encoding still refuses it
        with self.assertRaises(ValueError):
            e.to_bytes()

    def test_in_place_mutation_of_containers_is_caught_at_encode_time(self):
        e = make()
        e.nums[0] = 300  # not an assignment to the field, so only the encoder can see it
        with self.assertRaises(ValueError):
            e.to_bytes()


class Normalisation(unittest.TestCase):
    def test_sequences_become_lists_and_bytes_like_become_bytes(self):
        e = make(nums=(1, 2), blob=bytearray(b"\x00\x01"))
        self.assertEqual(e.nums, [1, 2])
        self.assertIsInstance(e.nums, list)
        self.assertEqual(e.blob, b"\x00\x01")
        self.assertIsInstance(e.blob, bytes)
        e.blob = memoryview(b"\x02\x03")
        self.assertIsInstance(e.blob, bytes)

    def test_input_list_is_copied(self):
        nums = [1, 2]
        e = make(nums=nums)
        nums[0] = 99
        self.assertEqual(e.nums, [1, 2])

    def test_ints_stay_ints_for_float_fields(self):
        @firstclass
        class F:
            d: float64_t
        self.assertEqual(F(3).d, 3)


class Defaults(unittest.TestCase):
    def test_bad_default_fails_when_used(self):
        @firstclass
        class D:
            a: uint8_t
            name: char_array[2] = "toolong"
        D(1, "ok")
        with self.assertRaises(ValueError):
            D(1)

    def test_bad_default_factory_fails_when_used(self):
        @firstclass
        class D:
            nums: array_t[uint8_t, 2] = dataclasses.field(default_factory=lambda: [1, 2, 3])
        with self.assertRaises(ValueError):
            D()
        self.assertEqual(D([1, 2]).nums, [1, 2])

    def test_good_defaults(self):
        @firstclass
        class D:
            a: uint8_t = 7
            nums: array_t[uint8_t, 2] = dataclasses.field(default_factory=lambda: [1, 2])
        self.assertEqual(D(), D(7, [1, 2]))

    def test_init_false_fields_are_checked_on_assignment(self):
        @firstclass
        class N:
            a: uint8_t
            computed: uint8_t = dataclasses.field(default=0, init=False)
        n = N(1)
        with self.assertRaises(ValueError):
            n.computed = 300


class Frozen(unittest.TestCase):
    def setUp(self):
        @firstclass(frozen=True)
        class Fz:
            a: uint8_t
            name: char_array[2] = "ab"
            nums: array_t[uint8_t, 2] = (1, 2)
        self.Fz = Fz

    def test_constructor_validates(self):
        self.Fz(1)
        with self.assertRaises(ValueError):
            self.Fz(300)
        with self.assertRaises(ValueError):
            self.Fz(1, "toolong")
        with self.assertRaises(ValueError):
            self.Fz(1, "ab", (1,))

    def test_frozen_is_still_frozen_and_hashable_after_normalising(self):
        f = self.Fz(1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            f.a = 2
        self.assertEqual(f.nums, [1, 2])  # normalised to a list even though a tuple was given

    def test_user_post_init_runs_after_validation(self):
        seen = []

        @firstclass(frozen=True)
        class P:
            a: uint8_t

            def __post_init__(self):
                seen.append(self.a)
        P(3)
        with self.assertRaises(ValueError):
            P(300)
        self.assertEqual(seen, [3])

    def test_frozen_slots(self):
        @firstclass(frozen=True, slots=True)
        class S:
            a: uint8_t
        S(1)
        with self.assertRaises(ValueError):
            S(256)

    def test_frozen_roundtrips(self):
        f = self.Fz(5, "xy", [3, 4])
        self.assertEqual(self.Fz.from_bytes(f.to_bytes()), f)
        self.assertEqual(self.Fz.from_json(f.to_json()), f)


class OtherDataclassFeatures(unittest.TestCase):
    def test_replace_revalidates(self):
        i = Inner(1)
        self.assertEqual(dataclasses.replace(i, v=2).v, 2)
        with self.assertRaises(ValueError):
            dataclasses.replace(i, v=300)

    def test_copy_deepcopy_pickle_keep_values(self):
        e = make()
        for clone in (copy.copy(e), copy.deepcopy(e), pickle.loads(pickle.dumps(e))):
            self.assertEqual(clone, e)

    def test_slots_class(self):
        @firstclass(slots=True)
        class S:
            a: uint8_t
        s = S(1)
        with self.assertRaises(ValueError):
            s.a = 300
        with self.assertRaises(AttributeError):
            s.other = 1

    def test_subclass_checks_inherited_and_new_fields(self):
        @firstclass
        class Child(Inner):
            w: uint16_t
        Child(1, 2)
        with self.assertRaises(ValueError):
            Child(300, 2)
        with self.assertRaises(ValueError):
            Child(1, 70000)

    def test_undecorated_subclass_still_checks_parent_fields(self):
        class Plain(Inner):
            pass
        Plain(1)
        with self.assertRaises(ValueError):
            Plain(300)

    def test_plain_typed_fields_are_not_checked(self):
        @firstclass
        class Loose:
            a: uint8_t
            label: str = "x"
            anything: object = None
        Loose(1, 5, [1, 2])

    def test_ordering_and_equality_unaffected(self):
        self.assertTrue(Inner(1) < Inner(2))
        self.assertEqual(Inner(1), Inner(1))


class Everywhere(unittest.TestCase):
    """The same rule for classes that come from other languages and for every conversion path."""

    def test_from_header_classes(self):
        t = from_header(text="""
            enum class Kind : uint8_t { A = 1, B };
            struct S { uint8_t a; char tag; char name[4]; uint8_t raw[2]; Kind k; uint8_t bits : 3; };
        """)
        s = t.S(1, "x", "abcd", b"\x00\x01", t.Kind.A, 7)
        for bad in ((256, "x", "abcd", b"\x00\x01", t.Kind.A, 7), (1, "xy", "abcd", b"\x00\x01", t.Kind.A, 7),
                    (1, "x", "abcde", b"\x00\x01", t.Kind.A, 7), (1, "x", "abcd", b"\x00", t.Kind.A, 7),
                    (1, "x", "abcd", b"\x00\x01", 1, 7), (1, "x", "abcd", b"\x00\x01", t.Kind.A, 8)):
            with self.subTest(bad):
                with self.assertRaises((TypeError, ValueError)):
                    t.S(*bad)
        self.assertEqual(t.S.from_bytes(s.to_bytes()), s)

    def test_from_rust_classes(self):
        t = from_rust(text="#[repr(C, packed)] struct S { a: u8, tag: [c_char; 2], w: [wchar_t; 2] }")
        t.S(1, "ab", "xy")
        with self.assertRaises(ValueError):
            t.S(1, "abc", "xy")
        with self.assertRaises(ValueError):
            t.S(1, "ab", "xyz")

    def test_from_dict_json_xml_still_validate(self):
        with self.assertRaises(ValueError):
            Inner.from_dict({"v": 300})
        with self.assertRaises(ValueError):
            Inner.from_json('{"v": 300}')
        with self.assertRaises(ValueError):
            Inner.from_xml("<Inner><v>300</v></Inner>")

    def test_bytes_that_decode_to_invalid_values_are_rejected(self):
        @firstclass
        class E:
            level: enum_t[Level, uint8_t]
        with self.assertRaises(ValueError):
            E.from_bytes(bytes([7]))  # 7 is not a Level

    def test_conversions_never_see_an_invalid_object(self):
        e = make()
        for convert in (lambda x: x.to_bytes(), lambda x: x.to_dict(), lambda x: x.to_json(),
                        lambda x: x.to_xml(), lambda x: x.to_ctypes()):
            convert(e)  # all succeed because the object was valid from the start



if __name__ == "__main__":
    unittest.main()
