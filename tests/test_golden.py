"""Snapshot tests: exact bytes and generated text. A change here is a deliberate format change."""
import enum
import unittest

from firstclass import (
    array_t, bitfield, bool_t, bytes_t, char_array, enum_t, firstclass, float32_t, int16_t, uint8_t, uint16_t,
    uint32_t, uint64_t, wchar_array,
)


class Level(enum.IntEnum):
    LOW = 1
    HIGH = 300


def define(**kw):
    @firstclass(**kw)
    class Point:
        x: int16_t
        y: int16_t

    @firstclass(**kw)
    class Golden:
        magic: uint32_t
        ok: bool_t
        level: enum_t[Level, uint16_t]
        ready: bitfield[uint8_t, 1]
        prio: bitfield[uint8_t, 7]
        temp: float32_t
        name: char_array[6]
        wide: wchar_array[3]
        mac: bytes_t[4]
        grid: array_t[uint16_t, 3]
        origin: Point
        path: array_t[Point, 2]
    return Point, Golden


def sample(Point, Golden):
    return Golden(0xFEEDFACE, True, Level.HIGH, 1, 42, 1.5, "héllo", "x€", b"\xde\xad\xbe\xef", [1, 2, 3],
                  Point(-1, 2), [Point(3, 4), Point(5, -6)])


LITTLE = ("cefaedfe" "01" "2c01" "55" "0000c03f" "68c3a96c6c6f" "7800ac200000" "deadbeef" "010002000300"
          "ffff0200" "03000400" "0500faff")
BIG = ("feedface" "01" "012c" "aa" "3fc00000" "68c3a96c6c6f" "007820ac0000" "deadbeef" "000100020003"
       "ffff0002" "00030004" "0005fffa")

JSON = ('{"magic": 4277009102, "ok": true, "level": 300, "ready": 1, "prio": 42, "temp": 1.5, '
        '"name": "h\\u00e9llo", "wide": "x\\u20ac", "mac": "deadbeef", "grid": [1, 2, 3], '
        '"origin": {"x": -1, "y": 2}, "path": [{"x": 3, "y": 4}, {"x": 5, "y": -6}]}')

XML = ("<Golden><magic>4277009102</magic><ok>true</ok><level>300</level><ready>1</ready><prio>42</prio>"
       "<temp>1.5</temp><name>héllo</name><wide>x€</wide><mac>deadbeef</mac>"
       "<grid><item>1</item><item>2</item><item>3</item></grid><origin><x>-1</x><y>2</y></origin>"
       "<path><item><x>3</x><y>4</y></item><item><x>5</x><y>-6</y></item></path></Golden>")

C_HEADER = """#include <stdint.h>
#include <stdbool.h>

/* byteorder: little */
#pragma pack(push, 1)
struct Point {
    int16_t x;
    int16_t y;
};
#pragma pack(pop)

enum Level {
    Level_LOW = 1,
    Level_HIGH = 300,
};

/* byteorder: little */
#pragma pack(push, 1)
struct Golden {
    uint32_t magic;
    bool ok;
    uint16_t level;
    uint8_t ready : 1;
    uint8_t prio : 7;
    float temp;
    char name[6];
    uint16_t wide[3];
    uint8_t mac[4];
    uint16_t grid[3];
    struct Point origin;
    struct Point path[2];
};
#pragma pack(pop)
"""

RUST = """// byteorder: little
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct Point {
    pub x: i16,
    pub y: i16,
}

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    LOW = 1,
    HIGH = 300,
}

// byteorder: little
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct Golden {
    pub magic: u32,
    pub ok: bool,
    pub level: Level,
    pub bits_0: u8, // ready: 1 bits, prio: 7 bits
    pub temp: f32,
    pub name: [u8; 6],
    pub wide: [u16; 3],
    pub mac: [u8; 4],
    pub grid: [u16; 3],
    pub origin: Point,
    pub path: [Point; 2],
}
"""


class Golden(unittest.TestCase):
    def setUp(self):
        self.Point, self.Golden = define()
        self.value = sample(self.Point, self.Golden)

    def test_little_endian_bytes(self):
        self.assertEqual(self.value.to_bytes().hex(), LITTLE)
        self.assertEqual(self.Golden.sizeof(), 46)
        self.assertEqual(self.Golden.from_bytes(bytes.fromhex(LITTLE)), self.value)

    def test_big_endian_bytes(self):
        point, golden = define(byteorder="big")
        value = sample(point, golden)
        self.assertEqual(value.to_bytes().hex(), BIG)
        self.assertEqual(golden.from_bytes(bytes.fromhex(BIG)), value)

    def test_json(self):
        self.assertEqual(self.value.to_json(), JSON)

    def test_xml(self):
        self.assertEqual(self.value.to_xml(), XML)

    def test_c_header(self):
        self.assertEqual(self.Golden.c_header(), C_HEADER)

    def test_rust_source(self):
        self.assertEqual(self.Golden.rust_source(), RUST)

    def test_aligned_layout_bytes(self):
        @firstclass(align=True)
        class A:
            a: uint8_t
            b: uint32_t
            c: uint16_t
            d: uint64_t
        self.assertEqual(A(1, 2, 3, 4).to_bytes().hex(),
                         "01000000" "02000000" "0300000000000000" "0400000000000000")
        self.assertEqual(A.sizeof(), 24)

    def test_the_documented_readme_numbers(self):
        """The sizes quoted in the README example."""
        from firstclass import float32_t as f32

        @firstclass
        class P:
            x: int16_t

        @firstclass
        class Packet:
            kind: uint8_t
            ok: bool_t
            temp: f32
            name: char_array[8]
            mac: bytes_t[6]
            samples: array_t[uint16_t, 4]
            origin: P
        self.assertEqual(Packet.sizeof(), 1 + 1 + 4 + 8 + 6 + 8 + 2)


if __name__ == "__main__":
    unittest.main()
