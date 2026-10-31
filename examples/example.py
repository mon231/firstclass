"""Tour of firstclass: one definition, many formats.

Run from the repo root:  python examples/example.py
"""
import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from firstclass import (  # noqa: E402
    array_t, bool_t, bytes_t, char_array, firstclass, float32_t, int16_t, uint8_t, uint16_t, uint32_t,
)


@firstclass
class Vec2:
    x: int16_t
    y: int16_t


@firstclass
class Header:
    magic: uint32_t
    version: uint8_t
    flags: bool_t


@firstclass
class Telemetry:
    header: Header
    device: char_array[8]            # NUL-padded UTF-8
    mac: bytes_t[6]
    readings: array_t[uint16_t, 4]
    temp: float32_t
    path: array_t[Vec2, 2]           # array of nested structs
    seq: ctypes.c_uint32             # raw ctypes types work too
    buf_addr: ctypes.c_void_p        # pointer = address as an int (0 = NULL)


def main():
    t = Telemetry(
        header=Header(0xC0FFEE, 2, True),
        device="probe-7",
        mac=bytes.fromhex("001122334455"),
        readings=[10, 20, 30, 40],
        temp=21.5,
        path=[Vec2(0, 0), Vec2(-5, 12)],
        seq=42,
        buf_addr=0,
    )

    # 1. bytes: packed, little-endian, same as a C `#pragma pack(1)` struct
    raw = t.to_bytes()
    print(f"{Telemetry.sizeof()} bytes: {raw.hex(' ')}")
    assert Telemetry.from_bytes(raw) == t

    # 2. JSON
    js = t.to_json(indent=2)
    print(js)
    assert Telemetry.from_json(js) == t

    # 3. XML
    xml = t.to_xml(indent=2)
    print(xml)
    assert Telemetry.from_xml(xml) == t

    # 4. ctypes: same bytes, usable with C libraries / shared memory
    c = t.to_ctypes()
    assert bytes(c) == raw
    assert Telemetry.from_ctypes(c) == t

    # 5. C header for the other side of the wire
    print(Telemetry.c_header())

    # 6. errors are loud, never silent truncation
    t.device = "way-too-long-name"
    try:
        t.to_bytes()
    except ValueError as e:
        print("rejected:", e)


if __name__ == "__main__":
    main()
