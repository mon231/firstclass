"""Tour of firstclass.

Part 1: define a class in Python and get bytes / JSON / XML / ctypes / a C header.
Part 2: import a real C/C++ header (enums, enum classes, bitfields, packed structs) and use the
        generated classes the same way.

Run from the repo root:  python examples/example.py
"""
import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from firstclass import (  # noqa: E402
    array_t, bool_t, bytes_t, char_array, firstclass, float32_t, from_header, int16_t, uint8_t, uint16_t,
    uint32_t,
)

HERE = Path(__file__).resolve().parent


# --- part 1: define in Python -------------------------------------------------

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


def define_in_python():
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

    # bytes: packed, little-endian, same as a C `#pragma pack(1)` struct
    raw = t.to_bytes()
    print(f"{Telemetry.sizeof()} bytes: {raw.hex(' ')}")
    assert Telemetry.from_bytes(raw) == t
    assert Telemetry.unpack_from(b"junk" + raw + b"more", 4) == t   # read at an offset in a stream

    # JSON / XML
    js = t.to_json(indent=2)
    print(js)
    assert Telemetry.from_json(js) == t
    xml = t.to_xml(indent=2)
    print(xml)
    assert Telemetry.from_xml(xml) == t

    # ctypes: same bytes, usable with C libraries / shared memory
    c = t.to_ctypes()
    assert bytes(c) == raw
    assert Telemetry.from_ctypes(c) == t

    # the C header for the other side of the wire
    print(Telemetry.c_header())

    # errors are loud, never silent truncation
    t.device = "way-too-long-name"
    try:
        t.to_bytes()
    except ValueError as e:
        print("rejected:", e)


# --- part 2: import a C header ------------------------------------------------

def import_c_header():
    types = from_header(HERE / "protocol.h")        # or firstclass.from_header(...)

    print("\nparsed structs:", ", ".join(types.structs))
    print("parsed enums:  ", ", ".join(types.enums))
    print("constants:      MAX_NAME =", types.constants["MAX_NAME"], " CMD_DATA =", types.constants["CMD_DATA"])
    print("skipped:       ", types.skipped)              # the union could not be converted

    # enums come back as IntEnum; `enum class A : uint16_t` keeps its 2-byte size on the wire
    assert types.Command.CMD_DATA == 0x10
    assert types.Status.Failed.value == 0x80
    assert types.Channel.Aux == 1000

    # the packed structs (#pragma pack(push, 1)) have no padding ...
    print(f"\nHeader  {types.Header.sizeof():2d} bytes (packed: 4 + 1 + 1 bitfield byte + 4 enum + 1 enum class)")
    print(f"Reading {types.Reading.sizeof():2d} bytes (uint16_t enum class + int16_t)")
    print(f"Message {types.Message.sizeof():2d} bytes")
    # ... while the struct outside the pragma uses normal C alignment
    print(f"InMemory {types.InMemory.sizeof():2d} bytes (padded, like a C compiler would)")
    assert types.Header.sizeof() == 11 and types.Message.sizeof() == 45 and types.InMemory.sizeof() == 24

    msg = types.Message(
        header=types.Header(
            magic=0xFEEDFACE, version=1,
            ready=1, error=0, priority=42,                  # bitfields are plain ints
            command=types.Command.CMD_DATA, status=types.Status.Busy,
        ),
        sender="sensor-1",
        mac=bytes.fromhex("deadbeef0001"),
        readings=[types.Reading(types.Channel.A, 250), types.Reading(types.Channel.B, -3),
                  types.Reading(types.Channel.Aux, 0), types.Reading(types.Channel.A, 7)],
        scale=0.5,
    )

    raw = msg.to_bytes()
    print("\nmessage on the wire:", raw.hex(" "))
    back = types.Message.from_bytes(raw)
    assert back == msg
    assert back.header.status is types.Status.Busy              # enums round-trip as members
    assert back.readings[1].channel is types.Channel.B

    # the generated classes are ordinary firstclass classes: JSON, XML, ctypes, C header ...
    assert types.Message.from_json(msg.to_json()) == msg
    assert types.Message.from_xml(msg.to_xml()) == msg
    assert bytes(msg.to_ctypes()) == raw
    print("\nback to C:\n" + types.Header.c_header())

    # unknown enum values and out-of-range fields are rejected, not guessed
    bad = bytearray(raw)
    bad[10] = 0x55                                              # Header.status is not a valid Status
    try:
        types.Message.from_bytes(bytes(bad))
    except ValueError as e:
        print("rejected:", e)

    # the same header, but for a big-endian peer
    be = from_header(HERE / "protocol.h", byteorder="big")
    assert be.Reading(be.Channel.Aux, 1).to_bytes() == bytes.fromhex("03e8 0001")


def main():
    define_in_python()
    import_c_header()


if __name__ == "__main__":
    main()
