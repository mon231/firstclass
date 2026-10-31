"""Random firstclass class generator shared by the fuzz and compiler-parity tests."""
import ctypes
import enum
import math
import random
import struct
import sys

from firstclass import (
    array_t, bitfield, bool_t, bytes_t, char_array, enum_t, firstclass, float32_t, float64_t, int8_t, int16_t,
    int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, wchar_array,
)

INT_TYPES = [
    (uint8_t, 0, 2 ** 8 - 1, 8), (uint16_t, 0, 2 ** 16 - 1, 16), (uint32_t, 0, 2 ** 32 - 1, 32),
    (uint64_t, 0, 2 ** 64 - 1, 64), (int8_t, -2 ** 7, 2 ** 7 - 1, 8), (int16_t, -2 ** 15, 2 ** 15 - 1, 16),
    (int32_t, -2 ** 31, 2 ** 31 - 1, 32), (int64_t, -2 ** 63, 2 ** 63 - 1, 64),
]
CHARS = ["a", "Z", "0", " ", "_", "é", "€"]       # 1, 1, 1, 1, 1, 2 and 3 UTF-8 bytes
WIDE = ["a", "é", "€", "\U0001F600"]               # 1, 1, 1 and 2 UTF-16 units
PTR_BYTES = ctypes.sizeof(ctypes.c_void_p)
NATIVE = sys.byteorder


class Built:
    def __init__(self, cls, make, pointers, bitfields, aligned, byteorder):
        self.cls, self.make, self.pointers = cls, make, pointers
        self.bitfields, self.aligned, self.byteorder = bitfields, aligned, byteorder

    @property
    def ctypes_parity(self):
        """ctypes lays out bitfields after small members differently from us in aligned structs."""
        return not (self.bitfields and self.aligned) and not (self.pointers and self.byteorder != NATIVE)


class Generator:
    def __init__(self, seed, *, bitfields=True, pointers=True, byteorder=None):
        self.r = random.Random(seed)
        self.counter = 0
        self.built: list[Built] = []
        self.allow_bitfields, self.allow_pointers = bitfields, pointers
        self.byteorder = byteorder or self.r.choice(["little", "big"])

    def name(self, prefix):
        self.counter += 1
        return f"{prefix}{self.counter}"

    # --- single fields: each returns (annotation, maker(rng) -> value, flags) ---------------------

    def field(self, depth=0):
        r = self.r
        kinds = ["int", "int", "int", "f64", "f32", "bool", "char", "wchar", "bytes", "enum"]
        if self.allow_pointers:
            kinds.append("ptr")
        if depth < 2:
            kinds += ["array", "array"]
            if self.built:
                kinds += ["nested", "nested"]
        kind = r.choice(kinds)
        flags = {"pointers": False, "bitfields": False}
        if kind == "int":
            tp, lo, hi, _ = r.choice(INT_TYPES)
            return tp, lambda g: g.choice([lo, hi, g.randint(lo, hi), g.randint(lo, hi)]), flags
        if kind == "f64":
            def make(g):
                v = struct.unpack("<d", g.randbytes(8))[0]
                return v if math.isfinite(v) else 1.5
            return float64_t, make, flags
        if kind == "f32":
            def make(g):
                v = struct.unpack("<f", g.randbytes(4))[0]
                return v if math.isfinite(v) else 0.25
            return float32_t, make, flags
        if kind == "bool":
            return bool_t, lambda g: g.random() < 0.5, flags
        if kind == "char":
            n = r.randint(1, 9)

            def make(g):
                out, size = [], 0
                for _ in range(g.randint(0, n)):
                    c = g.choice(CHARS)
                    if size + len(c.encode()) > n:
                        break
                    out.append(c)
                    size += len(c.encode())
                return "".join(out)
            return char_array[n], make, flags
        if kind == "wchar":
            n = r.randint(1, 6)

            def make(g):
                out, units = [], 0
                for _ in range(g.randint(0, n)):
                    c = g.choice(WIDE)
                    if units + len(c.encode("utf-16-le")) // 2 > n:
                        break
                    out.append(c)
                    units += len(c.encode("utf-16-le")) // 2
                return "".join(out)
            return wchar_array[n], make, flags
        if kind == "bytes":
            n = r.randint(1, 8)
            return bytes_t[n], lambda g: g.randbytes(n), flags
        if kind == "enum":
            storage, lo, hi, _ = r.choice(INT_TYPES[:3] + INT_TYPES[4:7])
            members = sorted({r.choice([lo, hi, r.randint(lo, min(hi, lo + 300))]) for _ in range(r.randint(1, 4))})
            cls = enum.IntEnum(self.name("Enum"), [(f"M{i}", v) for i, v in enumerate(members)])
            return enum_t[cls, storage], lambda g: g.choice(list(cls)), flags
        if kind == "ptr":
            flags["pointers"] = True
            return ctypes.c_void_p, lambda g: g.choice([0, g.randrange(0, 2 ** (8 * PTR_BYTES))]), flags
        if kind == "array":
            inner_ann, inner_make, inner_flags = self.field(depth + 1)
            n = r.randint(1, 3)
            return (array_t[inner_ann, n], lambda g: [inner_make(g) for _ in range(n)], inner_flags)
        built = r.choice(self.built)  # nested
        return built.cls, built.make, {"pointers": built.pointers, "bitfields": built.bitfields}

    # --- a whole class ------------------------------------------------------------------------------

    def build(self) -> Built:
        r = self.r
        aligned = r.random() < 0.4
        annotations, makers = {}, {}
        pointers = bitfields = False
        storage = r.choice(INT_TYPES)  # one storage type per class keeps bitfield layouts unambiguous
        for i in range(r.randint(1, 7)):
            if self.allow_bitfields and r.random() < 0.25:
                tp, lo, hi, bits = storage
                signed = lo < 0
                for _ in range(r.randint(1, 4)):
                    width = r.randint(1, min(bits, 12))
                    flo, fhi = (-(1 << (width - 1)), (1 << (width - 1)) - 1) if signed else (0, (1 << width) - 1)
                    name = f"b{len(annotations)}"
                    annotations[name] = bitfield[tp, width]
                    makers[name] = lambda g, flo=flo, fhi=fhi: g.randint(flo, fhi)
                bitfields = True
                continue
            ann, make, flags = self.field()
            name = f"f{len(annotations)}"
            annotations[name], makers[name] = ann, make
            pointers |= flags["pointers"]
            bitfields |= flags["bitfields"]
        cls = type(self.name("Gen"), (), {"__annotations__": annotations, "__module__": "gen"})
        cls = firstclass(byteorder=self.byteorder, align=aligned)(cls)
        built = Built(cls, lambda g, c=cls, m=makers: c(**{k: f(g) for k, f in m.items()}),
                      pointers, bitfields, aligned, self.byteorder)
        self.built.append(built)
        return built

    def build_many(self, count):
        return [self.build() for _ in range(count)]
