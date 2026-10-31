from ._core import firstclass
from ._header import from_header
from ._specs import (
    array_t,
    bitfield,
    bool_t,
    bytes_t,
    char_array,
    char_t,
    enum_t,
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

__all__ = [
    "firstclass", "from_header",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "float32_t", "float64_t", "bool_t", "char_t",
    "char_array", "bytes_t", "array_t", "bitfield", "enum_t",
]

firstclass.from_header = from_header  # also reachable as the decorator's attribute
