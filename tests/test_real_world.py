"""Well-known wire/file formats written as ordinary C and Rust definitions: sizes and real bytes."""
import struct
import sys
import unittest

from firstclass import from_header, from_rust

ELF = """
#include <stdint.h>
#define EI_NIDENT 16
typedef uint16_t Elf32_Half; typedef uint32_t Elf32_Word; typedef uint32_t Elf32_Addr; typedef uint32_t Elf32_Off;
typedef uint16_t Elf64_Half; typedef uint32_t Elf64_Word; typedef uint64_t Elf64_Addr; typedef uint64_t Elf64_Off;
typedef uint64_t Elf64_Xword;
typedef struct {
    unsigned char e_ident[EI_NIDENT];
    Elf32_Half e_type; Elf32_Half e_machine; Elf32_Word e_version; Elf32_Addr e_entry;
    Elf32_Off e_phoff; Elf32_Off e_shoff; Elf32_Word e_flags; Elf32_Half e_ehsize;
    Elf32_Half e_phentsize; Elf32_Half e_phnum; Elf32_Half e_shentsize; Elf32_Half e_shnum; Elf32_Half e_shstrndx;
} Elf32_Ehdr;
typedef struct {
    unsigned char e_ident[EI_NIDENT];
    Elf64_Half e_type; Elf64_Half e_machine; Elf64_Word e_version; Elf64_Addr e_entry;
    Elf64_Off e_phoff; Elf64_Off e_shoff; Elf64_Word e_flags; Elf64_Half e_ehsize;
    Elf64_Half e_phentsize; Elf64_Half e_phnum; Elf64_Half e_shentsize; Elf64_Half e_shnum; Elf64_Half e_shstrndx;
} Elf64_Ehdr;
typedef struct {
    Elf64_Word p_type; Elf64_Word p_flags; Elf64_Off p_offset; Elf64_Addr p_vaddr; Elf64_Addr p_paddr;
    Elf64_Xword p_filesz; Elf64_Xword p_memsz; Elf64_Xword p_align;
} Elf64_Phdr;
"""

BMP = """
#include <stdint.h>
#pragma pack(push, 1)
typedef struct { uint16_t bfType; uint32_t bfSize; uint16_t bfReserved1; uint16_t bfReserved2; uint32_t bfOffBits; } BITMAPFILEHEADER;
typedef struct {
    uint32_t biSize; int32_t biWidth; int32_t biHeight; uint16_t biPlanes; uint16_t biBitCount;
    uint32_t biCompression; uint32_t biSizeImage; int32_t biXPelsPerMeter; int32_t biYPelsPerMeter;
    uint32_t biClrUsed; uint32_t biClrImportant;
} BITMAPINFOHEADER;
#pragma pack(pop)
"""

NET = """
#include <stdint.h>
#pragma pack(push, 1)
struct ethhdr { uint8_t h_dest[6]; uint8_t h_source[6]; uint16_t h_proto; };
struct iphdr {
    uint8_t  version : 4, ihl : 4;       /* network (big-endian) bit order, as in the RFC diagram */
    uint8_t  tos; uint16_t tot_len; uint16_t id; uint16_t frag_off; uint8_t ttl; uint8_t protocol;
    uint16_t check; uint32_t saddr; uint32_t daddr;
};
struct udphdr { uint16_t source; uint16_t dest; uint16_t len; uint16_t check; };
struct tcphdr {
    uint16_t source; uint16_t dest; uint32_t seq; uint32_t ack_seq;
    uint16_t doff : 4, res1 : 4, cwr : 1, ece : 1, urg : 1, ack : 1, psh : 1, rst : 1, syn : 1, fin : 1;
    uint16_t window; uint16_t check; uint16_t urg_ptr;
};
#pragma pack(pop)
"""

PE = """
typedef unsigned short WORD; typedef unsigned long DWORD; typedef long LONG; typedef unsigned char BYTE;
#pragma pack(push, 2)
#pragma pack(pop)
typedef struct _IMAGE_DOS_HEADER {
    WORD e_magic; WORD e_cblp; WORD e_cp; WORD e_crlc; WORD e_cparhdr; WORD e_minalloc; WORD e_maxalloc;
    WORD e_ss; WORD e_sp; WORD e_csum; WORD e_ip; WORD e_cs; WORD e_lfarlc; WORD e_ovno; WORD e_res[4];
    WORD e_oemid; WORD e_oeminfo; WORD e_res2[10]; LONG e_lfanew;
} IMAGE_DOS_HEADER;
typedef struct _IMAGE_FILE_HEADER {
    WORD Machine; WORD NumberOfSections; DWORD TimeDateStamp; DWORD PointerToSymbolTable;
    DWORD NumberOfSymbols; WORD SizeOfOptionalHeader; WORD Characteristics;
} IMAGE_FILE_HEADER;
"""


class Elf(unittest.TestCase):
    def setUp(self):
        self.t = from_header(text=ELF)

    def test_sizes(self):
        self.assertEqual(self.t.skipped, {})
        self.assertEqual((self.t.Elf32_Ehdr.sizeof(), self.t.Elf64_Ehdr.sizeof(), self.t.Elf64_Phdr.sizeof()),
                         (52, 64, 56))

    def test_parse_a_real_header(self):
        ident = b"\x7fELF\x02\x01\x01\x00" + bytes(8)
        raw = ident + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0x1040, 64, 0x3a0, 0, 64, 56, 13, 64, 31, 30)
        h = self.t.Elf64_Ehdr.from_bytes(raw)
        self.assertEqual(h.e_ident[:4], b"\x7fELF")
        self.assertEqual((h.e_type, h.e_machine, h.e_entry, h.e_phoff, h.e_phnum), (3, 62, 0x1040, 64, 13))
        self.assertEqual(h.to_bytes(), raw)

    def test_this_python_binary_if_it_is_elf(self):
        try:
            with open(sys.executable, "rb") as f:
                head = f.read(64)
        except OSError:
            self.skipTest("cannot read the interpreter")
        if head[:4] != b"\x7fELF" or head[4] != 2 or head[5] != 1:
            self.skipTest("not a 64-bit little-endian ELF")
        h = self.t.Elf64_Ehdr.from_bytes(head)
        self.assertEqual((h.e_ehsize, h.e_phentsize), (64, 56))
        self.assertEqual(h.to_bytes(), head)


class Bmp(unittest.TestCase):
    def test_headers(self):
        t = from_header(text=BMP)
        self.assertEqual((t.BITMAPFILEHEADER.sizeof(), t.BITMAPINFOHEADER.sizeof()), (14, 40))
        raw = struct.pack("<2sIHHI", b"BM", 70, 0, 0, 54) + struct.pack("<IiiHHIIiiII", 40, 4, -4, 1, 24, 0, 16, 2835, 2835, 0, 0)
        self.assertEqual(len(raw), 54)
        fh = t.BITMAPFILEHEADER.from_bytes(raw[:14])
        ih = t.BITMAPINFOHEADER.from_bytes(raw[14:])
        self.assertEqual((fh.bfType, fh.bfSize, fh.bfOffBits), (0x4D42, 70, 54))
        self.assertEqual((ih.biWidth, ih.biHeight, ih.biBitCount), (4, -4, 24))
        self.assertEqual(fh.to_bytes() + ih.to_bytes(), raw)


class Network(unittest.TestCase):
    PACKET = bytes.fromhex(
        "aabbccddeeff 112233445566 0800"                       # ethernet
        "45 00 0030 1c46 4000 40 11 b1e6 c0a80001 c0a800c7"     # ipv4: version 4, ihl 5, ttl 64, udp
        "1f90 0035 001c 0000"                                    # udp 8080 -> 53
    )

    def setUp(self):
        self.t = from_header(text=NET, byteorder="big")

    def test_sizes(self):
        t = self.t
        self.assertEqual(t.skipped, {})
        self.assertEqual((t.ethhdr.sizeof(), t.iphdr.sizeof(), t.udphdr.sizeof(), t.tcphdr.sizeof()), (14, 20, 8, 20))

    def test_decode_a_real_packet(self):
        t = self.t
        eth = t.ethhdr.unpack_from(self.PACKET, 0)
        ip = t.iphdr.unpack_from(self.PACKET, 14)
        udp = t.udphdr.unpack_from(self.PACKET, 34)
        self.assertEqual(eth.h_proto, 0x0800)
        self.assertEqual(eth.h_dest, bytes.fromhex("aabbccddeeff"))
        self.assertEqual((ip.version, ip.ihl, ip.ttl, ip.protocol, ip.tot_len), (4, 5, 64, 17, 0x30))
        self.assertEqual((ip.saddr, ip.daddr), (0xC0A80001, 0xC0A800C7))
        self.assertEqual((udp.source, udp.dest, udp.len), (8080, 53, 28))

    def test_reencode_gives_the_same_bytes(self):
        t = self.t
        out = (t.ethhdr.unpack_from(self.PACKET, 0).to_bytes() + t.iphdr.unpack_from(self.PACKET, 14).to_bytes()
               + t.udphdr.unpack_from(self.PACKET, 34).to_bytes())
        self.assertEqual(out, self.PACKET)

    def test_tcp_flags(self):
        t = self.t
        # offset 5 words, SYN+ACK: 0x5012 on the wire
        raw = struct.pack(">HHIIHHHH", 443, 51000, 1000, 2000, 0x5012, 65535, 0, 0)
        tcp = t.tcphdr.from_bytes(raw)
        self.assertEqual((tcp.doff, tcp.syn, tcp.ack, tcp.fin, tcp.rst, tcp.psh), (5, 1, 1, 0, 0, 0))
        self.assertEqual(tcp.to_bytes(), raw)
        built = t.tcphdr(443, 51000, 1000, 2000, 5, 0, 0, 0, 0, 1, 0, 0, 1, 0, 65535, 0, 0)
        self.assertEqual(built.to_bytes(), raw)


class Pe(unittest.TestCase):
    def test_windows_typedefs_with_four_byte_long(self):
        t = from_header(text=PE, long_size=4)  # Windows data model: long is 4 bytes
        self.assertEqual(t.skipped, {})
        self.assertEqual((t.IMAGE_DOS_HEADER.sizeof(), t.IMAGE_FILE_HEADER.sizeof()), (64, 20))
        raw = bytearray(64)
        raw[0:2] = b"MZ"
        raw[60:64] = struct.pack("<i", 0x80)
        dos = t.IMAGE_DOS_HEADER.from_bytes(bytes(raw))
        self.assertEqual((dos.e_magic, dos.e_lfanew), (0x5A4D, 0x80))

    def test_long_is_eight_bytes_elsewhere(self):
        t = from_header(text=PE, long_size=8)
        self.assertGreater(t.IMAGE_FILE_HEADER.sizeof(), 20)


class RustFormats(unittest.TestCase):
    def test_sockaddr_and_timespec_like(self):
        t = from_rust(text="""
            #[repr(C)] pub struct in_addr { pub s_addr: u32 }
            #[repr(C)] pub struct sockaddr_in {
                pub sin_family: u16, pub sin_port: u16, pub sin_addr: in_addr, pub sin_zero: [u8; 8],
            }
            #[repr(C)] pub struct timespec { pub tv_sec: i64, pub tv_nsec: i64 }
            #[repr(C, packed)] pub struct tar_header_prefix { pub name: [c_char; 100], pub mode: [c_char; 8] }
        """)
        self.assertEqual(t.skipped, {})
        self.assertEqual((t.sockaddr_in.sizeof(), t.timespec.sizeof(), t.tar_header_prefix.sizeof()), (16, 16, 108))
        sa = t.sockaddr_in(2, 0x5000, t.in_addr(0x0100007F), bytes(8))
        self.assertEqual(sa.to_bytes(), struct.pack("<HHI8s", 2, 0x5000, 0x0100007F, bytes(8)))
        tar = t.tar_header_prefix("file.txt", "0000644")
        self.assertEqual(tar.to_bytes()[:8], b"file.txt")
        self.assertEqual(t.tar_header_prefix.from_bytes(tar.to_bytes()), tar)


if __name__ == "__main__":
    unittest.main()
