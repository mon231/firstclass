/* A small wire protocol, as a firmware team might ship it. */
#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>
#include <stdbool.h>

#define MAX_NAME      8
#define MAX_READINGS  4
#define MAC_LEN       6

enum Command {                 /* C enum: int-sized */
    CMD_PING  = 1,
    CMD_DATA  = 0x10,
    CMD_RESET = 0xFF,
};

enum class Status : uint8_t {  /* enum class with an underlying type */
    Ok     = 0,
    Busy,
    Failed = 0x80,
};

enum class Channel : uint16_t {
    A = 1,
    B,
    Aux = 1000,
};

/* everything on the wire is packed: no padding */
#pragma pack(push, 1)

typedef struct Header {
    uint32_t     magic;
    uint8_t      version;
    uint8_t      ready    : 1;   /* bitfields share one byte */
    uint8_t      error    : 1;
    uint8_t      priority : 6;
    enum Command command;
    Status       status;
} Header;

typedef struct Reading {
    Channel channel;
    int16_t value;
} Reading;

typedef struct Message {
    Header   header;
    char     sender[MAX_NAME];
    uint8_t  mac[MAC_LEN];
    Reading  readings[MAX_READINGS];
    float    scale;
} Message;

#pragma pack(pop)

/* no pragma: normal C layout with padding */
struct InMemory {
    uint8_t  tag;
    uint32_t id;
    uint8_t  level;
    uint64_t stamp;
};

union Unsupported { int i; float f; };   /* skipped, reported in .skipped */

#endif /* PROTOCOL_H */
