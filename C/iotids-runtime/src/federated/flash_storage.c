#include "flash_storage.h"
#include "../utils/debug_uart.h"

/* Pico SDK flash functions */
#include "pico/stdlib.h"
#include "hardware/flash.h"
#include "hardware/sync.h"   /* save_and_disable_interrupts / restore */

#include <string.h>
#include <stdint.h>

/* --------------------------------------------------------------------------
 * Flash address layout
 *
 * PICO_FLASH_SIZE_BYTES is defined by the Pico SDK (typically 4 MB = 4194304).
 * We place the two wear-levelling sectors at the very end of flash.
 *
 * Sector B: last sector           [flash_base + SIZE - 4096 .. SIZE-1]
 * Sector A: second-to-last sector [flash_base + SIZE - 8192 .. SIZE-4097]
 *
 * Both sectors are 4 KB (FLASH_SECTOR_SIZE). The model binary must fit in
 * FLASH_MODEL_MAX_BYTES = 4096 - 16 = 4080 bytes.
 *
 * Note: flash_range_erase / flash_range_program take an OFFSET from the
 * start of flash XIP base (not an absolute address). The Pico SDK defines
 * XIP_BASE as the start of the XIP memory window.
 * -------------------------------------------------------------------------- */

#define FLASH_TOTAL_SIZE    PICO_FLASH_SIZE_BYTES
#define SECTOR_A_OFFSET     (FLASH_TOTAL_SIZE - 2u * FLASH_SECTOR_SIZE)
#define SECTOR_B_OFFSET     (FLASH_TOTAL_SIZE - 1u * FLASH_SECTOR_SIZE)

/* Model store header (16 bytes, little-endian) */
#define HEADER_MAGIC    0x494F5449u   /* "IOTI" */
#define HEADER_SIZE     16u

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t model_len;
    uint32_t crc32;
    uint32_t write_counter;
} SectorHeader;

/* --------------------------------------------------------------------------
 * CRC32 — ISO 3309 / ITU-T V.42 (same as zlib, Ethernet, etc.)
 * Table-driven for speed; 1 KB table is negligible on a 4 MB flash.
 * -------------------------------------------------------------------------- */

static uint32_t crc32_table[256];
static bool     crc32_table_ready = false;

static void crc32_init_table(void) {
    for (uint32_t i = 0u; i < 256u; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++) {
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        crc32_table[i] = c;
    }
    crc32_table_ready = true;
}

uint32_t flash_crc32(const uint8_t *buf, size_t len) {
    if (!crc32_table_ready) crc32_init_table();
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0u; i < len; i++) {
        crc = crc32_table[(crc ^ buf[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

/* --------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

/* Read the header from a sector. Returns false if magic mismatch. */
static bool read_header(uint32_t sector_offset, SectorHeader *hdr) {
    const uint8_t *sector_ptr =
        (const uint8_t *)(XIP_BASE + sector_offset);
    memcpy(hdr, sector_ptr, HEADER_SIZE);
    return hdr->magic == HEADER_MAGIC;
}

/* Return the sector offset with the highest write counter (current model). */
static uint32_t find_current_sector(uint32_t *out_counter) {
    SectorHeader hdr_a, hdr_b;
    bool valid_a = read_header(SECTOR_A_OFFSET, &hdr_a);
    bool valid_b = read_header(SECTOR_B_OFFSET, &hdr_b);

    if (!valid_a && !valid_b) {
        if (out_counter) *out_counter = 0u;
        return 0u; /* no valid model stored */
    }
    if (!valid_a) {
        if (out_counter) *out_counter = hdr_b.write_counter;
        return SECTOR_B_OFFSET;
    }
    if (!valid_b) {
        if (out_counter) *out_counter = hdr_a.write_counter;
        return SECTOR_A_OFFSET;
    }
    /* Both valid — choose the higher counter */
    if (hdr_a.write_counter >= hdr_b.write_counter) {
        if (out_counter) *out_counter = hdr_a.write_counter;
        return SECTOR_A_OFFSET;
    } else {
        if (out_counter) *out_counter = hdr_b.write_counter;
        return SECTOR_B_OFFSET;
    }
}

/* Return the sector to write NEXT (alternates for wear levelling). */
static uint32_t find_next_write_sector(uint32_t current_sector) {
    if (current_sector == SECTOR_A_OFFSET) return SECTOR_B_OFFSET;
    return SECTOR_A_OFFSET; /* default: write to A first */
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

FlashStatus flash_erase_sector(uint32_t offset) {
    /* flash_range_erase must run from SRAM. The Pico SDK handles this
     * via __no_inline_not_in_flash_func internally. We just call it. */
    uint32_t irq = save_and_disable_interrupts();
    flash_range_erase(offset, FLASH_SECTOR_SIZE);
    restore_interrupts(irq);
    return FLASH_OK;
}

FlashStatus flash_write_model(const uint8_t *data, size_t len) {
    if (data == NULL)                  return FLASH_ERR_NULL;
    if (len > FLASH_MODEL_MAX_BYTES)   return FLASH_ERR_TOO_BIG;

    /* Determine current sector & write counter */
    uint32_t cur_counter = 0u;
    uint32_t cur_sector  = find_current_sector(&cur_counter);
    uint32_t next_sector = find_next_write_sector(cur_sector);

    /* Build the header + payload into a 4 KB page-aligned buffer.
     * flash_range_program requires FLASH_PAGE_SIZE (256 B) alignment and
     * length. We pad to the next 256-byte boundary with 0xFF (flash erased
     * state) which is harmless. */
    static uint8_t write_buf[FLASH_SECTOR_SIZE] __attribute__((aligned(4)));
    memset(write_buf, 0xFF, FLASH_SECTOR_SIZE);

    SectorHeader hdr;
    hdr.magic         = HEADER_MAGIC;
    hdr.model_len     = (uint32_t)len;
    hdr.crc32         = flash_crc32(data, len);
    hdr.write_counter = cur_counter + 1u;

    memcpy(write_buf,              &hdr,  HEADER_SIZE);
    memcpy(write_buf + HEADER_SIZE, data, len);

    /* Erase sector then program */
    DLOG_INFO("Flash: erasing sector at offset 0x%08X", (unsigned)next_sector);
    flash_erase_sector(next_sector);

    /* Program in 256-byte pages */
    size_t   pages    = (HEADER_SIZE + len + FLASH_PAGE_SIZE - 1u)
                        / FLASH_PAGE_SIZE;
    uint32_t irq = save_and_disable_interrupts();
    flash_range_program(next_sector, write_buf, pages * FLASH_PAGE_SIZE);
    restore_interrupts(irq);

    /* Verify by re-reading header */
    SectorHeader verify;
    if (!read_header(next_sector, &verify) ||
        verify.crc32  != hdr.crc32 ||
        verify.model_len != hdr.model_len) {
        DLOG_ERROR("Flash verify failed after write");
        return FLASH_ERR_WRITE;
    }

    DLOG_INFO("Flash: model written (%u bytes, CRC 0x%08X, ctr %u)",
              (unsigned)len, (unsigned)hdr.crc32, (unsigned)hdr.write_counter);
    return FLASH_OK;
}

FlashStatus flash_read_model(uint8_t *buf, size_t max_len, size_t *out_len) {
    if (buf == NULL || out_len == NULL) return FLASH_ERR_NULL;

    uint32_t cur_sector = find_current_sector(NULL);
    if (cur_sector == 0u) {
        DLOG_INFO("Flash: no persisted model found (first boot)");
        return FLASH_ERR_MAGIC;
    }

    const uint8_t *sector_ptr = (const uint8_t *)(XIP_BASE + cur_sector);

    SectorHeader hdr;
    memcpy(&hdr, sector_ptr, HEADER_SIZE);
    if (hdr.magic != HEADER_MAGIC) return FLASH_ERR_MAGIC;

    size_t model_len = hdr.model_len;
    if (model_len > max_len || model_len > FLASH_MODEL_MAX_BYTES) {
        DLOG_ERROR("Flash: stored model (%u B) exceeds buffer (%u B)",
                   (unsigned)model_len, (unsigned)max_len);
        return FLASH_ERR_TOO_BIG;
    }

    const uint8_t *model_ptr = sector_ptr + HEADER_SIZE;
    uint32_t actual_crc = flash_crc32(model_ptr, model_len);
    if (actual_crc != hdr.crc32) {
        DLOG_ERROR("Flash: CRC mismatch (stored 0x%08X, actual 0x%08X)",
                   (unsigned)hdr.crc32, (unsigned)actual_crc);
        return FLASH_ERR_CRC;
    }

    memcpy(buf, model_ptr, model_len);
    *out_len = model_len;

    DLOG_INFO("Flash: loaded model (%u bytes, CRC OK, ctr %u)",
              (unsigned)model_len, (unsigned)hdr.write_counter);
    return FLASH_OK;
}
