#ifndef IOTIDS_FLASH_STORAGE_H
#define IOTIDS_FLASH_STORAGE_H

/*
 * flash_storage.h — model persistence in RP2350 flash.
 *
 * The RP2350 has 4 MB of XIP flash. The last flash sector is reserved for
 * the iotids runtime code and read-only model embedded via iotids_model.tflite.h.
 * This module uses the SECOND-TO-LAST sector region for mutable weight storage,
 * so federated updates survive power cycles without touching the immutable
 * default model.
 *
 * Flash layout (top of 4 MB):
 *   0x001F_E000 — 0x001F_EFFF (4 KB)  : sector A — primary model store
 *   0x001F_F000 — 0x001F_FFFF (4 KB)  : sector B — wear-levelling alternate
 *
 * Wear levelling: writes alternate between sector A and B, chosen by a
 * monotonic write counter stored in the sector header. The sector with the
 * higher counter is the current one. This halves the write cycles per sector,
 * extending flash life from ~100 K to ~200 K erase cycles.
 *
 * Each stored model is prefixed with a 16-byte header:
 *   [0..3]  : magic number 0x494F5449 ("IOTI")
 *   [4..7]  : model length in bytes (uint32_t, little-endian)
 *   [8..11] : CRC32 of the model bytes
 *   [12..15]: write counter (uint32_t, monotonically increasing)
 *
 * The RP2350 requires an erase before every write. flash_write_model()
 * handles this automatically.
 *
 * WARNING: Flash writes disable XIP temporarily. The RP2350 must be running
 * from SRAM during flash erase/write. The Pico SDK's flash_range_erase() and
 * flash_range_program() handle this via the ROM flash functions automatically.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Flash erase granularity on RP2350 is 4096 bytes (one sector). */
#define FLASH_SECTOR_SIZE  4096u

/* Maximum model size that fits in one sector minus the 16-byte header.
 * 4080 bytes > the 30 KB target, but federated updates may be partial
 * (layer delta only). Full model must fit in one sector. */
#define FLASH_MODEL_MAX_BYTES  (FLASH_SECTOR_SIZE - 16u)

typedef enum {
    FLASH_OK          =  0,
    FLASH_ERR_NULL    = -1,   /* NULL pointer argument                  */
    FLASH_ERR_TOO_BIG = -2,   /* model_len > FLASH_MODEL_MAX_BYTES      */
    FLASH_ERR_CRC     = -3,   /* CRC32 mismatch on read                 */
    FLASH_ERR_MAGIC   = -4,   /* magic number not found (sector empty)  */
    FLASH_ERR_WRITE   = -5,   /* flash_range_program failed             */
} FlashStatus;

/*
 * flash_write_model — persist a model (or weight delta) to flash.
 *
 *   data : pointer to model bytes (typically the INT8 weight array).
 *   len  : number of bytes to write. Must be <= FLASH_MODEL_MAX_BYTES.
 *
 * Automatically selects the next wear-levelling sector, erases it, and writes
 * the header + data. Returns FLASH_OK on success.
 *
 * This function must NOT be called from an interrupt or while TFLM is running
 * (flash writes temporarily disable the XIP cache). Call only from the
 * federated weight update handler when the inference loop is idle.
 */
FlashStatus flash_write_model(const uint8_t *data, size_t len);

/*
 * flash_read_model — load the most recently persisted model from flash.
 *
 *   buf     : output buffer of at least max_len bytes.
 *   max_len : size of buf. Should be >= FLASH_MODEL_MAX_BYTES.
 *   out_len : set to actual model length on FLASH_OK.
 *
 * Performs CRC32 validation. Returns FLASH_ERR_MAGIC if no valid model has
 * ever been written (first boot with no FL update yet — caller falls back to
 * the default iotids_model.tflite.h embedded model).
 */
FlashStatus flash_read_model(uint8_t *buf, size_t max_len, size_t *out_len);

/*
 * flash_erase_sector — erase a single 4 KB sector at the given flash offset.
 *   offset : byte offset from the start of flash (must be sector-aligned).
 * Direct use is rarely needed; flash_write_model() calls this internally.
 */
FlashStatus flash_erase_sector(uint32_t offset);

/*
 * flash_crc32 — compute CRC32 (ISO 3309 polynomial) over buf[0..len-1].
 * Exposed so weight_receiver.c can pre-validate incoming payloads before
 * calling flash_write_model().
 */
uint32_t flash_crc32(const uint8_t *buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* IOTIDS_FLASH_STORAGE_H */
