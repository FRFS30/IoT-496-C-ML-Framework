#include "debug_uart.h"

/* Pico SDK headers */
#include "pico/stdlib.h"
#include "hardware/uart.h"
#include "hardware/gpio.h"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * Configuration — override in CMakeLists.txt if needed
 * -------------------------------------------------------------------------- */

#ifndef IOTIDS_UART_ID
#define IOTIDS_UART_ID uart0
#endif

#ifndef IOTIDS_UART_TX_PIN
#define IOTIDS_UART_TX_PIN 0u   /* GP0 */
#endif

#ifndef IOTIDS_UART_RX_PIN
#define IOTIDS_UART_RX_PIN 1u   /* GP1 */
#endif

/* Maximum formatted log line length. Fits on an 80-col terminal with prefix. */
#define LOG_BUF_LEN 256u

/* --------------------------------------------------------------------------
 * Internal state
 * -------------------------------------------------------------------------- */

static bool g_uart_ready = false;

static const char *const LEVEL_STRINGS[] = {
    "[DBG] ",
    "[INF] ",
    "[WRN] ",
    "[ERR] "
};

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

void debug_init(uint32_t baud) {
    uart_init(IOTIDS_UART_ID, baud);

    gpio_set_function(IOTIDS_UART_TX_PIN, GPIO_FUNC_UART);
    gpio_set_function(IOTIDS_UART_RX_PIN, GPIO_FUNC_UART);

    /* 8 data bits, 1 stop bit, no parity — standard serial console settings */
    uart_set_format(IOTIDS_UART_ID, 8, 1, UART_PARITY_NONE);
    uart_set_fifo_enabled(IOTIDS_UART_ID, true);

    g_uart_ready = true;

    /* First message confirms UART is alive */
    debug_log(LOG_INFO, "iotids runtime booting...");
}

void debug_log(LogLevel level, const char *fmt, ...) {
    if (!g_uart_ready) return;

    /* Clamp unknown levels to ERROR */
    if ((unsigned)level > (unsigned)LOG_ERROR) level = LOG_ERROR;

    char buf[LOG_BUF_LEN];
    int  prefix_len = (int)strlen(LEVEL_STRINGS[level]);

    /* Write level prefix */
    memcpy(buf, LEVEL_STRINGS[level], (size_t)prefix_len);

    /* Format the caller's message into the rest of the buffer */
    va_list args;
    va_start(args, fmt);
    int written = vsnprintf(buf + prefix_len,
                            LOG_BUF_LEN - (size_t)prefix_len - 2u, /* -2 for \r\n */
                            fmt, args);
    va_end(args);

    if (written < 0) written = 0;

    /* Append CR+LF for compatibility with Windows serial terminals */
    int total = prefix_len + written;
    if (total > (int)(LOG_BUF_LEN - 2)) total = (int)(LOG_BUF_LEN - 2);
    buf[total]     = '\r';
    buf[total + 1] = '\n';
    total += 2;

    uart_write_blocking(IOTIDS_UART_ID, (const uint8_t *)buf, (size_t)total);
}
