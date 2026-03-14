#ifndef IOTIDS_DEBUG_UART_H
#define IOTIDS_DEBUG_UART_H

/*
 * debug_uart.h — UART0 serial logging for the iotids runtime.
 *
 * Provides printf-style logging with four severity levels. DEBUG-level logs
 * are stripped at compile time in production builds (NDEBUG defined) to
 * avoid the cost of string formatting on the hot inference path.
 *
 * Uses Pico SDK uart_* functions internally. UART0 defaults to GPIO 0/1.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Log levels — numeric so level comparisons work */
typedef enum {
    LOG_DEBUG = 0,
    LOG_INFO  = 1,
    LOG_WARN  = 2,
    LOG_ERROR = 3,
} LogLevel;

/*
 * debug_init — configure UART0 at the given baud rate and enable it.
 *   baud : typically 115200 for development, 921600 for high-throughput logs.
 * Call once at the very start of main(), before anything else, so boot
 * messages are captured.
 */
void debug_init(uint32_t baud);

/*
 * debug_log — printf-style log with level prefix and newline.
 *   level : LOG_DEBUG / LOG_INFO / LOG_WARN / LOG_ERROR
 *   fmt   : printf format string
 *   ...   : format arguments
 *
 * In NDEBUG builds, LOG_DEBUG calls compile to nothing (zero cost).
 * All other levels always emit.
 */
void debug_log(LogLevel level, const char *fmt, ...);

/*
 * Convenience macros — recommended over calling debug_log directly because
 * DEBUG-level calls are completely eliminated in production builds.
 */
#ifdef NDEBUG
#define DLOG_DEBUG(fmt, ...) ((void)0)
#else
#define DLOG_DEBUG(fmt, ...) debug_log(LOG_DEBUG, fmt, ##__VA_ARGS__)
#endif

#define DLOG_INFO(fmt, ...)  debug_log(LOG_INFO,  fmt, ##__VA_ARGS__)
#define DLOG_WARN(fmt, ...)  debug_log(LOG_WARN,  fmt, ##__VA_ARGS__)
#define DLOG_ERROR(fmt, ...) debug_log(LOG_ERROR, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* IOTIDS_DEBUG_UART_H */
