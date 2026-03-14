#ifndef IOTIDS_WIFI_INIT_H
#define IOTIDS_WIFI_INIT_H

/*
 * wifi_init.h — CYW43 Wi-Fi driver init and WPA2 association for Pico 2W.
 *
 * Wraps the Pico W SDK's pico_cyw43_arch (lwIP + CYW43 driver) to provide
 * a simple blocking connect/status API.
 *
 * IMPORTANT: The CYW43 driver runs on core 1 via the Pico SDK multicore
 * API. Do NOT call inference (model_runner) from core 1. Run Wi-Fi polling
 * and weight receiving on core 1; run inference on core 0.
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum SSID and password lengths (including NUL terminator) */
#define WIFI_SSID_MAX_LEN 33u
#define WIFI_PASS_MAX_LEN 64u

/* IP address string buffer length (e.g. "192.168.1.123\0") */
#define WIFI_IP_STR_LEN   16u

/*
 * wifi_connect — initialise the CYW43 driver and join the given WPA2 network.
 *
 *   ssid     : null-terminated network name (max 32 chars)
 *   password : null-terminated WPA2 passphrase (max 63 chars)
 *
 * This call is non-blocking: it starts the association process and returns
 * immediately. Use wifi_wait_connected() to block until the link is up.
 *
 * Returns true if the driver was initialised successfully, false otherwise.
 * A false return is fatal — blink the LED and halt.
 */
bool wifi_connect(const char *ssid, const char *password);

/*
 * wifi_wait_connected — block until the Wi-Fi link is up, or timeout.
 *
 *   timeout_ms : maximum milliseconds to wait. Use 10000 (10 s) as a
 *                reasonable default for a typical home/lab network.
 *
 * If the connection does not succeed within timeout_ms, the watchdog timer
 * will trigger a system reset to avoid hanging indefinitely.
 *
 * Returns true if connected, false on timeout (system will reset).
 */
bool wifi_wait_connected(uint32_t timeout_ms);

/*
 * wifi_is_connected — non-blocking status poll. Safe to call from either core.
 * Returns true if the Wi-Fi link is currently up.
 */
bool wifi_is_connected(void);

/*
 * wifi_get_ip_str — copy the assigned IPv4 address into buf as a string.
 *   buf : output buffer of at least WIFI_IP_STR_LEN bytes.
 * Returns buf for convenience. Returns "0.0.0.0" if not connected.
 */
char *wifi_get_ip_str(char *buf);

#ifdef __cplusplus
}
#endif

#endif /* IOTIDS_WIFI_INIT_H */
