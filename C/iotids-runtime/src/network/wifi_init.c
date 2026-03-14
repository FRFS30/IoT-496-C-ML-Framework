#include "wifi_init.h"
#include "../utils/debug_uart.h"

/* Pico W SDK — CYW43 + lwIP */
#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"

#include <string.h>
#include <stdio.h>

/* --------------------------------------------------------------------------
 * Internal state
 * -------------------------------------------------------------------------- */

static bool g_driver_ready = false;

/* Credentials stored for reconnect (watchdog reset recovery) */
static char g_ssid[WIFI_SSID_MAX_LEN];
static char g_pass[WIFI_PASS_MAX_LEN];

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

bool wifi_connect(const char *ssid, const char *password) {
    if (ssid == NULL || password == NULL) {
        DLOG_ERROR("wifi_connect: NULL credentials");
        return false;
    }

    /*
     * cyw43_arch_init — initialise the CYW43 driver and lwIP stack.
     * Uses cyw43_arch_poll (cooperative polling) because the inference loop
     * is on core 0 and the Wi-Fi driver runs its own background poll on core 1.
     * If your design uses RTOS threading, switch to cyw43_arch_init_freertos().
     */
    if (cyw43_arch_init() != 0) {
        DLOG_ERROR("CYW43 driver init failed");
        return false;
    }

    /* Enable station (STA) mode */
    cyw43_arch_enable_sta_mode();

    /* Store credentials for potential reconnection after watchdog reset */
    strncpy(g_ssid, ssid, WIFI_SSID_MAX_LEN - 1u);
    strncpy(g_pass, password, WIFI_PASS_MAX_LEN - 1u);
    g_ssid[WIFI_SSID_MAX_LEN - 1u] = '\0';
    g_pass[WIFI_PASS_MAX_LEN - 1u] = '\0';

    DLOG_INFO("Connecting to SSID: %s", g_ssid);

    /*
     * Start the association. CYW43_AUTH_WPA2_AES_PSK covers the vast majority
     * of modern WPA2 networks. For WPA3 or enterprise, see cyw43 docs.
     */
    int rc = cyw43_arch_wifi_connect_async(g_ssid, g_pass,
                                           CYW43_AUTH_WPA2_AES_PSK);
    if (rc != 0) {
        DLOG_ERROR("cyw43_arch_wifi_connect_async failed: %d", rc);
        return false;
    }

    g_driver_ready = true;
    return true;
}

bool wifi_wait_connected(uint32_t timeout_ms) {
    if (!g_driver_ready) return false;

    uint32_t start = to_ms_since_boot(get_absolute_time());

    while (!wifi_is_connected()) {
        cyw43_arch_poll();
        sleep_ms(100u);

        if ((to_ms_since_boot(get_absolute_time()) - start) > timeout_ms) {
            DLOG_ERROR("Wi-Fi timeout after %u ms — resetting via watchdog",
                       (unsigned)timeout_ms);
            /*
             * Trigger a hardware watchdog reset. The Pico SDK's watchdog will
             * fire in ~100 ms, giving the UART time to flush the error message.
             * On reboot, main() will re-attempt wifi_connect().
             */
            watchdog_enable(100u, true);
            while (true) { /* spin until watchdog fires */ }
            return false;  /* unreachable, silences compiler warning */
        }
    }

    char ip_buf[WIFI_IP_STR_LEN];
    DLOG_INFO("Wi-Fi connected. IP: %s", wifi_get_ip_str(ip_buf));
    return true;
}

bool wifi_is_connected(void) {
    if (!g_driver_ready) return false;
    return cyw43_tcpip_link_status(&cyw43_state, CYW43_ITF_STA)
           == CYW43_LINK_UP;
}

char *wifi_get_ip_str(char *buf) {
    if (buf == NULL) return NULL;

    if (!wifi_is_connected()) {
        strncpy(buf, "0.0.0.0", WIFI_IP_STR_LEN);
        return buf;
    }

    uint32_t ip = cyw43_state.netif[CYW43_ITF_STA].ip_addr.addr;
    snprintf(buf, WIFI_IP_STR_LEN, "%u.%u.%u.%u",
             (unsigned)((ip      ) & 0xFFu),
             (unsigned)((ip >>  8) & 0xFFu),
             (unsigned)((ip >> 16) & 0xFFu),
             (unsigned)((ip >> 24) & 0xFFu));
    return buf;
}
