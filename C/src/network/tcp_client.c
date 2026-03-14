#include "tcp_client.h"
#include "../utils/debug_uart.h"

/* Pico SDK / lwIP */
#include "pico/stdlib.h"
#include "pico/cyw43_arch.h"
#include "lwip/tcp.h"
#include "lwip/dns.h"
#include "lwip/ip_addr.h"

#include <string.h>
#include <stdint.h>

/* --------------------------------------------------------------------------
 * Internal state — single TCP connection context
 * -------------------------------------------------------------------------- */

typedef enum {
    CONN_IDLE       = 0,
    CONN_RESOLVING  = 1,
    CONN_CONNECTING = 2,
    CONN_OPEN       = 3,
    CONN_CLOSING    = 4,
    CONN_ERROR      = 5,
} ConnState;

static struct {
    struct tcp_pcb *pcb;
    ConnState       state;

    /* Receive ring — lwIP delivers data in recv callback on the lwIP thread.
     * We copy it into this flat buffer and read it from the main thread. */
    uint8_t  rx_buf[TCP_RECV_BUF_LEN];
    size_t   rx_head;   /* write cursor (lwIP callback) */
    size_t   rx_tail;   /* read cursor (main thread)    */

    /* Resolved IP (from DNS or direct parse) */
    ip_addr_t server_ip;
    uint16_t  server_port;

    /* Last error code from lwIP */
    err_t last_err;
} g_tcp;

/* --------------------------------------------------------------------------
 * lwIP callback implementations
 * -------------------------------------------------------------------------- */

static err_t on_connected(void *arg, struct tcp_pcb *pcb, err_t err) {
    (void)arg;
    (void)pcb;
    if (err != ERR_OK) {
        DLOG_ERROR("TCP on_connected: error %d", (int)err);
        g_tcp.state    = CONN_ERROR;
        g_tcp.last_err = err;
        return err;
    }
    g_tcp.state = CONN_OPEN;
    DLOG_INFO("TCP connection established");
    return ERR_OK;
}

static err_t on_recv(void *arg, struct tcp_pcb *pcb,
                     struct pbuf *p, err_t err) {
    (void)arg;
    if (err != ERR_OK || p == NULL) {
        /* NULL pbuf means remote closed gracefully */
        DLOG_INFO("TCP remote closed connection");
        g_tcp.state = CONN_CLOSING;
        if (p) pbuf_free(p);
        return ERR_OK;
    }

    /* Copy pbuf chain into our linear rx_buf */
    struct pbuf *cur = p;
    while (cur != NULL) {
        size_t available = TCP_RECV_BUF_LEN - g_tcp.rx_head;
        size_t to_copy   = (cur->len < available) ? cur->len : available;
        memcpy(g_tcp.rx_buf + g_tcp.rx_head, cur->payload, to_copy);
        g_tcp.rx_head += to_copy;

        if (to_copy < cur->len) {
            DLOG_WARN("TCP rx_buf overflow — %u bytes dropped",
                      (unsigned)(cur->len - to_copy));
        }
        cur = cur->next;
    }

    /* Acknowledge received bytes to lwIP */
    tcp_recved(pcb, p->tot_len);
    pbuf_free(p);
    return ERR_OK;
}

static void on_error(void *arg, err_t err) {
    (void)arg;
    DLOG_ERROR("TCP error callback: %d", (int)err);
    g_tcp.state    = CONN_ERROR;
    g_tcp.last_err = err;
    g_tcp.pcb      = NULL; /* lwIP already freed the PCB on error */
}

static void on_dns_found(const char *name, const ip_addr_t *ipaddr, void *arg) {
    (void)name;
    (void)arg;
    if (ipaddr == NULL) {
        DLOG_ERROR("DNS resolution failed for %s", name);
        g_tcp.state = CONN_ERROR;
        return;
    }
    ip_addr_copy(g_tcp.server_ip, *ipaddr);
    g_tcp.state = CONN_CONNECTING;
}

/* --------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

static void reset_state(void) {
    g_tcp.state    = CONN_IDLE;
    g_tcp.rx_head  = 0u;
    g_tcp.rx_tail  = 0u;
    g_tcp.last_err = ERR_OK;
    g_tcp.pcb      = NULL;
}

static TcpStatus do_connect(void) {
    g_tcp.pcb = tcp_new_ip_type(IPADDR_TYPE_V4);
    if (g_tcp.pcb == NULL) {
        DLOG_ERROR("tcp_new failed");
        return TCP_ERR_CONNECT;
    }

    tcp_arg(g_tcp.pcb, NULL);
    tcp_recv(g_tcp.pcb, on_recv);
    tcp_err(g_tcp.pcb, on_error);

    err_t err = tcp_connect(g_tcp.pcb, &g_tcp.server_ip,
                            g_tcp.server_port, on_connected);
    if (err != ERR_OK) {
        DLOG_ERROR("tcp_connect failed: %d", (int)err);
        tcp_abort(g_tcp.pcb);
        g_tcp.pcb = NULL;
        return TCP_ERR_CONNECT;
    }
    return TCP_OK;
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

TcpStatus tcp_connect(const char *host, uint16_t port) {
    if (host == NULL) return TCP_ERR_NULL;

    reset_state();
    g_tcp.server_port = port;

    /* Try direct IP parse first (avoids DNS round-trip for numeric addresses) */
    if (ipaddr_aton(host, &g_tcp.server_ip)) {
        g_tcp.state = CONN_CONNECTING;
    } else {
        /* Resolve hostname via DNS */
        g_tcp.state = CONN_RESOLVING;
        cyw43_arch_lwip_begin();
        err_t dns_err = dns_gethostbyname(host, &g_tcp.server_ip,
                                          on_dns_found, NULL);
        cyw43_arch_lwip_end();

        if (dns_err == ERR_OK) {
            g_tcp.state = CONN_CONNECTING; /* cache hit */
        } else if (dns_err != ERR_INPROGRESS) {
            DLOG_ERROR("DNS error: %d", (int)dns_err);
            return TCP_ERR_CONNECT;
        }

        /* Wait for DNS resolution (max 5 s) */
        uint32_t dns_start = to_ms_since_boot(get_absolute_time());
        while (g_tcp.state == CONN_RESOLVING) {
            cyw43_arch_poll();
            sleep_ms(10u);
            if (to_ms_since_boot(get_absolute_time()) - dns_start > 5000u) {
                DLOG_ERROR("DNS timeout for %s", host);
                return TCP_ERR_CONNECT;
            }
        }
        if (g_tcp.state == CONN_ERROR) return TCP_ERR_CONNECT;
    }

    /* Issue TCP connect */
    cyw43_arch_lwip_begin();
    TcpStatus s = do_connect();
    cyw43_arch_lwip_end();
    if (s != TCP_OK) return s;

    /* Wait for connection (max 5 s) */
    uint32_t conn_start = to_ms_since_boot(get_absolute_time());
    while (g_tcp.state == CONN_CONNECTING) {
        cyw43_arch_poll();
        sleep_ms(10u);
        if (to_ms_since_boot(get_absolute_time()) - conn_start > 5000u) {
            DLOG_ERROR("TCP connect timeout");
            tcp_abort(g_tcp.pcb);
            g_tcp.pcb = NULL;
            return TCP_ERR_CONNECT;
        }
    }

    return (g_tcp.state == CONN_OPEN) ? TCP_OK : TCP_ERR_CONNECT;
}

TcpStatus tcp_connect_with_retry(const char *host, uint16_t port,
                                 uint32_t max_retries) {
    uint32_t delay_ms = 100u;
    uint32_t attempt  = 0u;

    while (max_retries == 0u || attempt < max_retries) {
        DLOG_INFO("TCP connect attempt %u to %s:%u",
                  (unsigned)(attempt + 1u), host, (unsigned)port);

        TcpStatus s = tcp_connect(host, port);
        if (s == TCP_OK) return TCP_OK;

        DLOG_WARN("TCP connect failed, retrying in %u ms",
                  (unsigned)delay_ms);
        sleep_ms(delay_ms);

        /* Exponential backoff: 100, 200, 400, 800, 1600, 3200, 6400, 8000 ms */
        delay_ms *= 2u;
        if (delay_ms > 8000u) delay_ms = 8000u;

        attempt++;
    }
    DLOG_ERROR("TCP connect failed after %u attempts", (unsigned)max_retries);
    return TCP_ERR_CONNECT;
}

TcpStatus tcp_send(const uint8_t *buf, size_t len) {
    if (buf == NULL || len == 0u) return TCP_ERR_NULL;
    if (g_tcp.state != CONN_OPEN || g_tcp.pcb == NULL) return TCP_ERR_CLOSED;

    cyw43_arch_lwip_begin();
    err_t err = tcp_write(g_tcp.pcb, buf, (u16_t)len, TCP_WRITE_FLAG_COPY);
    if (err == ERR_OK) {
        err = tcp_output(g_tcp.pcb);
    }
    cyw43_arch_lwip_end();

    if (err != ERR_OK) {
        DLOG_ERROR("tcp_send error: %d", (int)err);
        return TCP_ERR_SEND;
    }
    return TCP_OK;
}

TcpStatus tcp_recv(uint8_t *buf, size_t max_len, uint32_t timeout_ms,
                   size_t *received) {
    if (buf == NULL || received == NULL) return TCP_ERR_NULL;
    *received = 0u;

    uint32_t start = to_ms_since_boot(get_absolute_time());

    /* Poll until data arrives or timeout */
    while (g_tcp.rx_head == g_tcp.rx_tail) {
        if (g_tcp.state == CONN_ERROR || g_tcp.state == CONN_CLOSING) {
            return TCP_ERR_CLOSED;
        }
        if (timeout_ms > 0u &&
            (to_ms_since_boot(get_absolute_time()) - start) >= timeout_ms) {
            return TCP_ERR_RECV;
        }
        cyw43_arch_poll();
        sleep_ms(1u);
    }

    /* Drain available bytes (up to max_len) from the linear rx_buf */
    size_t avail = g_tcp.rx_head - g_tcp.rx_tail;
    size_t to_copy = (avail < max_len) ? avail : max_len;
    memcpy(buf, g_tcp.rx_buf + g_tcp.rx_tail, to_copy);
    g_tcp.rx_tail += to_copy;

    /* Compact the buffer once fully drained to reset cursors */
    if (g_tcp.rx_tail == g_tcp.rx_head) {
        g_tcp.rx_head = 0u;
        g_tcp.rx_tail = 0u;
    }

    *received = to_copy;
    return TCP_OK;
}

void tcp_close(void) {
    if (g_tcp.pcb == NULL) return;
    cyw43_arch_lwip_begin();
    tcp_arg(g_tcp.pcb, NULL);
    tcp_recv(g_tcp.pcb, NULL);
    tcp_err(g_tcp.pcb, NULL);
    tcp_close(g_tcp.pcb);
    cyw43_arch_lwip_end();
    g_tcp.pcb   = NULL;
    g_tcp.state = CONN_IDLE;
    DLOG_INFO("TCP connection closed");
}

bool tcp_is_connected(void) {
    return g_tcp.state == CONN_OPEN;
}
