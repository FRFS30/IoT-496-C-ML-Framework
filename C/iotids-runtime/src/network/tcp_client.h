#ifndef IOTIDS_TCP_CLIENT_H
#define IOTIDS_TCP_CLIENT_H

/*
 * tcp_client.h — lightweight TCP client for federated learning communication.
 *
 * Handles two data flows:
 *   OUTBOUND : inference results (label + probability) sent to the Python
 *              FedAvg server after each classification.
 *   INBOUND  : weight update payloads received from the server, forwarded
 *              to weight_receiver for validation and application.
 *
 * Built on lwIP raw TCP API (no POSIX sockets — Pico W does not have them
 * unless using the full lwIP socket layer, which costs ~20 KB extra).
 *
 * All operations are non-blocking. tcp_recv uses a timeout so the inference
 * loop is never stalled indefinitely waiting for the server.
 *
 * Retry policy: exponential backoff (100 ms base, 2x multiplier, 8 s cap).
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum receive buffer size — must be >= largest weight update payload.
 * A fully quantized INT8 DNN with 77 inputs and 3 hidden layers of 64 neurons
 * has roughly (77*64 + 64*64 + 64*64 + 64*1) = ~12 KB of weights.
 * Set to 16 KB for headroom. Adjust if the model architecture changes. */
#ifndef TCP_RECV_BUF_LEN
#define TCP_RECV_BUF_LEN (16u * 1024u)
#endif

/* Maximum host string length including NUL */
#define TCP_HOST_MAX_LEN 64u

typedef enum {
    TCP_OK           =  0,
    TCP_ERR_CONNECT  = -1,   /* connection refused or network unreachable */
    TCP_ERR_SEND     = -2,   /* send failed                               */
    TCP_ERR_RECV     = -3,   /* receive timed out or connection dropped   */
    TCP_ERR_CLOSED   = -4,   /* remote side closed the connection         */
    TCP_ERR_NULL     = -5,   /* NULL pointer argument                     */
    TCP_ERR_OVERFLOW = -6,   /* incoming payload exceeds recv buffer      */
} TcpStatus;

/*
 * tcp_connect — open a TCP connection to the Python FedAvg server.
 *
 *   host : IPv4 address string (e.g. "192.168.1.10") or hostname.
 *          Use an IP address to avoid DNS overhead on the Pico.
 *   port : TCP port number (e.g. 5555).
 *
 * Blocks until connection is established or fails. Returns TCP_OK on success.
 * On failure, returns TCP_ERR_CONNECT. The caller should retry with
 * exponential backoff (see tcp_connect_with_retry).
 */
TcpStatus tcp_connect(const char *host, uint16_t port);

/*
 * tcp_connect_with_retry — tcp_connect with built-in exponential backoff.
 *
 *   max_retries : maximum number of attempts (0 = try forever).
 *
 * Returns TCP_OK if any attempt succeeds. Returns TCP_ERR_CONNECT if all
 * retries are exhausted. Logs each attempt via debug_uart.
 */
TcpStatus tcp_connect_with_retry(const char *host, uint16_t port,
                                 uint32_t max_retries);

/*
 * tcp_send — transmit buf[0..len-1] over the open connection.
 *
 * The send is synchronous from the caller's perspective (blocks until lwIP
 * has accepted the data into its send buffer). Returns TCP_OK on success.
 */
TcpStatus tcp_send(const uint8_t *buf, size_t len);

/*
 * tcp_recv — receive up to max_len bytes into buf, waiting up to timeout_ms.
 *
 *   buf        : output buffer (must be at least max_len bytes).
 *   max_len    : maximum bytes to accept.
 *   timeout_ms : milliseconds before giving up. Use 0 for non-blocking poll.
 *   received   : set to actual bytes written into buf on TCP_OK.
 *
 * Returns TCP_OK if at least one byte was received, TCP_ERR_RECV on timeout,
 * TCP_ERR_CLOSED if the server closed the connection.
 */
TcpStatus tcp_recv(uint8_t *buf, size_t max_len, uint32_t timeout_ms,
                   size_t *received);

/*
 * tcp_close — graceful shutdown (sends FIN, waits for ACK).
 * Safe to call even if the connection is already closed.
 */
void tcp_close(void);

/*
 * tcp_is_connected — non-blocking status check.
 */
bool tcp_is_connected(void);

#ifdef __cplusplus
}
#endif

#endif /* IOTIDS_TCP_CLIENT_H */
