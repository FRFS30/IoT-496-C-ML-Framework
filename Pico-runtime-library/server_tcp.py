#This file host the local network TCP server that will send the data to the Pico microcontroller
import socket
import os
import time

HOST = "0.0.0.0"
PORT = 5001
FILE = "samples.bin"

CHUNK_SIZE = 4096

MAX_BYTES = 10 * 97 #debugging


def main():

    if not os.path.exists(FILE):
        print("Binary file not found!")
        return
    filesize = os.path.getsize(FILE)
    print(f"File size: {filesize} bytes")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server.bind((HOST, PORT))
    server.listen(1)

    print("Waiting for Pico connection...")
    conn, addr = server.accept()
    print("Connected:", addr)

    total_sent = 0

    with open(FILE, "rb") as f:

        while True:

            data = f.read(CHUNK_SIZE)

            if not data:
                break

            conn.sendall(data)
            time.sleep(0.005)

            total_sent += len(data)
            #if total_sent >= MAX_BYTES:
            #    break

    print("Finished sending.")
    print("Bytes sent:", total_sent)

    conn.shutdown(socket.SHUT_WR)

    conn.close()
    server.close()


if __name__ == "__main__":
    main()