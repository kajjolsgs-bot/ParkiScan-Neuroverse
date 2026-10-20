"""Read JSON-line IMU packets from an ESP32 over USB serial and forward them to the local service.

Usage (macOS):  python -m parkiscan.sensors.serial_bridge --port /dev/cu.usbserial-0001 --session <id>
List ports:     python -m serial.tools.list_ports
Batches of up to 50 packets are POSTed to http://127.0.0.1:8765/sensors/imu every 0.5 s.
"""
from __future__ import annotations

import argparse
import time

import httpx

from .imu import parse_line


def run(port: str, session: str, base: str = "http://127.0.0.1:8765", baud: int = 921600, batch: int = 50):
    import serial  # pyserial

    sent = bad = 0
    buf = []
    last = time.time()
    with serial.Serial(port, baud, timeout=0.2) as ser, httpx.Client(timeout=5) as client:
        print(f"[serial_bridge] reading {port} @ {baud} → {base}/sensors/imu (Ctrl+C to stop)")
        while True:
            raw = ser.readline().decode("utf-8", errors="ignore")
            p = parse_line(raw)
            if p is None:
                if raw.strip():
                    bad += 1
            else:
                buf.append(p.model_dump())
            if buf and (len(buf) >= batch or time.time() - last > 0.5):
                try:
                    r = client.post(f"{base}/sensors/imu", json={"session_id": session, "packets": buf})
                    r.raise_for_status()
                    sent += len(buf)
                except httpx.HTTPError as e:
                    print(f"[serial_bridge] post failed ({e}); keeping {len(buf)} packets for retry")
                    time.sleep(1)
                    continue
                buf.clear()
                last = time.time()
                if sent % 1000 < batch:
                    print(f"[serial_bridge] sent={sent} malformed={bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--baud", type=int, default=921600)
    a = ap.parse_args()
    try:
        run(a.port, a.session, a.base, a.baud)
    except KeyboardInterrupt:
        print("\n[serial_bridge] stopped")


if __name__ == "__main__":
    main()
