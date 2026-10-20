"""Software-only check of the receiver → Unity path (NO hardware involved).

Feeds firmware-format lines that are explicitly marked SIMULATED ("sim": true, device ids "SIM-SELFTEST-…") through
the real Receiver validation code and UDP publisher, so you can confirm that Unity shows the status strip, records
sensors/imu_*.csv and labels everything SIMULATED. It proves the software path only — it says nothing about whether
a physical sensor works. Unity must be in Play mode.

    python -m parkiscan.sensors.pipeline_selftest --seconds 15
    python -m parkiscan.sensors.pipeline_selftest --seconds 15 --only wrist      # leave the ankle "not connected"
"""
from __future__ import annotations

import argparse
import json
import math
import time

from .receiver import PLACEMENTS, UDP_PORT, Receiver, UdpPublisher


def make_line(placement: str, seq: int, t_ms: int) -> str:
    t = t_ms / 1000.0
    # a smooth, obviously artificial test waveform — labelled sim:true so it can never pass as a measurement
    return json.dumps({"type": "imu", "v": 2, "dev": f"SIM-SELFTEST-{placement}", "placement": placement, "seq": seq,
                       "t_ms": t_ms, "t_us": t_ms * 1000, "ax": round(0.2 * math.sin(2 * math.pi * 1.0 * t), 4),
                       "ay": -1.0, "az": 0.0, "gx": round(30 * math.sin(2 * math.pi * 0.5 * t), 3), "gy": 0.0, "gz": 0.0,
                       "sim": True})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--udp-port", type=int, default=UDP_PORT)
    ap.add_argument("--only", choices=PLACEMENTS)
    a = ap.parse_args(argv)
    rx, pub = Receiver(), UdpPublisher(a.udp_port)
    places = [a.only] if a.only else list(PLACEMENTS)
    t0 = time.monotonic()
    seq = 0
    last_status = 0.0
    print(f"[selftest] sending SIMULATED test packets for {a.seconds:.0f} s to 127.0.0.1:{a.udp_port} ({', '.join(places)})")
    while time.monotonic() - t0 < a.seconds:
        t_ms = int((time.monotonic() - t0) * 1000)
        for p in places:
            rx.feed_line(make_line(p, seq, t_ms), port="selftest")
        seq += 1
        pub.publish(rx)
        if time.monotonic() - last_status > 0.25:
            pub.send(rx.status())
            last_status = time.monotonic()
        time.sleep(0.01)
    st = rx.status()
    for s in st["sensors"]:
        print(f"[selftest] {s['placement']}: state={s['state']} accepted={s['accepted']} sim={s['sim']}")
    print(f"[selftest] datagrams sent: {pub.sent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
