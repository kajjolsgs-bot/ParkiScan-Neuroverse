"""Receiver tests. All lines below are hand-written TEST FIXTURES in the firmware's format — they exercise parsing,
validation and state handling only and say nothing about real hardware."""
import json
import os
import socket
import threading
import time

import pytest

from parkiscan.sensors.receiver import (PacketValidator, Receiver, SerialLink, UdpPublisher, _check_report)


def line(placement="wrist", seq=0, t_ms=None, dev=None, **kw):
    d = {"type": "imu", "v": 2, "dev": dev or f"esp32-{placement}-abc123", "placement": placement, "seq": seq,
         "t_ms": 1000 + seq * 10 if t_ms is None else t_ms, "t_us": 1_000_000 + seq * 10_000,
         "ax": 0.01, "ay": -0.98, "az": 0.12, "gx": 1.2, "gy": -0.4, "gz": 0.3, "sim": False}
    d.update(kw)
    return json.dumps(d)


def status(placement="wrist", state="ready", dev=None, msg="x"):
    return json.dumps({"type": "status", "v": 2, "dev": dev or f"esp32-{placement}-abc123", "placement": placement,
                       "state": state, "msg": msg, "fw": "2.0.0", "who_am_i": "0x68", "t_ms": 5})


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_accepts_valid_and_counts_gaps():
    rx = Receiver(clock=Clock())
    assert rx.feed_line(line(seq=0)) is None
    assert rx.feed_line(line(seq=1)) is None
    assert rx.feed_line(line(seq=4)) is None          # 2 and 3 lost
    s = rx.status()["sensors"][0]
    assert s["accepted"] == 3 and s["gaps"] == 2 and s["state"] == "streaming"


@pytest.mark.parametrize("bad,reason", [
    ("not json at all", "noise"),
    ('{"type":"imu","dev":"x"', "not_json"),
    (line(ax="nan"), "schema"),
    (line(ax=9.0), "range"),                    # beyond the ±4 g the v2 firmware configures
    (line(gz=900.0), "range"),
    (line(placement="elbow"), "placement"),
    (json.dumps({"type": "imu", "placement": "wrist"}), "schema"),
])
def test_rejects_invalid(bad, reason):
    rx = Receiver(clock=Clock())
    assert rx.feed_line(bad) == reason
    assert rx.status()["sensors"][0]["accepted"] == 0
    assert not rx.take_pending()


def test_duplicates_out_of_order_and_time_backwards():
    rx = Receiver(clock=Clock())
    rx.feed_line(line(seq=10))
    assert rx.feed_line(line(seq=10)) == "duplicate"
    assert rx.feed_line(line(seq=9)) == "out_of_order"
    assert rx.feed_line(line(seq=11, t_ms=50 + 1000 + 9 * 10 - 200)) == "time_backwards"


def test_device_reboot_is_accepted_as_reset():
    v = PacketValidator()
    for s in range(100, 160):
        v.check(json.loads(line(seq=s)))
    p, why = v.check(json.loads(line(seq=0, t_ms=5)))
    assert p is not None and why is None and v.tracks["esp32-wrist-abc123"].resets == 1


def test_states_stale_no_sensor_disconnected():
    clk = Clock()
    rx = Receiver(clock=clk)
    assert rx.status()["sensors"][1]["state"] == "not_connected"      # ankle never seen
    rx.feed_line(line(seq=0), port="/dev/cu.A")
    assert rx.status()["sensors"][0]["state"] == "streaming"
    clk.t += 1.0                                                     # no data for 1 s
    assert rx.status()["sensors"][0]["state"] == "stale"
    rx.feed_line(status(state="sensor_lost", msg="20 consecutive I2C read failures"), port="/dev/cu.A")
    s = rx.status()["sensors"][0]
    assert s["state"] == "board_no_sensor" and "I2C" in s["msg"]
    rx.port_closed("/dev/cu.A", "unplugged")
    clk.t += 5.0
    assert rx.status()["sensors"][0]["state"] == "disconnected"


def test_board_alive_but_sensor_missing_never_streams():
    rx = Receiver(clock=Clock())
    rx.feed_line(status(placement="ankle", state="sensor_error", msg="No device answers at I2C 0x68 or 0x69"), port="/dev/cu.B")
    s = rx.status()["sensors"][1]
    assert s["state"] == "board_no_sensor" and s["accepted"] == 0
    assert not rx.take_pending()


def test_second_board_with_same_placement_is_rejected():
    rx = Receiver(clock=Clock())
    rx.feed_line(line(seq=0, dev="esp32-wrist-aaaaaa"))
    assert rx.feed_line(line(seq=0, dev="esp32-wrist-bbbbbb")) == "duplicate_placement"
    assert "two boards" in rx.status()["sensors"][0]["msg"]


def test_simulated_packets_stay_labelled():
    rx = Receiver(clock=Clock())
    rx.feed_line(line(seq=0, sim=True, dev="SIM-wrist"))
    s = rx.status()["sensors"][0]
    assert s["sim"] is True
    assert _check_report(rx) == 2          # simulated data never passes the hardware check


def test_udp_publish_roundtrip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2)
    port = sock.getsockname()[1]
    rx = Receiver(clock=Clock())
    for s in range(3):
        rx.feed_line(line(placement="ankle", seq=s))
    pub = UdpPublisher(port)
    pub.publish(rx)
    pub.send(rx.status())
    msgs = [json.loads(sock.recv(65535)) for _ in range(2)]
    imu = next(m for m in msgs if m["type"] == "imu")
    st = next(m for m in msgs if m["type"] == "status")
    assert imu["placement"] == "ankle" and [x["seq"] for x in imu["samples"]] == [0, 1, 2]
    assert {s["placement"]: s["state"] for s in st["sensors"]} == {"wrist": "not_connected", "ankle": "streaming"}
    with pytest.raises(ValueError):
        UdpPublisher(port, host="192.168.1.5")       # never off this computer


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="needs a pseudo-terminal")
def test_serial_link_over_pty_reconnects():
    """End-to-end over a pseudo-terminal (stands in for the USB serial port): partial line, noise, valid data."""
    import serial
    master, slave = os.openpty()
    name = os.ttyname(slave)
    rx = Receiver()
    stop = threading.Event()
    link = SerialLink(name, rx, baud=115200, stop=stop, opener=lambda p, b: serial.Serial(p, b, timeout=0.1))
    link.start()
    time.sleep(0.3)
    os.write(master, b'0.12,"gx":1}\nets Jul 29 2019 boot\n')       # cut-off line + boot ROM text
    os.write(master, (status() + "\n").encode())
    for s in range(20):
        os.write(master, (line(seq=s) + "\n").encode())
    deadline = time.time() + 3
    while time.time() < deadline and rx.status()["sensors"][0]["accepted"] < 20:
        time.sleep(0.05)
    stop.set()
    s = rx.status()["sensors"][0]
    assert s["accepted"] == 20 and s["state"] == "streaming" and rx.noise_lines >= 2
    os.close(master)
    os.close(slave)


def test_firmware_sketches_differ_only_in_placement():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "firmware"
    w = (root / "ParkiScan_IMU_Wrist" / "ParkiScan_IMU_Wrist.ino").read_text().splitlines()
    a = (root / "ParkiScan_IMU_Ankle" / "ParkiScan_IMU_Ankle.ino").read_text().splitlines()
    diff = [(x, y) for x, y in zip(w, a) if x != y]
    assert len(w) == len(a) and len(diff) == 2
    assert '#define SENSOR_PLACEMENT "wrist"' in diff[1][0] and '#define SENSOR_PLACEMENT "ankle"' in diff[1][1]
    src = "\n".join(w)
    assert "random(" not in src and "sim\\\":true" not in src     # firmware never invents values
