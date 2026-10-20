"""Real-time receiver for the ParkiScan wrist / ankle IMU boards (ESP32 + MPU-6050, firmware v2).

Reads newline-delimited JSON from one or more USB serial ports, validates every line, tracks the state of each
placement (wrist, ankle) and streams the accepted readings to Unity over local UDP (127.0.0.1 only).

    python -m parkiscan.sensors.receiver --list                 # show serial ports
    python -m parkiscan.sensors.receiver                        # auto-detect ESP32 USB ports, stream to Unity
    python -m parkiscan.sensors.receiver --port /dev/cu.usbserial-0001 --port /dev/cu.usbserial-0002
    python -m parkiscan.sensors.receiver --check 10             # hardware check: 10 s report, exit 0 only if real data arrived
    python -m parkiscan.sensors.receiver --log-dir sessions/imu_raw     # also keep a CSV of every accepted reading

What it guarantees:
* Only lines that pass validation reach Unity: JSON object, firmware data line, schema, finite values within the
  sensor's configured range (±4 g, ±500 deg/s for firmware v2), placement "wrist"/"ankle", strictly increasing seq and
  non-decreasing device time. Duplicates and out-of-order lines are dropped and counted; gaps in seq are counted.
* Nothing is ever generated here. If a board is unplugged, the sensor is missing, or data stops, the placement's state
  says so ("not_connected", "board_no_sensor", "stale", "disconnected") and no readings are sent.
* Packets marked "sim": true (only the separate simulator produces them) are forwarded with sim=true so Unity labels
  them SIMULATED and never treats them as participant data.

UDP messages (JSON, one per datagram, to 127.0.0.1:<udp-port>, default 47811):
  {"type":"imu","placement":"wrist","dev":"esp32-wrist-a1b2c3","sim":false,
   "samples":[{"seq":1,"t_ms":100,"t_us":100123,"rx":12.345,"ax":0.01,"ay":-0.98,"az":0.12,"gx":1.2,"gy":-0.4,"gz":0.3}]}
  {"type":"status","rx":12.5,"receiver":"parkiscan-imu-receiver","version":"2",
   "sensors":[{"placement":"wrist","state":"streaming","dev":"...","port":"...","rate_hz":99.9,"age_s":0.01,...}]}
"rx" is the receiver's monotonic clock (seconds) at arrival.
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .imu import ImuPacket

PLACEMENTS = ("wrist", "ankle")
UDP_PORT = 47811
STALE_S = 0.5            # no accepted reading for this long → "stale"
HEARTBEAT_LOST_S = 3.0   # no line at all for this long → board not responding
# firmware v2 configures ±4 g and ±500 deg/s; values beyond (plus a small margin) can only be corruption
V2_ACC_LIMIT_G = 4.05
V2_GYRO_LIMIT_DPS = 505.0
ESP_PORT_HINTS = ("usbserial", "SLAB_USBtoUART", "wchusbserial", "usbmodem", "ttyUSB", "ttyACM")
ESP_USB_VIDS = {0x10C4, 0x1A86, 0x0403, 0x303A}   # Silicon Labs CP210x, WCH CH34x, FTDI, Espressif native USB

REJECT_REASONS = ("not_json", "schema", "placement", "range", "duplicate", "out_of_order", "time_backwards",
                  "duplicate_placement", "simulated_mixed")


# ---------------------------------------------------------------------------------------------- validation
@dataclass
class DeviceTrack:
    """Per-device sequence/time continuity."""
    last_seq: int | None = None
    last_t_ms: int | None = None
    gaps: int = 0          # number of missing seq values
    resets: int = 0        # device restarts detected (seq/time jumped back a lot)


class PacketValidator:
    """Stateful validator: one instance per receiver, tracks continuity per device id."""

    def __init__(self):
        self.tracks: dict[str, DeviceTrack] = {}

    def check(self, obj: dict) -> tuple[ImuPacket | None, str | None]:
        try:
            p = ImuPacket(**obj)
        except Exception:
            return None, "schema"
        if p.placement not in PLACEMENTS:
            return None, "placement"
        if p.v >= 2 and not p.sim:
            if any(abs(x) > V2_ACC_LIMIT_G for x in (p.ax, p.ay, p.az)) or \
               any(abs(x) > V2_GYRO_LIMIT_DPS for x in (p.gx, p.gy, p.gz)):
                return None, "range"
        tr = self.tracks.setdefault(p.dev, DeviceTrack())
        if tr.last_seq is not None:
            if p.seq <= tr.last_seq:
                # board restarted (seq restarts from 0 and its clock from ~0)?
                if p.seq < tr.last_seq - 50 and (tr.last_t_ms is None or p.t_ms < tr.last_t_ms):
                    tr.resets += 1
                    tr.last_seq, tr.last_t_ms = p.seq, p.t_ms
                    return p, None
                return None, "duplicate" if p.seq == tr.last_seq else "out_of_order"
            if tr.last_t_ms is not None and p.t_ms < tr.last_t_ms:
                return None, "time_backwards"
            tr.gaps += p.seq - tr.last_seq - 1
        tr.last_seq, tr.last_t_ms = p.seq, p.t_ms
        return p, None


# ---------------------------------------------------------------------------------------------- state
@dataclass
class PlacementState:
    placement: str
    state: str = "not_connected"
    dev: str = ""
    port: str = ""
    msg: str = "no board has reported this placement"
    fw: str = ""
    who_am_i: str = ""
    sim: bool = False
    accepted: int = 0
    rejected: dict = field(default_factory=lambda: {r: 0 for r in REJECT_REASONS})
    gaps: int = 0
    resets: int = 0
    last_valid: float = 0.0       # receiver monotonic time of last accepted reading
    last_line: float = 0.0        # receiver monotonic time of last line of any kind from this placement's board
    board_state: str = ""         # last firmware status state (ready, sensor_error, sensor_lost, paused, heartbeat, ...)
    port_open: bool = False
    _rate_t: list = field(default_factory=list)

    def rate_hz(self, now: float) -> float:
        self._rate_t = [t for t in self._rate_t if now - t <= 2.0]
        if len(self._rate_t) < 2:
            return 0.0
        return (len(self._rate_t) - 1) / max(1e-6, self._rate_t[-1] - self._rate_t[0])

    def refresh(self, now: float, stale_s: float = STALE_S) -> str:
        """Derive the public state from what has (and has not) arrived."""
        fresh = self.last_valid > 0 and now - self.last_valid <= stale_s
        alive = self.last_line > 0 and now - self.last_line <= HEARTBEAT_LOST_S
        if fresh:
            s = "streaming"
        elif not self.port_open and not alive:
            s = "disconnected" if self.last_line > 0 else "not_connected"
        elif self.board_state in ("sensor_error", "sensor_lost", "sensor_stuck"):
            s = "board_no_sensor"
        elif self.board_state == "paused" and alive:
            s = "paused"
        elif self.last_valid > 0:
            s = "stale"
        elif alive:
            s = "board_no_data"
        else:
            s = "not_connected"
        self.state = s
        return s

    def to_dict(self, now: float) -> dict:
        self.refresh(now)
        age = now - self.last_valid if self.last_valid > 0 else None
        return {"placement": self.placement, "state": self.state, "dev": self.dev, "port": self.port, "msg": self.msg,
                "fw": self.fw, "who_am_i": self.who_am_i, "sim": self.sim, "rate_hz": round(self.rate_hz(now), 2),
                "age_s": round(age, 3) if age is not None else -1.0, "accepted": self.accepted,
                "rejected": sum(self.rejected.values()), "rejected_by_reason": {k: v for k, v in self.rejected.items() if v},
                "gaps": self.gaps, "resets": self.resets, "board_state": self.board_state}


# ---------------------------------------------------------------------------------------------- receiver core
class Receiver:
    """Parses lines (from any transport), validates, keeps state, and hands accepted samples to sinks."""

    def __init__(self, stale_s: float = STALE_S, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.stale_s = stale_s
        self.validator = PacketValidator()
        self.states = {p: PlacementState(p) for p in PLACEMENTS}
        self.unassigned_rejects = {r: 0 for r in REJECT_REASONS}
        self.noise_lines = 0            # boot ROM text, partial lines at connect, etc.
        self.lock = threading.Lock()
        self.pending: dict[str, list[dict]] = {p: [] for p in PLACEMENTS}
        self.sinks: list[Callable[[str, ImuPacket, float], None]] = []
        self.dev_port: dict[str, str] = {}

    # -- input
    def feed_line(self, raw: str, port: str = "") -> str | None:
        """Process one line. Returns None if accepted (data or status) or the rejection reason."""
        now = self.clock()
        line = raw.strip()
        if not line:
            return "empty"
        if not line.startswith("{"):
            self.noise_lines += 1
            return "noise"
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError
        except Exception:
            # a cut-off line (port opened mid-line) is noise, not a sensor fault
            self.noise_lines += 1
            return "not_json"
        kind = obj.get("type", "imu" if "ax" in obj else "")
        placement = str(obj.get("placement", "")).strip().lower()
        with self.lock:
            st = self.states.get(placement)
            if kind == "status":
                if st is None:
                    self.unassigned_rejects["placement"] += 1
                    return "placement"
                if not self._claim(st, str(obj.get("dev", "")), port, now):
                    st.rejected["duplicate_placement"] += 1
                    return "duplicate_placement"
                st.last_line = now
                st.port_open = True
                st.board_state = str(obj.get("state", ""))
                if obj.get("msg") and obj.get("msg") != "heartbeat":
                    st.msg = str(obj.get("msg"))[:200]
                st.fw = str(obj.get("fw", st.fw))
                st.who_am_i = str(obj.get("who_am_i", st.who_am_i))
                return None
            if kind != "imu":
                self.unassigned_rejects["schema"] += 1
                return "schema"
            p, why = self.validator.check(obj)
            if p is None:
                (st.rejected if st is not None else self.unassigned_rejects)[why] += 1
                if st is not None:
                    st.last_line = now
                return why
            st = self.states[p.placement]
            if not self._claim(st, p.dev, port, now):
                st.rejected["duplicate_placement"] += 1
                return "duplicate_placement"
            if st.accepted > 0 and st.sim != p.sim and now - st.last_valid < 2.0:
                st.rejected["simulated_mixed"] += 1
                return "simulated_mixed"
            tr = self.validator.tracks[p.dev]
            st.gaps, st.resets = tr.gaps, tr.resets
            st.sim = p.sim
            st.accepted += 1
            st.last_valid = st.last_line = now
            st.port_open = True
            if st.board_state in ("", "booting", "sensor_error", "sensor_lost", "sensor_stuck"):
                st.board_state = "streaming"
            st._rate_t.append(now)
            if len(st._rate_t) > 400:
                del st._rate_t[:200]
            self.pending[p.placement].append(
                {"seq": p.seq, "t_ms": p.t_ms, "t_us": p.t_us if p.t_us is not None else -1, "rx": round(now, 4),
                 "ax": p.ax, "ay": p.ay, "az": p.az, "gx": p.gx, "gy": p.gy, "gz": p.gz})
        for s in self.sinks:
            try:
                s(p.placement, p, now)
            except Exception as e:  # a sink failure must never stop reception
                print(f"[receiver] sink error: {e}", file=sys.stderr)
        return None

    def _claim(self, st: PlacementState, dev: str, port: str, now: float) -> bool:
        """One device per placement. A second board claiming the same placement is rejected while the first is alive."""
        if not dev:
            return True
        if st.dev and st.dev != dev and now - st.last_line <= HEARTBEAT_LOST_S:
            st.msg = f"two boards claim '{st.placement}': {st.dev} and {dev} — flash the other board with the other sketch"
            return False
        if st.dev != dev:
            st.dev = dev
            st.accepted = 0
            st._rate_t.clear()
        if port:
            st.port = port
        return True

    def port_closed(self, port: str, reason: str):
        with self.lock:
            for st in self.states.values():
                if st.port == port:
                    st.port_open = False
                    st.msg = reason[:200]

    # -- output
    def take_pending(self) -> dict[str, list[dict]]:
        with self.lock:
            out = {p: v for p, v in self.pending.items() if v}
            self.pending = {p: [] for p in PLACEMENTS}
        return out

    def status(self) -> dict:
        now = self.clock()
        with self.lock:
            return {"type": "status", "rx": round(now, 4), "receiver": "parkiscan-imu-receiver", "version": "2",
                    "noise_lines": self.noise_lines,
                    "sensors": [self.states[p].to_dict(now) for p in PLACEMENTS]}


# ---------------------------------------------------------------------------------------------- transports
class SerialLink(threading.Thread):
    """Reads one serial port, reconnecting when the board is unplugged and plugged back in."""

    def __init__(self, port: str, rx: Receiver, baud: int = 921600, stop: threading.Event | None = None,
                 opener: Callable | None = None):
        super().__init__(daemon=True, name=f"serial:{port}")
        self.port, self.rx, self.baud = port, rx, baud
        self.stop_evt = stop or threading.Event()
        self.opener = opener
        self.connected = False
        self.last_error = ""

    def _open(self):
        if self.opener is not None:
            return self.opener(self.port, self.baud)
        import serial  # pyserial
        s = serial.Serial()
        s.port, s.baudrate, s.timeout = self.port, self.baud, 0.2
        # do not toggle DTR/RTS on open more than necessary (CP2102 auto-reset); the firmware boots in <1 s anyway
        s.dtr = False
        s.rts = False
        s.open()
        return s

    def run(self):
        buf = b""
        while not self.stop_evt.is_set():
            try:
                ser = self._open()
            except Exception as e:
                self.connected = False
                self.last_error = f"cannot open {self.port}: {e}"
                self.rx.port_closed(self.port, self.last_error)
                self.stop_evt.wait(2.0)
                continue
            self.connected = True
            print(f"[receiver] opened {self.port} @ {self.baud}")
            try:
                while not self.stop_evt.is_set():
                    chunk = ser.read(4096)
                    if not chunk:
                        continue
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        self.rx.feed_line(line.decode("utf-8", errors="replace"), self.port)
                    if len(buf) > 8192:      # no newline for 8 KB: wrong baud rate or binary garbage
                        buf = b""
                        self.rx.noise_lines += 1
            except Exception as e:
                self.last_error = f"{self.port} disconnected: {e}"
                print(f"[receiver] {self.last_error}")
            finally:
                self.connected = False
                buf = b""
                try:
                    ser.close()
                except Exception:
                    pass
                self.rx.port_closed(self.port, self.last_error or f"{self.port} closed")
            self.stop_evt.wait(1.0)


class UdpPublisher:
    """Sends accepted readings and status to Unity on 127.0.0.1 (never off this computer)."""

    MAX_SAMPLES_PER_DATAGRAM = 40

    def __init__(self, port: int = UDP_PORT, host: str = "127.0.0.1"):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("UDP output is restricted to this computer (127.0.0.1)")
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0

    def send(self, obj: dict):
        data = json.dumps(obj, separators=(",", ":"), allow_nan=False).encode()
        try:
            self.sock.sendto(data, self.addr)
            self.sent += 1
        except OSError:
            pass   # Unity not listening: fine, nothing queues up

    def publish(self, rx: Receiver):
        for placement, samples in rx.take_pending().items():
            st = rx.states[placement]
            for i in range(0, len(samples), self.MAX_SAMPLES_PER_DATAGRAM):
                self.send({"type": "imu", "placement": placement, "dev": st.dev, "sim": st.sim,
                           "samples": samples[i:i + self.MAX_SAMPLES_PER_DATAGRAM]})


class CsvLog:
    """Optional audit log of every accepted reading (one CSV per placement)."""

    COLS = ["rx_monotonic", "rx_unix", "dev", "placement", "seq", "t_ms", "t_us", "ax", "ay", "az", "gx", "gy", "gz", "sim"]

    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.files, self.writers = {}, {}
        for p in PLACEMENTS:
            f = (folder / f"imu_{p}_{stamp}.csv").open("w", newline="")
            w = csv.writer(f)
            w.writerow(self.COLS)
            self.files[p], self.writers[p] = f, w
        self.offset = time.time() - time.monotonic()

    def __call__(self, placement: str, p: ImuPacket, now: float):
        self.writers[placement].writerow([f"{now:.4f}", f"{now + self.offset:.4f}", p.dev, placement, p.seq, p.t_ms,
                                          p.t_us if p.t_us is not None else "", p.ax, p.ay, p.az, p.gx, p.gy, p.gz, int(p.sim)])

    def flush(self):
        for f in self.files.values():
            f.flush()

    def close(self):
        for f in self.files.values():
            f.close()


class ServiceForwarder:
    """Optional: POST accepted readings to the local ParkiScanML service for a session (/sensors/imu).
    Do not use together with Unity's own forwarding for the same session (Unity forwards when it is running)."""

    def __init__(self, session: str, base: str = "http://127.0.0.1:8765"):
        self.session, self.base = session, base.rstrip("/")
        self.q: "queue.Queue[dict]" = queue.Queue(maxsize=20000)
        threading.Thread(target=self._pump, daemon=True).start()

    def __call__(self, placement: str, p: ImuPacket, now: float):
        try:
            self.q.put_nowait(p.model_dump())
        except queue.Full:
            pass

    def _pump(self):
        import httpx
        with httpx.Client(timeout=5) as c:
            while True:
                batch = [self.q.get()]
                while len(batch) < 500:
                    try:
                        batch.append(self.q.get_nowait())
                    except queue.Empty:
                        break
                try:
                    c.post(f"{self.base}/sensors/imu", json={"session_id": self.session, "packets": batch}).raise_for_status()
                except Exception as e:
                    print(f"[receiver] service forward failed: {e}", file=sys.stderr)
                    time.sleep(1.0)
                time.sleep(0.5)


# ---------------------------------------------------------------------------------------------- port discovery
def list_ports() -> list[dict]:
    from serial.tools import list_ports as lp
    out = []
    for p in lp.comports():
        out.append({"device": p.device, "vid": p.vid, "pid": p.pid, "description": p.description or "",
                    "manufacturer": p.manufacturer or "", "likely_esp32": _likely_esp(p.device, p.vid)})
    return out


def _likely_esp(device: str, vid: int | None) -> bool:
    if vid in ESP_USB_VIDS:
        return True
    return any(h in device for h in ESP_PORT_HINTS)


def discover_ports() -> list[str]:
    ports = [p["device"] for p in list_ports() if p["likely_esp32"]]
    # on macOS each USB serial device appears as /dev/cu.* and /dev/tty.*; use the cu.* (call-out) device only
    if sys.platform == "darwin":
        ports = [p for p in ports if not p.startswith("/dev/tty.")]
    return sorted(set(ports))


# ---------------------------------------------------------------------------------------------- main loop
def run(ports: Iterable[str] | None, udp_port: int = UDP_PORT, baud: int = 921600, log_dir: str | None = None,
        session: str | None = None, service: str = "http://127.0.0.1:8765", check_s: float | None = None,
        stale_s: float = STALE_S, rescan: bool = True) -> int:
    rx = Receiver(stale_s=stale_s)
    pub = UdpPublisher(udp_port)
    log = CsvLog(Path(log_dir)) if log_dir else None
    if log:
        rx.sinks.append(log)
    if session:
        rx.sinks.append(ServiceForwarder(session, service))
    stop = threading.Event()
    links: dict[str, SerialLink] = {}
    fixed = list(ports or [])

    def ensure_links():
        want = fixed or discover_ports()
        for p in want:
            if p not in links:
                links[p] = SerialLink(p, rx, baud, stop)
                links[p].start()

    ensure_links()
    if not links:
        print("[receiver] no ESP32 USB serial port found yet — plug a board in (waiting; Ctrl+C to stop)")
    t0 = time.monotonic()
    last_status = last_scan = last_print = 0.0
    try:
        while True:
            now = time.monotonic()
            pub.publish(rx)
            if now - last_status >= 0.25:
                pub.send(rx.status())
                last_status = now
            if rescan and not fixed and now - last_scan >= 2.0:
                ensure_links()
                last_scan = now
            if now - last_print >= (1.0 if check_s else 5.0):
                last_print = now
                print(_summary_line(rx.status()))
                if log:
                    log.flush()
            if check_s is not None and now - t0 >= check_s:
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[receiver] stopped")
    finally:
        stop.set()
        if log:
            log.close()
    if check_s is not None:
        return _check_report(rx)
    return 0


def _summary_line(status: dict) -> str:
    parts = []
    for s in status["sensors"]:
        parts.append(f"{s['placement']}: {s['state']}" + (f" {s['rate_hz']:.0f} Hz ok={s['accepted']} rej={s['rejected']} gaps={s['gaps']}"
                                                          if s["accepted"] else "") + (" SIMULATED" if s["sim"] else ""))
    return "[receiver] " + " | ".join(parts) + f" | noise lines={status['noise_lines']}"


def _check_report(rx: Receiver) -> int:
    """Hardware check verdict: real data must have arrived from at least one placement."""
    st = rx.status()
    print("\n=== ParkiScan IMU hardware check ===")
    ok_any = False
    for s in st["sensors"]:
        real = s["accepted"] > 0 and not s["sim"]
        ok_any |= real and s["state"] == "streaming"
        print(f"{s['placement']:>6}: state={s['state']:<16} dev={s['dev'] or '-':<24} port={s['port'] or '-'}")
        print(f"        accepted={s['accepted']} rejected={s['rejected']} {s['rejected_by_reason'] or ''} gaps={s['gaps']} "
              f"rate={s['rate_hz']} Hz fw={s['fw'] or '-'} who_am_i={s['who_am_i'] or '-'}")
        print(f"        last message: {s['msg']}")
    print("RESULT:", "REAL SENSOR DATA RECEIVED" if ok_any else "NO REAL SENSOR DATA — see states above")
    return 0 if ok_any else 2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", action="append", help="serial port (repeat for wrist and ankle). Default: auto-detect")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--udp-port", type=int, default=UDP_PORT)
    ap.add_argument("--log-dir", help="also write every accepted reading to CSV files here")
    ap.add_argument("--session", help="also POST readings to the local service for this session id (not needed when Unity runs)")
    ap.add_argument("--service", default="http://127.0.0.1:8765")
    ap.add_argument("--stale", type=float, default=STALE_S, help="seconds without data before a sensor is 'stale'")
    ap.add_argument("--check", type=float, metavar="SECONDS", help="run for N seconds, print a verdict, exit 0 only if real data arrived")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    a = ap.parse_args(argv)
    if a.list:
        for p in list_ports():
            print(f"{p['device']:<32} vid={hex(p['vid']) if p['vid'] else '-':<7} {p['description']} "
                  f"{'<- likely ESP32 USB' if p['likely_esp32'] else ''}")
        return 0
    return run(a.port, a.udp_port, a.baud, a.log_dir, a.session, a.service, a.check, a.stale)


if __name__ == "__main__":
    sys.exit(main())
