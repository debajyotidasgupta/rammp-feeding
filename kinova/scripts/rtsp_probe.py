#!/usr/bin/env python3
"""Diagnose the Gen3 vision module's RTSP session handling.

Background
----------
The depth feed delivered by `kinova_vision` drops roughly every 62s (UDP) and
less regularly under TCP. It is not a network fault -- the link measures 0%
packet loss -- and it is not the camera restarting.

The arm runs a GStreamer RTSP server that advertises a **30 second** session
timeout in its SETUP reply:

    Session: <id>;timeout=30

A client must refresh the session inside that window. This probe establishes a
session by hand and refreshes it with GET_PARAMETER (which the server lists in
its OPTIONS `Public:` header) at a chosen interval. With a 10s interval the
session survives indefinitely -- measured 130s / 12 keepalives / zero drops --
while GStreamer's `rtspsrc` in the same conditions drops at ~62s, i.e. about
2x the advertised timeout.

So the defect is in the client's keepalive cadence, not the camera. This script
is the reproducer to attach to an upstream report against ros2_kortex_vision /
GStreamer rtspsrc.

    python -m kinova.scripts.rtsp_probe                    # report session timeout
    python -m kinova.scripts.rtsp_probe --hold 130         # endurance test
    python -m kinova.scripts.rtsp_probe --hold 130 --interval 40   # too slow -> expect a drop
"""

from __future__ import annotations

import argparse
import re
import socket
import sys
import time

from kinova.connection import DEFAULT_IP

DEFAULT_PORT = 554


class RtspSession:
    def __init__(self, host: str, path: str, port: int = DEFAULT_PORT, timeout: float = 10.0):
        self.base = f"rtsp://{host}{path}"
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.cseq = 0
        self.session_id: str | None = None
        self.advertised_timeout: int | None = None

    def _request(self, method: str, url: str, extra: str = "", pause: float = 0.35) -> str:
        self.cseq += 1
        self.sock.sendall(f"{method} {url} RTSP/1.0\r\nCSeq: {self.cseq}\r\n{extra}\r\n".encode())
        time.sleep(pause)
        try:
            return self.sock.recv(16384).decode(errors="replace")
        except socket.timeout:
            return "<timeout>"

    def setup(self) -> None:
        self._request("OPTIONS", self.base)
        sdp = self._request("DESCRIBE", self.base, "Accept: application/sdp\r\n")
        # Per-media control attr; the session-level 'a=control:*' is not a SETUP target.
        controls = [c for c in re.findall(r"a=control:(\S+)", sdp) if c != "*"]
        control = controls[0] if controls else "stream=0"
        reply = self._request(
            "SETUP", f"{self.base}/{control}",
            "Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n")
        header = next((l for l in reply.splitlines() if l.lower().startswith("session:")), "")
        if not header:
            raise RuntimeError("no Session header in SETUP reply:\n" + reply)
        value = header.split(":", 1)[1].strip()
        self.session_id = value.split(";")[0]
        m = re.search(r"timeout=(\d+)", value)
        self.advertised_timeout = int(m.group(1)) if m else None

    def play(self) -> str:
        return self._request("PLAY", self.base,
                             f"Session: {self.session_id}\r\nRange: npt=0.000-\r\n")

    def keepalive(self) -> None:
        self.cseq += 1
        self.sock.sendall(
            f"GET_PARAMETER {self.base} RTSP/1.0\r\nCSeq: {self.cseq}\r\n"
            f"Session: {self.session_id}\r\n\r\n".encode())

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


def main() -> int:
    p = argparse.ArgumentParser(description="Probe the Gen3 RTSP session timeout")
    p.add_argument("--ip", default=DEFAULT_IP)
    p.add_argument("--stream", default="depth", choices=("depth", "color"))
    p.add_argument("--hold", type=float, default=0.0,
                   help="seconds to hold the session open (0 = just report)")
    p.add_argument("--interval", type=float, default=10.0, help="keepalive interval, seconds")
    a = p.parse_args()

    s = RtspSession(a.ip, f"/{a.stream}")
    try:
        s.setup()
        print(f"  server session timeout : {s.advertised_timeout}s"
              if s.advertised_timeout else "  server advertises no timeout (RFC default 60s)")
        print(f"  session id             : {s.session_id}")
        if s.advertised_timeout and a.interval >= s.advertised_timeout:
            print(f"  WARNING: keepalive interval {a.interval}s >= timeout "
                  f"{s.advertised_timeout}s -- the session will expire")
        if a.hold <= 0:
            return 0

        print("  PLAY ->", s.play().splitlines()[0].strip())
        t0 = last = time.time()
        sent = 0
        s.sock.settimeout(1.0)
        while time.time() - t0 < a.hold:
            if time.time() - last >= a.interval:
                s.keepalive()
                sent += 1
                last = time.time()
                print("  t+%-4ds keepalive #%d" % (time.time() - t0, sent), flush=True)
            try:
                data = s.sock.recv(65536)
                if not data:
                    print("  t+%ds SERVER CLOSED CONNECTION" % (time.time() - t0))
                    return 1
                if b"RTSP/1.0 454" in data:
                    print("  t+%ds SESSION NOT FOUND (454) -- expired" % (time.time() - t0))
                    return 1
            except socket.timeout:
                pass
        print("  survived %.0fs with %d keepalives, no drop" % (a.hold, sent))
        return 0
    finally:
        s.close()


if __name__ == "__main__":
    raise SystemExit(main())
