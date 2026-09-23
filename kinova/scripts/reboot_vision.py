#!/usr/bin/env python3
"""Reboot the arm's vision module and wait for its RTSP server to come back.

Why this exists: the Gen3 vision module can wedge -- it keeps accepting TCP on
:554 while no longer answering RTSP at all, so a port check reports it healthy
while both /color and /depth are dead and kinova_vision sits in an endless
"Failed to start stream (timeout)" retry loop. The only fix is a module reboot.

This reboots ONLY the vision device. The arm is not moved and the base
(device 0) is never touched -- the device type is asserted before sending.

    python -m kinova.scripts.reboot_vision [--ip 192.168.2.67] [--yes]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import Common_pb2, DeviceConfig_pb2

from kinova.connection import DEFAULT_IP, ConnectionConfig, KinovaConnection


def rtsp_responds(ip: str, endpoint: str, timeout_s: int = 8) -> bool:
    """True only if ffprobe gets a real stream description back."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-rtsp_transport", "tcp",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0",
             f"rtsp://{ip}/{endpoint}"],
            capture_output=True, text=True, timeout=timeout_s)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    # A refused/hung server writes to stderr and leaves stdout empty.
    return out.returncode == 0 and out.stdout.strip() != ""


def main() -> int:
    p = argparse.ArgumentParser(description="Reboot the Gen3 vision module")
    p.add_argument("--ip", default=DEFAULT_IP)
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--wait", type=int, default=120, help="seconds to wait for recovery")
    args = p.parse_args()

    with KinovaConnection(ConnectionConfig(ip=args.ip)) as conn:
        devices = DeviceManagerClient(conn._tcp_router).ReadAllDevices().device_handle
        vision = [d for d in devices
                  if Common_pb2.DeviceTypes.Name(d.device_type) == "VISION"]
        if not vision:
            print("No VISION device found on this arm.", file=sys.stderr)
            return 1
        if len(vision) > 1:
            print(f"Refusing: {len(vision)} VISION devices found.", file=sys.stderr)
            return 1

        handle = vision[0]
        dev_id = handle.device_identifier
        # Belt and braces: device 0 is the BASE. Rebooting it reboots the arm.
        assert Common_pb2.DeviceTypes.Name(handle.device_type) == "VISION"
        assert dev_id != 0, "refusing to reboot device 0 (BASE)"

        print(f"Vision module: device_identifier={dev_id} on {args.ip}")
        if not args.yes:
            if input("Reboot it? the camera drops for ~30s [y/N] ").strip().lower() != "y":
                print("aborted")
                return 1

        req = DeviceConfig_pb2.RebootRqst()
        req.delay = 0
        DeviceConfigClient(conn._tcp_router).RebootRequest(req, deviceId=dev_id)
        print("  reboot sent")

    print(f"  waiting up to {args.wait}s for RTSP to answer ...")
    t0 = time.time()
    while time.time() - t0 < args.wait:
        if rtsp_responds(args.ip, "depth"):
            print("  /depth responds after %.0fs" % (time.time() - t0))
            for ep in ("color", "depth"):
                print(f"    /{ep:<6} {'OK' if rtsp_responds(args.ip, ep) else 'still down'}")
            print("\nVision module is back. Restart any RTSP clients "
                  "(kinova_vision, ffplay, HLS) -- they do not recover on their own.")
            return 0
        time.sleep(3)

    print(f"  still down after {args.wait}s -- may need a full arm power cycle.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
