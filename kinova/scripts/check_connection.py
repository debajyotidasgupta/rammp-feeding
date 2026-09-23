#!/usr/bin/env python3
"""Connect to the arm and report identity, firmware and live joint state.

Read-only: it never commands motion. Run it first after any network, firmware
or dependency change to confirm the whole stack end to end.

    python -m kinova.scripts.check_connection [--ip 192.168.2.67] [--real-time]
"""

from __future__ import annotations

import argparse
import sys

from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import ProductConfiguration_pb2 as pc

from kinova.connection import (
    DEFAULT_IP,
    DEFAULT_PASSWORD,
    DEFAULT_USERNAME,
    ConnectionConfig,
    KinovaConnection,
)


def _probe(obj, method: str, field: str | None = None):
    """Call one RPC by name and optionally pull a single field off the reply.

    Firmware revisions differ in which RPCs they expose, so resolve the
    attribute inside the guard: a missing one must not abort the whole report.
    """
    try:
        reply = getattr(obj, method)()
    except Exception as exc:  # noqa: BLE001 - diagnostic breadth is the point
        return None, f"{type(exc).__name__}: {exc}"
    return (getattr(reply, field) if field else reply), None


def _fmt_version(raw: int) -> str:
    """Kortex packs versions into a uint32, one octet per component."""
    octets = [(raw >> shift) & 0xFF for shift in (24, 16, 8, 0)]
    return f"{'.'.join(str(o) for o in octets)}  (0x{raw:08X})"


def _enum(module, enum_name: str, value: int) -> str:
    """Enums are module-level in the generated protobuf, not nested on the
    message. Fall back to the raw int if this firmware reports an unknown one."""
    try:
        return getattr(module, enum_name).Name(value)
    except Exception:  # noqa: BLE001
        return f"<{value}>"


def _fmt_mac(raw: bytes | str) -> str:
    data = raw.encode("latin-1") if isinstance(raw, str) else bytes(raw)
    return ":".join(f"{b:02X}" for b in data)


def _row(label: str, value, err: str | None, fmt=None) -> None:
    if err is not None:
        print(f"  {label:<14} (unavailable: {err})")
    else:
        print(f"  {label:<14} {fmt(value) if fmt else value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Kinova Gen3 connection check")
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--real-time", action="store_true",
                        help="also open the UDP cyclic channel on :10001")
    args = parser.parse_args()

    config = ConnectionConfig(ip=args.ip, username=args.username, password=args.password)
    print(f"Connecting to {config.ip}:10000 as {config.username!r} ...")

    try:
        conn = KinovaConnection(config, real_time=args.real_time).connect()
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("\nCheck: arm powered on, IP reachable, credentials correct, and "
              "that no other client already holds a session.", file=sys.stderr)
        return 1

    try:
        print("  session established\n")
        dev, base = conn.device_config, conn.base

        print("-- device --")
        _row("serial", *_probe(dev, "GetSerialNumber", "serial_number"))
        _row("model", *_probe(dev, "GetModelNumber", "model_number"))
        _row("part number", *_probe(dev, "GetPartNumber", "part_number"))
        _row("firmware", *_probe(dev, "GetFirmwareVersion", "firmware_version"), fmt=_fmt_version)
        _row("bootloader", *_probe(dev, "GetBootloaderVersion", "bootloader_version"), fmt=_fmt_version)
        _row("MAC", *_probe(dev, "GetMACAddress", "mac_address"), fmt=_fmt_mac)

        print("\n-- configuration --")
        cfg, err = _probe(base, "GetProductConfiguration")
        if err:
            print(f"  (unavailable: {err})")
        else:
            _row("kinematics", cfg.kin, None)
            _row("DoF", cfg.degree_of_freedom, None)
            _row("wrist", _enum(pc, "WristType", cfg.wrist_type), None)
            _row("end effector",
                 _enum(pc, "EndEffectorType", cfg.end_effector_type), None)
            _row("vision", _enum(pc, "VisionModuleType", cfg.vision_module_type), None)
            _row("base type", _enum(pc, "BaseType", cfg.base_type), None)

        print("\n-- state --")
        state, err = _probe(base, "GetArmState")
        _row("arm state", _enum(pc, "ArmState", state.active_state) if not err else None, err)
        mode, err = _probe(base, "GetServoingMode")
        _row("servoing", _enum(Base_pb2, "ServoingMode", mode.servoing_mode) if not err else None, err)
        _row("actuators", *_probe(base, "GetActuatorCount", "count"))

        print("\n-- live feedback --")
        fb, err = _probe(conn.base_cyclic, "RefreshFeedback")
        if err:
            print(f"  (unavailable: {err})")
        else:
            print("  joint angles (deg): " + " ".join(f"{a.position:8.2f}" for a in fb.actuators))
            print("  joint torques (Nm): " + " ".join(f"{a.torque:8.2f}" for a in fb.actuators))
            b = fb.base
            print(f"  tool pose:          x={b.tool_pose_x:7.3f} y={b.tool_pose_y:7.3f} "
                  f"z={b.tool_pose_z:7.3f} m   "
                  f"rx={b.tool_pose_theta_x:7.2f} ry={b.tool_pose_theta_y:7.2f} "
                  f"rz={b.tool_pose_theta_z:7.2f} deg")
            print(f"  power:              {b.arm_voltage:.1f} V  {b.arm_current:.2f} A   "
                  f"cpu {b.temperature_cpu:.1f} C")
            if b.fault_bank_a | b.fault_bank_b:
                faults = (f"bank_a=0x{b.fault_bank_a:08X} "
                          f"bank_b=0x{b.fault_bank_b:08X}  <-- inspect")
            else:
                faults = "CLEAR"
            print(f"  faults:             {faults}")

        if args.real_time:
            print("\n-- real-time channel (UDP :10001) --")
            rt, err = _probe(conn.base_cyclic_rt, "RefreshFeedback")
            _row("cyclic feedback", f"OK, {len(rt.actuators)} actuators" if not err else None, err)

        print("\nConnection check complete.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
