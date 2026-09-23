#!/usr/bin/env python3
"""Ask the arm what its cameras' calibration actually is, and compare.

`kinova_vision` publishes `camera_info` from GENERIC files shipped in the
package (`launch/calibration/default_*_calib_WxH.ini`) unless you hand it
`color_camera_info_url` / `depth_camera_info_url`. The vision module itself
stores per-unit intrinsics, reachable over the Kortex API. Those are two
different sets of numbers and nothing warns you when they disagree.

There is also something odd in the shipped colour defaults worth settling: the
principal point is centred at 4:3 but not at 16:9 --

    320x240   cy/h 0.491      1280x720   cy/h 0.331
    640x480   cy/h 0.484      1920x1080  cy/h 0.090

while cx/w stays ~0.49 throughout. Either the 16:9 profiles encode an
asymmetric vertical crop, or they are wrong. This script answers that by asking
the device.

    python -m kinova.scripts.camera_intrinsics            # needs the arm's network
"""
from __future__ import annotations

import sys

from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.messages import VisionConfig_pb2, DeviceConfig_pb2

from kinova.connection import kinova_connection

# The shipped defaults, for comparison. Keep in sync with
# ros2_kortex_vision/launch/calibration/.
SHIPPED = {
    "COLOR": {(1280, 720): (1297.673, 1298.631, 620.914, 238.280),
              (640, 480): (653.682, 651.856, 311.753, 232.401),
              (1920, 1080): (2612.346, 2609.466, 946.434, 96.968)},
    "DEPTH": {(480, 270): (360.013, 360.013, 243.872, 137.922),
              (424, 240): (314.625, 314.625, 212.257, 120.856)},
}


def res_wh(name: str) -> tuple[int, int]:
    """'RESOLUTION_1280x720' -> (1280, 720)."""
    w, h = name.split("_")[-1].split("x")
    return int(w), int(h)


def vision_device_id(conn) -> int:
    """The vision module's device id. NOT 0 -- that is the arm base."""
    from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
    dm = DeviceManagerClient(conn._tcp_router)
    for d in dm.ReadAllDevices().device_handle:
        if d.device_type == DeviceConfig_pb2.VISION:
            return d.device_identifier
    raise RuntimeError("no VISION device on this arm")


def show(name: str, intr, tag: str) -> None:
    w, h = res_wh(VisionConfig_pb2.Resolution.Name(intr.resolution))
    fx, fy = intr.focal_length_x, intr.focal_length_y
    cx, cy = intr.principal_point_x, intr.principal_point_y
    d = intr.distortion_coeffs
    ship = SHIPPED.get(name, {}).get((w, h))
    flag = ""
    if ship:
        dev = max(abs(fx - ship[0]), abs(fy - ship[1]),
                  abs(cx - ship[2]), abs(cy - ship[3]))
        flag = "  <-- differs from shipped by %.2f px" % dev if dev > 1.0 else "  (matches shipped)"
    print("  %-5s %4dx%-4d fx=%9.3f fy=%9.3f cx=%8.3f cy=%8.3f  cx/w=%.3f cy/h=%.3f  k=[%.4f %.4f %.4f] p=[%.4f %.4f]%s"
          % (tag, w, h, fx, fy, cx, cy, cx / w, cy / h, d.k1, d.k2, d.k3, d.p1, d.p2, flag))
    if ship:
        print("        shipped default   fx=%9.3f fy=%9.3f cx=%8.3f cy=%8.3f  cy/h=%.3f"
              % (ship[0], ship[1], ship[2], ship[3], ship[3] / h))


def main() -> int:
    with kinova_connection() as conn:
        dev = vision_device_id(conn)
        vc = VisionConfigClient(conn._tcp_router)
        print("vision device id: %d\n" % dev)

        for name, sensor in (("COLOR", VisionConfig_pb2.SENSOR_COLOR),
                             ("DEPTH", VisionConfig_pb2.SENSOR_DEPTH)):
            print("%s -- currently configured stream" % name)
            try:
                sid = VisionConfig_pb2.SensorIdentifier(sensor=sensor)
                show(name, vc.GetIntrinsicParameters(sid, deviceId=dev), "now")
            except Exception as ex:                          # noqa: BLE001
                print("  GetIntrinsicParameters failed: %s" % ex)

            print("%s -- every resolution profile the device holds" % name)
            for rname in [v.name for v in VisionConfig_pb2._RESOLUTION.values]:
                if rname == "RESOLUTION_UNSPECIFIED":
                    continue
                try:
                    pid = VisionConfig_pb2.IntrinsicProfileIdentifier(
                        sensor=sensor,
                        resolution=VisionConfig_pb2.Resolution.Value(rname))
                    show(name, vc.GetIntrinsicParametersProfile(pid, deviceId=dev), "prof")
                except Exception:                            # noqa: BLE001, S112
                    pass          # not every sensor supports every resolution
            print()

        try:
            ext = vc.GetExtrinsicParameters(deviceId=dev)
            r, t = ext.rotation, ext.translation
            print("DEPTH -> COLOR extrinsics (factory)")
            # TranslationVector fields are t_x/t_y/t_z, not x/y/z.
            print("  t = [%+.5f %+.5f %+.5f] m   (|t| = %.1f mm)"
                  % (t.t_x, t.t_y, t.t_z,
                     1000 * (t.t_x ** 2 + t.t_y ** 2 + t.t_z ** 2) ** 0.5))
            for i, row in enumerate((r.row1, r.row2, r.row3)):
                print("  R[%d] = [%+.6f %+.6f %+.6f]" % (i, row.column1, row.column2, row.column3))
        except Exception as ex:                              # noqa: BLE001
            print("GetExtrinsicParameters failed: %s" % ex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
