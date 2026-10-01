"""Procedural textures and meshes for the twin scene: made on first use into out/sim/cache (nothing binary in git).

    from sim.procedural import cache_dir, wood_png, floor_png, rounded_cube_obj, torus_obj
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent


def cache_dir() -> Path:
    d = HERE.parent / "out" / "sim" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _noise(shape: tuple[int, int], scale: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth value noise in [0, 1]: a coarse random grid, bicubic-upsampled."""
    h, w = shape
    g = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 255


def _streaks(shape: tuple[int, int], along: int, across: int, rng: np.random.Generator) -> np.ndarray:
    """Noise stretched along u: long thin streaks (wood grain)."""
    h, w = shape
    g = rng.random((max(2, h // across), max(2, w // along))).astype(np.float32)
    return np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 255


def wood_png(name: str = "walnut.png", size: int = 1024, seed: int = 3) -> Path:
    """Dark walnut planks: fine straight grain along u, a few soft figure bands, a seam and a tone per plank."""
    out = cache_dir() / name
    if out.exists():
        return out
    rng = np.random.default_rng(seed)
    h = w = size
    v = np.mgrid[0:h, 0:w][0].astype(np.float32)
    fine = _streaks((h, w), 90, 2, rng)
    mid = _streaks((h, w), 260, 7, rng)
    figure = _noise((h, w), 180, rng)
    t = (0.45 * fine + 0.35 * mid + 0.2 * figure)[..., None]
    plank = (v // (h / 4)).astype(int)
    tone = np.array([0.92, 1.0, 0.86, 0.96])[plank % 4][..., None]
    light, dark = np.array([0.36, 0.22, 0.13]), np.array([0.16, 0.09, 0.05])
    rgb = (dark + (light - dark) * t ** 1.3) * tone
    rgb[(v % (h / 4)) < 2] *= 0.5
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(out)
    return out


def floor_png(name: str = "concrete.png", size: int = 1024, seed: int = 7) -> Path:
    """Polished dark concrete with faint large tiles."""
    out = cache_dir() / name
    if out.exists():
        return out
    rng = np.random.default_rng(seed)
    n = 0.55 * _noise((size, size), 128, rng) + 0.3 * _noise((size, size), 24, rng) + 0.15 * _noise((size, size), 3, rng)
    g = 0.10 + 0.06 * n
    v, u = np.mgrid[0:size, 0:size]
    tile = (u % (size // 2) < 2) | (v % (size // 2) < 2)
    g[tile] *= 0.6
    rgb = np.stack([g * 0.96, g, g * 1.06], axis=-1)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(out)
    return out


def fabric_png(name: str = "fabric.png", size: int = 256, seed: int = 11) -> Path:
    """A fine woven texture (the wheelchair's seat), mid grey: the material's rgba tints it."""
    out = cache_dir() / name
    if out.exists():
        return out
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:size, 0:size]
    weave = 0.5 + 0.25 * np.sin(u * 1.6) * np.sign(np.sin(v * 0.4)) + 0.25 * np.sin(v * 1.6) * np.sign(np.sin(u * 0.4))
    g = 0.55 + 0.25 * weave + 0.1 * _noise((size, size), 4, rng)
    Image.fromarray((np.clip(np.stack([g] * 3, -1), 0, 1) * 255).astype(np.uint8)).save(out)
    return out


def _write_obj(path: Path, v: np.ndarray, f: np.ndarray) -> Path:
    with path.open("w") as fh:
        fh.writelines("v %.5f %.5f %.5f\n" % tuple(p) for p in v)
        fh.writelines("f %d %d %d\n" % tuple(t + 1) for t in f)
    return path


def _lathe(profile: np.ndarray, n: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """A surface of revolution about z from a (r, z) polyline (closed around; the ends are not capped)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = np.array([[r * np.cos(t), r * np.sin(t), z] for r, z in profile for t in a])
    f = []
    for i in range(len(profile) - 1):
        for j in range(n):
            p, q = i * n + j, i * n + (j + 1) % n
            f += [[p, q, q + n], [p, q + n, p + n]]
    return v, np.array(f)


def rounded_cube_obj(bevel: float = 0.18, name: str = "food_cube.obj", n: int = 14) -> Path:
    """A unit cube (half-size 1) with rounded edges: a superellipsoid, exponent from `bevel`."""
    out = cache_dir() / name
    if out.exists():
        return out
    e = 2.0 / (1.0 + 30 * bevel)                      # small exponent = square; 1 = sphere
    th = np.linspace(-np.pi / 2, np.pi / 2, n)
    ph = np.linspace(-np.pi, np.pi, 2 * n, endpoint=False)

    def sp(w, m):
        return np.sign(w) * np.abs(w) ** m

    v = [[sp(np.cos(a), e) * sp(np.cos(b), e), sp(np.cos(a), e) * sp(np.sin(b), e), sp(np.sin(a), e)]
         for a in th for b in ph]
    m = 2 * n
    f = []
    for i in range(n - 1):
        for j in range(m):
            p, q = i * m + j, i * m + (j + 1) % m
            f += [[p, p + m, q], [q, p + m, q + m]]
    return _write_obj(out, np.array(v), np.array(f))


def torus_obj(R: float, r: float, name: str = "tyre", n: int = 72, k: int = 14) -> Path:
    """A torus about z (a wheel's tyre / push rim)."""
    out = cache_dir() / ("%s_%.3f_%.3f.obj" % (name, R, r))
    if out.exists():
        return out
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    b = np.linspace(0, 2 * np.pi, k, endpoint=False)
    v = np.array([[(R + r * np.cos(q)) * np.cos(p), (R + r * np.cos(q)) * np.sin(p), r * np.sin(q)] for p in a for q in b])
    f = []
    for i in range(n):
        for j in range(k):
            p0, p1 = i * k + j, i * k + (j + 1) % k
            q0, q1 = ((i + 1) % n) * k + j, ((i + 1) % n) * k + (j + 1) % k
            f += [[p0, q0, p1], [p1, q0, q1]]
    return _write_obj(out, v, np.array(f))


def tub_obj(half_x: float, half_y: float, height: float, corner: float = 0.018, taper: float = 0.86,
            lip: float = 0.006, name: str = "tub") -> Path:
    """An open rounded-rectangle tub (the clear clamshell the food came in), origin at the centre of its base:
    a floor, walls that flare from `taper` x the top size, and a flat lip round the rim."""
    out = cache_dir() / ("%s_%.3f_%.3f_%.3f.obj" % (name, half_x, half_y, height))
    if out.exists():
        return out

    def loop(ax: float, ay: float, r: float, n: int = 10) -> np.ndarray:
        pts = []
        for cx, cy, a0 in ((ax - r, ay - r, 0), (-ax + r, ay - r, 90), (-ax + r, -ay + r, 180), (ax - r, -ay + r, 270)):
            for t in np.radians(np.linspace(a0, a0 + 90, n, endpoint=False)):
                pts.append((cx + r * np.cos(t), cy + r * np.sin(t)))
        return np.array(pts)

    rings = [(taper, 0.0), (taper, 0.0)]                     # (scale, z): the floor edge
    rings += [(taper + (1 - taper) * s, height * s) for s in np.linspace(0, 1, 8)[1:]]
    ring_pts = [np.column_stack([loop(half_x * sc, half_y * sc, corner * sc), np.full(40, z)]) for sc, z in rings]
    lip_pts = np.column_stack([loop(half_x + lip, half_y + lip, corner + lip), np.full(40, height)])
    v = [np.array([[0.0, 0.0, 0.0]])] + ring_pts[1:] + [lip_pts]
    V = np.vstack(v)
    n = 40
    f = [[0, 1 + (k + 1) % n, 1 + k] for k in range(n)]      # the floor fan
    for r in range(len(v) - 2):
        a0, b0 = 1 + r * n, 1 + (r + 1) * n
        for k in range(n):
            f += [[a0 + k, a0 + (k + 1) % n, b0 + (k + 1) % n], [a0 + k, b0 + (k + 1) % n, b0 + k]]
    return _write_obj(out, V, np.array(f))


def screen_png(name: str = "screen.png", w: int = 800, h: int = 500) -> Path:
    """The laptop's screen: a dark dashboard (a bite timeline and a camera tile), for the glow, not for reading."""
    out = cache_dir() / name
    if out.exists():
        return out
    from PIL import ImageDraw
    im = Image.new("RGB", (w, h), (10, 14, 22))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 34], fill=(18, 26, 40))
    for i, c in enumerate([(239, 68, 68), (245, 158, 11), (34, 197, 94)]):
        d.ellipse([14 + 22 * i, 11, 26 + 22 * i, 23], fill=c)
    d.rectangle([24, 60, 470, 330], fill=(20, 30, 46), outline=(56, 189, 248), width=2)
    for k in range(9):
        x = 40 + k * 46
        d.rectangle([x, 250 - 14 * (k % 4) - 40, x + 30, 310], fill=(56, 189, 248) if k % 3 else (34, 197, 94))
    d.rectangle([496, 60, 776, 200], fill=(26, 38, 58))
    d.rectangle([496, 214, 776, 330], fill=(26, 38, 58))
    for k in range(6):
        d.rectangle([24 + k * 125, 360, 24 + k * 125 + 110, 470], fill=(22, 34, 52), outline=(34, 197, 94), width=2)
    im.save(out)
    return out
