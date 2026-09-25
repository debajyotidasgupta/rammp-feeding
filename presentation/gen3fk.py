"""Kinova Gen3 7-DoF forward kinematics (classic DH, from Kinova's user guide), for drawing only."""
import numpy as np

D = [-(0.1564 + 0.1284), -(0.0054 + 0.0064), -(0.2104 + 0.2104), -(0.0064 + 0.0064), -(0.2084 + 0.1059), 0.0,
     -(0.1059 + 0.0615)]
ALPHA = [np.pi / 2] * 6 + [np.pi]
OFF = [0.0] + [np.pi] * 6
TB0 = np.diag([1.0, -1.0, -1.0, 1.0])
TOOL_Z = 0.120


def dh(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1]])


def frames(q_deg):
    """Base-frame transforms of every joint frame, then the tool frame (120 mm along the flange z)."""
    T = TB0.copy(); out = [T.copy()]
    for i, q in enumerate(np.radians(q_deg)):
        T = T @ dh(q + OFF[i], D[i], 0.0, ALPHA[i]); out.append(T.copy())
    tool = T @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, TOOL_Z], [0, 0, 0, 1.0]])
    out.append(tool)
    return out


def tool_xyz(q_deg):
    return frames(q_deg)[-1][:3, 3]


if __name__ == "__main__":
    pairs = [([-90.0, 8.071, 0.0, -139.199, 0.446, -49.528, -174.304], (0.0236, -0.1766, 0.2082)),
             ([154.812, -9.686, -31.181, -59.005, 4.715, -113.549, 35.326], (0.2205, 0.2826, 0.5344)),
             ([-90.501, 14.243, 0.441, -78.165, -0.354, -116.713, -173.798], (0.0236, -0.1767, 0.5453)),
             ([-102.4, -42.0, -37.2, -97.5, -41.1, -111.2, 131.8], (0.2997, -0.0968, 0.4038)),
             ([176.973, 7.795, -4.804, -73.442, -1.175, -114.949, 81.197], (0.2254, 0.0616, 0.5445))]
    for q, p in pairs:
        e = tool_xyz(q) - np.array(p)
        print("FK %s vs controller %s: error %.1f mm" % (tool_xyz(q).round(4), p, 1000 * np.linalg.norm(e)))
