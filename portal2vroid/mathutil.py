"""Small linear-algebra helpers (numpy based).

Conventions
-----------
* Matrices are 4x4 numpy arrays acting on column vectors: ``world = parent @ local``.
* Quaternions are ``(x, y, z, w)`` like Source and glTF.
* Source "RadianEuler" (also used by SMD files) is ``(x, y, z)`` and builds the
  rotation ``Rz(z) @ Ry(y) @ Rx(x)``.
"""

from __future__ import annotations

import math

import numpy as np


def quat_to_mat3(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def mat3_to_quat(m) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def euler_to_mat3(rx: float, ry: float, rz: float) -> np.ndarray:
    """Source RadianEuler -> rotation matrix (Rz @ Ry @ Rx)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz_m @ ry_m @ rx_m


def mat3_to_euler(m) -> tuple[float, float, float]:
    """Rotation matrix -> Source RadianEuler (x, y, z), inverse of euler_to_mat3."""
    m = np.asarray(m, dtype=float)
    sy = -m[2, 0]
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    if abs(sy) < 0.999999:
        rx = math.atan2(m[2, 1], m[2, 2])
        rz = math.atan2(m[1, 0], m[0, 0])
    else:  # gimbal lock: fold everything into z
        rx = 0.0
        rz = math.atan2(-m[0, 1], m[1, 1])
    return rx, ry, rz


def make_mat4(rot3=None, pos=None) -> np.ndarray:
    m = np.eye(4)
    if rot3 is not None:
        m[:3, :3] = rot3
    if pos is not None:
        m[:3, 3] = pos
    return m


def trs_to_mat4(t=None, r=None, s=None) -> np.ndarray:
    m = np.eye(4)
    rot = quat_to_mat3(r) if r is not None else np.eye(3)
    scale = np.asarray(s, dtype=float) if s is not None else np.ones(3)
    m[:3, :3] = rot * scale[None, :]
    if t is not None:
        m[:3, 3] = t
    return m


def orthonormalize(m3) -> np.ndarray:
    """Closest rotation matrix (removes scale / shear)."""
    u, _, vt = np.linalg.svd(np.asarray(m3, dtype=float))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def rigid_inverse(m) -> np.ndarray:
    """Inverse of a rotation+translation 4x4 matrix."""
    inv = np.eye(4)
    r = m[:3, :3]
    inv[:3, :3] = r.T
    inv[:3, 3] = -r.T @ m[:3, 3]
    return inv


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v * 0.0
    return v / n


def axis_angle_mat3(axis, angle: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def rotation_between(a, b) -> np.ndarray:
    """Smallest rotation (3x3) taking direction ``a`` onto direction ``b``."""
    a = normalize(a)
    b = normalize(b)
    if not a.any() or not b.any():
        return np.eye(3)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 degrees: pick any perpendicular axis
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        return axis_angle_mat3(perp, math.pi)
    return axis_angle_mat3(axis / s, math.atan2(s, c))


def twist_to_align(axis, cur, target) -> np.ndarray:
    """Rotation about ``axis`` that best aligns ``cur`` with ``target`` (both projected)."""
    axis = normalize(axis)
    cp = normalize(cur - np.dot(cur, axis) * axis)
    tp = normalize(target - np.dot(target, axis) * axis)
    if not cp.any() or not tp.any():
        return np.eye(3)
    angle = math.atan2(np.dot(np.cross(cp, tp), axis), np.dot(cp, tp))
    return axis_angle_mat3(axis, angle)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rotate_about_point(rot3, point) -> np.ndarray:
    """4x4 transform that rotates by ``rot3`` around ``point``."""
    point = np.asarray(point, dtype=float)
    m = np.eye(4)
    m[:3, :3] = rot3
    m[:3, 3] = point - rot3 @ point
    return m
