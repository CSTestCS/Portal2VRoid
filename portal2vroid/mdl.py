"""Read the skeleton (and a few header fields) from a Source .mdl file (v44-v49)."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from .mathutil import make_mat4, quat_to_mat3, rigid_inverse

BONE_STRUCT_SIZE = 216
IKLINK_SIZE = 28


@dataclass
class MDLBone:
    name: str
    parent: int
    pos: np.ndarray
    quat: np.ndarray
    pose_to_bone: np.ndarray  # 4x4
    flags: int = 0
    proctype: int = 0


@dataclass
class MDLIKChain:
    name: str
    links: list[int]
    knee_dir: np.ndarray


@dataclass
class MDLSkeleton:
    name: str
    version: int
    bones: list[MDLBone]
    ikchains: list[MDLIKChain] = field(default_factory=list)
    include_models: list[str] = field(default_factory=list)
    eye_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    illum_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hull_min: np.ndarray = field(default_factory=lambda: np.array([-16.0, -16.0, 0.0]))
    hull_max: np.ndarray = field(default_factory=lambda: np.array([16.0, 16.0, 72.0]))
    num_sequences: int = 0

    def index(self, name: str) -> int:
        for i, b in enumerate(self.bones):
            if b.name == name:
                return i
        return -1

    def local_matrices(self) -> list[np.ndarray]:
        return [make_mat4(quat_to_mat3(b.quat), b.pos) for b in self.bones]

    def world_matrices(self) -> list[np.ndarray]:
        """Default (animation) rest pose in model space: forward kinematics of pos/quat."""
        local = self.local_matrices()
        world: list[np.ndarray] = []
        for i, b in enumerate(self.bones):
            if 0 <= b.parent < i:
                world.append(world[b.parent] @ local[i])
            else:
                world.append(local[i])
        return world

    def bind_matrices(self) -> list[np.ndarray | None]:
        """Bind pose the mesh was skinned in (inverse of poseToBone); None if unusable."""
        out: list[np.ndarray | None] = []
        for b in self.bones:
            p2b = b.pose_to_bone
            if abs(np.linalg.det(p2b[:3, :3]) - 1.0) < 1e-2:
                out.append(rigid_inverse(p2b))
            else:
                out.append(None)
        return out

    def rest_vs_bind_report(self) -> list[str]:
        """Where the animation rest pose and the mesh bind pose disagree (debug aid)."""
        lines = []
        for b, fk, bind in zip(self.bones, self.world_matrices(), self.bind_matrices()):
            if bind is None:
                lines.append(f"{b.name}: no bind matrix")
                continue
            dp = float(np.linalg.norm(fk[:3, 3] - bind[:3, 3]))
            r = fk[:3, :3].T @ bind[:3, :3]
            ang = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))
            if dp > 0.05 or ang > 0.5:
                lines.append(f"{b.name}: rest/bind differ by {dp:.2f} units, {ang:.1f} deg")
        return lines


def _cstr(data: bytes, off: int) -> str:
    if off < 0 or off >= len(data):
        return ""
    end = data.find(b"\0", off)
    if end < 0:
        end = len(data)
    return data[off:end].decode("latin-1")


def _vec(data: bytes, off: int) -> np.ndarray:
    return np.array(struct.unpack_from("<3f", data, off), dtype=float)


def parse_mdl(data: bytes) -> MDLSkeleton:
    if data[:4] != b"IDST":
        raise ValueError("not a Source model (missing IDST header)")
    if len(data) < 408:
        raise ValueError("truncated model file")
    version = struct.unpack_from("<i", data, 4)[0]
    if version < 44 or version > 53:
        raise ValueError(f"unsupported MDL version {version}")
    name = data[12:76].split(b"\0", 1)[0].decode("latin-1")
    eye = _vec(data, 80)
    illum = _vec(data, 92)
    hull_min = _vec(data, 104)
    hull_max = _vec(data, 116)
    numbones, boneindex = struct.unpack_from("<ii", data, 156)
    numlocalseq = struct.unpack_from("<i", data, 188)[0]
    if numbones <= 0 or numbones > 256:
        raise ValueError(f"implausible bone count {numbones}")

    bones: list[MDLBone] = []
    for i in range(numbones):
        base = boneindex + i * BONE_STRUCT_SIZE
        nameoff, parent = struct.unpack_from("<ii", data, base)
        pos = _vec(data, base + 32)
        quat = np.array(struct.unpack_from("<4f", data, base + 44), dtype=float)
        m = struct.unpack_from("<12f", data, base + 96)
        p2b = np.eye(4)
        p2b[:3, :] = np.array(m, dtype=float).reshape(3, 4)
        flags, proctype = struct.unpack_from("<ii", data, base + 160)
        bones.append(MDLBone(_cstr(data, base + nameoff), parent, pos, quat, p2b, flags, proctype))

    skel = MDLSkeleton(
        name=name,
        version=version,
        bones=bones,
        eye_position=eye,
        illum_position=illum,
        hull_min=hull_min,
        hull_max=hull_max,
        num_sequences=numlocalseq,
    )

    try:
        skel.ikchains = _parse_ikchains(data, numbones)
    except (struct.error, ValueError):
        skel.ikchains = []
    try:
        numinc, incindex = struct.unpack_from("<ii", data, 336)
        if 0 <= numinc < 64:
            for i in range(numinc):
                base = incindex + i * 8
                _, nameoff = struct.unpack_from("<ii", data, base)
                skel.include_models.append(_cstr(data, base + nameoff))
    except struct.error:
        pass
    return skel


def _parse_ikchains(data: bytes, numbones: int) -> list[MDLIKChain]:
    num, index = struct.unpack_from("<ii", data, 284)
    if num <= 0:
        return []
    if num > 32:
        raise ValueError("implausible ik chain count")
    # The chain struct is 16 bytes in most branches; some add padding. Pick the
    # stride that yields sane names and bone indices.
    for stride in (16, 32, 20, 24, 28):
        chains = []
        ok = True
        for i in range(num):
            base = index + i * stride
            nameoff, linktype, numlinks, linkindex = struct.unpack_from("<iiii", data, base)
            nm = _cstr(data, base + nameoff)
            if not nm or not nm.isprintable() or not (1 <= numlinks <= 8):
                ok = False
                break
            links, knee = [], np.zeros(3)
            for j in range(numlinks):
                lb = base + linkindex + j * IKLINK_SIZE
                bone = struct.unpack_from("<i", data, lb)[0]
                if not (0 <= bone < numbones):
                    ok = False
                    break
                links.append(bone)
                kd = _vec(data, lb + 4)
                if j == 0:
                    knee = kd
            if not ok:
                break
            chains.append(MDLIKChain(nm, links, knee))
        if ok:
            return chains
    return []


def vvd_height(data: bytes) -> float | None:
    """Top-of-mesh height (max Z) of a .vvd vertex file, or None if unreadable."""
    try:
        if data[:4] != b"IDSV":
            return None
        vstart, tstart = struct.unpack_from("<ii", data, 56)
        count = (tstart - vstart) // 48 if tstart > vstart else (len(data) - vstart) // 48
        if count <= 0:
            return None
        verts = np.frombuffer(data, dtype=np.float32, count=count * 12, offset=vstart).reshape(count, 12)
        z = verts[:, 6]  # position.z (weights 3f+bones 4b = 16 bytes, then x,y,z)
        z = z[np.isfinite(z)]
        return float(z.max()) if len(z) else None
    except (struct.error, ValueError):
        return None


def vvd_positions(data: bytes) -> np.ndarray | None:
    """All vertex positions (N,3) in a .vvd file, or None if unreadable."""
    try:
        if data[:4] != b"IDSV":
            return None
        vstart, tstart = struct.unpack_from("<ii", data, 56)
        count = (tstart - vstart) // 48 if tstart > vstart else (len(data) - vstart) // 48
        if count <= 0:
            return None
        verts = np.frombuffer(data, dtype=np.float32, count=count * 12, offset=vstart).reshape(count, 12)
        pos = verts[:, 4:7].astype(float)
        return pos[np.all(np.isfinite(pos), axis=1)]
    except (struct.error, ValueError):
        return None


def vvd_skin(data: bytes):
    """(positions (N,3), bones (N,3) int, weights (N,3)) from a .vvd file, or None.

    Bone indices refer to the model's bone table.
    """
    try:
        if data[:4] != b"IDSV":
            return None
        vstart, tstart = struct.unpack_from("<ii", data, 56)
        count = (tstart - vstart) // 48 if tstart > vstart else (len(data) - vstart) // 48
        if count <= 0:
            return None
        rec = np.dtype([("w", "<f4", 3), ("b", "u1", 3), ("n", "u1"), ("p", "<f4", 3), ("nrm", "<f4", 3), ("uv", "<f4", 2)])
        v = np.frombuffer(data, dtype=rec, count=count, offset=vstart)
        n = v["n"].astype(int)
        w = v["w"].astype(float).copy()
        for k in range(3):
            w[n <= k, k] = 0.0
        return v["p"].astype(float), v["b"].astype(int), w
    except (struct.error, ValueError):
        return None
