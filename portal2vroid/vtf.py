"""Write Valve Texture Format (VTF 7.2) files with DXT1/DXT5 or uncompressed data."""

from __future__ import annotations

import struct

import numpy as np

FMT_RGBA8888 = 0
FMT_BGR888 = 3
FMT_BGRA8888 = 12
FMT_DXT1 = 13
FMT_DXT5 = 15
FMT_NONE = 0xFFFFFFFF

FLAG_CLAMPS = 0x4
FLAG_CLAMPT = 0x8
FLAG_NOMIP = 0x100
FLAG_NOLOD = 0x200
FLAG_ONEBITALPHA = 0x1000
FLAG_EIGHTBITALPHA = 0x2000


def _to_565(rgb: np.ndarray) -> np.ndarray:
    r = (rgb[..., 0].astype(np.uint32) * 31 + 127) // 255
    g = (rgb[..., 1].astype(np.uint32) * 63 + 127) // 255
    b = (rgb[..., 2].astype(np.uint32) * 31 + 127) // 255
    return ((r << 11) | (g << 5) | b).astype(np.uint16)


def _from_565(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.uint32)
    r = ((c >> 11) & 31) * 255 // 31
    g = ((c >> 5) & 63) * 255 // 63
    b = (c & 31) * 255 // 31
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _blocks(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """(H,W,C) -> (nblocks,16,C), padding to multiples of 4 by edge replication."""
    h, w, c = img.shape
    ph, pw = (-h) % 4, (-w) % 4
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
    H, W = img.shape[:2]
    b = img.reshape(H // 4, 4, W // 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(-1, 16, c)
    return b, H // 4, W // 4


def _encode_color_blocks(rgb: np.ndarray) -> np.ndarray:
    """rgb: (N,16,3) uint8 -> (N,8) uint8 DXT1 colour blocks (always 4-colour mode)."""
    px = rgb.astype(np.float32)
    mean = px.mean(axis=1, keepdims=True)
    centered = px - mean
    # principal axis via a few power iterations on the covariance
    cov = np.einsum("nki,nkj->nij", centered, centered)
    axis = np.ones((len(px), 3), dtype=np.float32)
    for _ in range(4):
        axis = np.einsum("nij,nj->ni", cov, axis)
        norm = np.linalg.norm(axis, axis=1, keepdims=True)
        axis = np.where(norm > 1e-6, axis / np.maximum(norm, 1e-6), np.array([0.577, 0.577, 0.577], dtype=np.float32))
    proj = np.einsum("nki,ni->nk", centered, axis)
    lo = proj.min(axis=1)
    hi = proj.max(axis=1)
    c0 = np.clip(mean[:, 0] + hi[:, None] * axis, 0, 255)
    c1 = np.clip(mean[:, 0] + lo[:, None] * axis, 0, 255)
    e0 = _to_565(c0)
    e1 = _to_565(c1)
    swap = e0 < e1
    e0, e1 = np.where(swap, e1, e0), np.where(swap, e0, e1)
    p0, p1 = _from_565(e0), _from_565(e1)
    palette = np.stack([p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3], axis=1)  # (N,4,3)
    d = ((px[:, :, None, :] - palette[:, None, :, :]) ** 2).sum(axis=-1)  # (N,16,4)
    idx = d.argmin(axis=2).astype(np.uint32)
    idx[e0 == e1] = 0
    bits = np.zeros(len(px), dtype=np.uint32)
    for k in range(16):
        bits |= idx[:, k] << (2 * k)
    out = np.zeros((len(px), 8), dtype=np.uint8)
    out[:, 0] = e0 & 0xFF
    out[:, 1] = e0 >> 8
    out[:, 2] = e1 & 0xFF
    out[:, 3] = e1 >> 8
    for k in range(4):
        out[:, 4 + k] = (bits >> (8 * k)) & 0xFF
    return out


def _encode_alpha_blocks(a: np.ndarray) -> np.ndarray:
    """a: (N,16) uint8 -> (N,8) uint8 DXT5 alpha blocks (8-alpha mode)."""
    a = a.astype(np.int32)
    a0 = a.max(axis=1)
    a1 = a.min(axis=1)
    same = a0 == a1
    a0 = np.where(same & (a0 < 255), a0 + 1, a0)
    a1 = np.where(same & (a0 == 255) & (a1 == 255), 254, a1)
    levels = np.stack(
        [a0, a1] + [((8 - k) * a0 + (k - 1) * a1) // 7 for k in range(2, 8)], axis=1
    )  # (N,8): idx2 = (6a0+1a1)/7 ... idx7 = (a0+6a1)/7
    d = np.abs(a[:, :, None] - levels[:, None, :])
    idx = d.argmin(axis=2).astype(np.uint64)
    bits = np.zeros(len(a), dtype=np.uint64)
    for k in range(16):
        bits |= idx[:, k] << np.uint64(3 * k)
    out = np.zeros((len(a), 8), dtype=np.uint8)
    out[:, 0] = a0
    out[:, 1] = a1
    for k in range(6):
        out[:, 2 + k] = ((bits >> np.uint64(8 * k)) & np.uint64(0xFF)).astype(np.uint8)
    return out


BATCH = 16384  # blocks per batch, keeps peak memory around 100 MB


def encode_dxt1(rgba: np.ndarray) -> bytes:
    b, _, _ = _blocks(rgba[..., :3])
    return b"".join(_encode_color_blocks(b[i : i + BATCH]).tobytes() for i in range(0, len(b), BATCH))


def encode_dxt5(rgba: np.ndarray) -> bytes:
    b, _, _ = _blocks(rgba)
    out = []
    for i in range(0, len(b), BATCH):
        chunk = b[i : i + BATCH]
        alpha = _encode_alpha_blocks(chunk[:, :, 3])
        color = _encode_color_blocks(chunk[:, :, :3])
        out.append(np.concatenate([alpha, color], axis=1).tobytes())
    return b"".join(out)


def _encode(rgba: np.ndarray, fmt: int) -> bytes:
    if fmt == FMT_DXT1:
        return encode_dxt1(rgba)
    if fmt == FMT_DXT5:
        return encode_dxt5(rgba)
    if fmt == FMT_BGRA8888:
        return rgba[..., [2, 1, 0, 3]].astype(np.uint8).tobytes()
    if fmt == FMT_BGR888:
        return rgba[..., [2, 1, 0]].astype(np.uint8).tobytes()
    if fmt == FMT_RGBA8888:
        return rgba.astype(np.uint8).tobytes()
    raise ValueError(f"unsupported format {fmt}")


def _mip_chain(img, count: int):
    """PIL RGBA image -> list of numpy arrays, largest first."""
    from PIL import Image

    mips = []
    cur = img
    for _ in range(count):
        mips.append(np.asarray(cur, dtype=np.uint8))
        w, h = cur.size
        if w == 1 and h == 1:
            break
        cur = cur.resize((max(1, w // 2), max(1, h // 2)), Image.BOX)
    return mips


def build_vtf(img, fmt: int = FMT_DXT1, flags: int = 0, mipmaps: bool = True) -> bytes:
    """Build a VTF 7.2 file from a PIL image (converted to RGBA, power-of-two sized)."""
    from PIL import Image

    img = img.convert("RGBA")
    w, h = img.size
    if w & (w - 1) or h & (h - 1):
        raise ValueError("VTF textures must be power-of-two sized")
    if not mipmaps:
        flags |= FLAG_NOMIP
    count = 1
    if mipmaps:
        count = max(w, h).bit_length()
    mips = _mip_chain(img, count)
    count = len(mips)

    arr = mips[0].astype(np.float32) / 255.0
    lin = arr[..., :3] ** 2.2
    reflectivity = lin.reshape(-1, 3).mean(axis=0)

    lw, lh = min(16, w), min(16, h)
    low = np.asarray(img.resize((lw, lh), Image.BOX), dtype=np.uint8)
    low_data = encode_dxt1(low)

    header = struct.pack(
        "<4s2IIHHIHH4x3f4xfIBIBBH",
        b"VTF\0",
        7,
        2,
        80,
        w,
        h,
        flags,
        1,
        0,
        float(reflectivity[0]),
        float(reflectivity[1]),
        float(reflectivity[2]),
        1.0,
        fmt,
        count,
        FMT_DXT1,
        lw,
        lh,
        1,
    )
    header = header.ljust(80, b"\0")
    body = bytearray(low_data)
    for mip in reversed(mips):  # smallest first
        body += _encode(mip, fmt)
    return bytes(header) + bytes(body)


def toon_ramp_vtf(shade: float = 0.72, edge: float = 0.42, softness: float = 0.06) -> bytes:
    """A light-warp ramp that gives VertexLitGeneric a cel-shaded (MToon-like) look."""
    from PIL import Image

    x = np.linspace(0.0, 1.0, 256)
    t = np.clip((x - (edge - softness)) / (2 * softness), 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    v = shade + (1.0 - shade) * t
    v = np.clip(v * 255.0, 0, 255).astype(np.uint8)
    row = np.stack([v, v, v, np.full_like(v, 255)], axis=-1)
    img = Image.fromarray(np.repeat(row[None], 16, axis=0), "RGBA")
    return build_vtf(img, FMT_BGRA8888, FLAG_CLAMPS | FLAG_CLAMPT | FLAG_NOLOD, mipmaps=False)
