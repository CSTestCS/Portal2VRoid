"""Convert VRM materials (MToon / glTF PBR) into Source VMT + VTF files."""

from __future__ import annotations

import os
import re

import numpy as np

from .vrm import VRMModel, decode_image
from .vtf import FLAG_EIGHTBITALPHA, FMT_DXT1, FMT_DXT5, build_vtf, toon_ramp_vtf

SHARED_DIR = "models/vroid/shared"
TOON_RAMP = f"{SHARED_DIR}/toon_ramp"


def _safe(name: str, taken: set[str]) -> str:
    base = re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]", "_", name.lower())).strip("_")[:40] or "mat"
    cand, k = base, 1
    while cand in taken:
        k += 1
        cand = f"{base}_{k}"
    taken.add(cand)
    return cand


def _pow2_floor(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _prepare(img, max_size: int):
    from PIL import Image

    w, h = img.size
    tw = min(_pow2_floor(max(w, 4)), max_size)
    th = min(_pow2_floor(max(h, 4)), max_size)
    if (tw, th) != (w, h):
        img = img.resize((tw, th), Image.LANCZOS)
    return img


def write_materials(
    vrm: VRMModel,
    game_dir: str,
    model_id: str,
    max_texture: int = 2048,
    toon: bool = True,
    log=print,
) -> list[str]:
    """Write VMT/VTF files under ``game_dir/materials``. Returns VMT basenames per VRM material."""
    mat_dir = os.path.join(game_dir, "materials", "models", "vroid", model_id)
    os.makedirs(mat_dir, exist_ok=True)
    if toon:
        shared = os.path.join(game_dir, "materials", *SHARED_DIR.split("/"))
        os.makedirs(shared, exist_ok=True)
        with open(os.path.join(shared, "toon_ramp.vtf"), "wb") as f:
            f.write(toon_ramp_vtf())

    # Which images need alpha?
    needs_alpha: dict[int, bool] = {}
    for m in vrm.materials:
        if m.image is not None:
            needs_alpha[m.image] = needs_alpha.get(m.image, False) or m.alpha_mode in ("MASK", "BLEND")

    tex_names: dict[int, str] = {}
    taken_tex: set[str] = set()
    for img_idx, alpha in needs_alpha.items():
        if not (0 <= img_idx < len(vrm.images)):
            continue
        data, _mime = vrm.images[img_idx]
        if not data:
            continue
        try:
            img = decode_image(data)
        except Exception as e:  # noqa: BLE001 - bad embedded image, skip it
            log(f"  ! could not decode texture {img_idx}: {e}")
            continue
        img = _prepare(img, max_texture)
        arr = np.asarray(img)
        has_alpha = alpha and arr[..., 3].min() < 250
        fmt = FMT_DXT5 if has_alpha else FMT_DXT1
        flags = FLAG_EIGHTBITALPHA if has_alpha else 0
        name = _safe(f"tex{img_idx}", taken_tex)
        with open(os.path.join(mat_dir, name + ".vtf"), "wb") as f:
            f.write(build_vtf(img, fmt, flags))
        tex_names[img_idx] = name
        log(f"  texture {name}: {img.size[0]}x{img.size[1]} {'DXT5' if has_alpha else 'DXT1'}")

    vmt_names: list[str] = []
    taken: set[str] = set()
    for m in vrm.materials:
        name = _safe(m.name, taken)
        vmt_names.append(name)
        lines = ['"VertexLitGeneric"', "{"]
        if m.image in tex_names:
            lines.append(f'\t"$basetexture" "models/vroid/{model_id}/{tex_names[m.image]}"')
        else:
            lines.append('\t"$basetexture" "models/debug/debugwhite"')
        r, g, b, a = (list(m.base_color) + [1, 1, 1, 1])[:4]
        if abs(r - 1) > 1e-3 or abs(g - 1) > 1e-3 or abs(b - 1) > 1e-3:
            lines.append(f'\t"$color2" "[{r:.3f} {g:.3f} {b:.3f}]"')
        lines.append('\t"$halflambert" "1"')
        if toon:
            lines.append(f'\t"$lightwarptexture" "{TOON_RAMP}"')
        if m.double_sided:
            lines.append('\t"$nocull" "1"')
        if m.alpha_mode == "MASK":
            lines.append('\t"$alphatest" "1"')
            lines.append(f'\t"$alphatestreference" "{m.alpha_cutoff:.3f}"')
            lines.append('\t"$allowalphatocoverage" "1"')
        elif m.alpha_mode == "BLEND":
            lines.append('\t"$translucent" "1"')
            if a < 0.999:
                lines.append(f'\t"$alpha" "{a:.3f}"')
        lines.append("}")
        with open(os.path.join(mat_dir, name + ".vmt"), "w", encoding="ascii", errors="replace", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    return vmt_names
