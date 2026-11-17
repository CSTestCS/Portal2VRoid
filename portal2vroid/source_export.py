"""Write SMD + QC files for studiomdl from a ConversionResult."""

from __future__ import annotations

import os

import numpy as np

from .mathutil import mat3_to_euler, rigid_inverse
from .mdl import MDLSkeleton
from .retarget import ConversionResult

# Portal 2's studiomdl keeps SMD coordinates as they are (verified on a compiled
# model: no default rotation is applied), so SMD space == Chell's model space.
SMD_FROM_GAME = np.eye(4)

MAX_TRIS_PER_SMD = 20000


def _fmt(v: float) -> str:
    s = f"{v:.6f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s


def _locals(worlds: list[np.ndarray], parents: list[int]) -> list[np.ndarray]:
    out = []
    for i, w in enumerate(worlds):
        p = parents[i]
        if p >= 0:
            out.append(rigid_inverse(worlds[p]) @ w)
        else:
            out.append(SMD_FROM_GAME @ w)
    return out


def _nodes_block(names: list[str], parents: list[int]) -> list[str]:
    lines = ["nodes"]
    for i, (n, p) in enumerate(zip(names, parents)):
        lines.append(f'{i} "{n}" {p}')
    lines.append("end")
    return lines


def _skeleton_block(locals_: list[np.ndarray]) -> list[str]:
    lines = ["skeleton", "time 0"]
    for i, m in enumerate(locals_):
        rx, ry, rz = mat3_to_euler(m[:3, :3])
        x, y, z = m[:3, 3]
        lines.append(f"{i} {_fmt(x)} {_fmt(y)} {_fmt(z)} {_fmt(rx)} {_fmt(ry)} {_fmt(rz)}")
    lines.append("end")
    return lines


def write_smd(path: str, names, parents, worlds, triangles: list[str] | None = None) -> None:
    lines = ["version 1"]
    lines += _nodes_block(names, parents)
    lines += _skeleton_block(_locals(worlds, parents))
    if triangles is not None:
        lines.append("triangles")
        lines += triangles
        lines.append("end")
    with open(path, "w", encoding="ascii", errors="replace", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _mesh_triangles(mesh, material_names: list[str]) -> list[list[str]]:
    """Triangle blocks for one mesh, split into chunks studiomdl can handle."""
    r = SMD_FROM_GAME[:3, :3]
    pos = mesh.positions @ r.T
    nrm = mesh.normals @ r.T
    uv = mesh.uvs
    mat = material_names[mesh.material] if mesh.material < len(material_names) else material_names[0]
    vert_lines = []
    for i in range(len(pos)):
        links = []
        for b, w in zip(mesh.bones[i], mesh.weights[i]):
            if w > 1e-5:
                links.append(f"{int(b)} {_fmt(float(w))}")
        parent = int(mesh.bones[i][0])
        vert_lines.append(
            f"{parent} {_fmt(pos[i,0])} {_fmt(pos[i,1])} {_fmt(pos[i,2])} "
            f"{_fmt(nrm[i,0])} {_fmt(nrm[i,1])} {_fmt(nrm[i,2])} "
            f"{_fmt(uv[i,0])} {_fmt(1.0 - uv[i,1])} {len(links)} " + " ".join(links)
        )
    chunks = []
    tris = mesh.indices
    for start in range(0, len(tris), MAX_TRIS_PER_SMD):
        block = []
        for a, b, c in tris[start : start + MAX_TRIS_PER_SMD]:
            block.append(mat)
            block.append(vert_lines[a])
            block.append(vert_lines[b])
            block.append(vert_lines[c])
        chunks.append(block)
    return chunks


def write_model_sources(
    out_dir: str,
    model_id: str,
    result: ConversionResult,
    chell: MDLSkeleton,
    material_names: list[str],
    chell_model_path="player/chell/player.mdl",
) -> str:
    """``chell_model_path`` may be a list: every entry becomes an $includemodel."""
    """Write all SMDs and the QC. Returns the QC path."""
    os.makedirs(out_dir, exist_ok=True)
    names = [b.name for b in result.bones]
    parents = [b.parent for b in result.bones]
    worlds = [b.world for b in result.bones]

    # Group meshes by their glTF mesh name -> one $body each (keeps studiomdl limits happy).
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for mesh in result.meshes:
        key = "".join(ch if ch.isalnum() else "_" for ch in mesh.name.lower())[:24] or "mesh"
        for ci, chunk in enumerate(_mesh_triangles(mesh, material_names)):
            k = key if ci == 0 else f"{key}_{ci}"
            if k not in groups:
                groups[k] = []
                order.append(k)
            groups[k].extend(chunk)
    body_files = []
    for gi, key in enumerate(order):
        fn = f"body_{gi:02d}_{key}.smd"
        write_smd(os.path.join(out_dir, fn), names, parents, worlds, groups[key])
        body_files.append((f"body_{gi:02d}", fn))

    # Proportion trick sources.
    chell_worlds = list(result.chell_world) + worlds[result.n_chell :]
    # Extra bones keep their own locals in the reference, so their delta is zero.
    ref_locals_src = _locals(chell_worlds[: result.n_chell], parents[: result.n_chell])
    ours = _locals(worlds, parents)
    ref_all = ref_locals_src + ours[result.n_chell :]
    _write_anim_smd(os.path.join(out_dir, "chell_reference.smd"), names, parents, ref_all)
    _write_anim_smd(os.path.join(out_dir, "proportions.smd"), names, parents, ours)

    hull_min, hull_max = chell.hull_min, chell.hull_max
    has_hull = bool(np.all(hull_max > hull_min))
    q = [
        "// Generated by Portal2VRoid - do not edit by hand",
        f'$modelname "vroid/{model_id}.mdl"',
        f'$cdmaterials "models/vroid/{model_id}"',
        '$surfaceprop "flesh"',
        # Copy Chell's box only if she has one (Portal 2's Chell doesn't; studiomdl then
        # computes it from the mesh, which is right for her Y-up model space).
        f"$bbox {_v(hull_min)} {_v(hull_max)}" if has_hull else "",
        "$cbox 0 0 0 0 0 0" if has_hull else "",
        "",
    ]
    for body, fn in body_files:
        q.append(f'$body "{body}" "{fn}"')
    q += [
        "",
        "// Proportion trick: Chell's animations also store bone positions, so add a",
        "// delta that moves every joint back to where this avatar has it.",
        '$sequence "reference" "chell_reference.smd" fps 1',
        '$animation "a_vroid_proportions" "proportions.smd" subtract "reference" 0',
        '$sequence "vroid_proportions" "a_vroid_proportions" predelta autoplay',
        "",
    ]
    includes = [chell_model_path] if isinstance(chell_model_path, str) else list(chell_model_path)
    seen = set()
    for inc in includes:
        if inc.lower() not in seen:
            seen.add(inc.lower())
            q.append(f'$includemodel "{inc}"')
    q += [
        "",
    ]
    # No $ikchain: studiomdl rewrites the frames of IK-chain bones that point along -X
    # (Chell's right arm and leg) by 180 degrees, which made her animations twist the
    # avatar's right arm and leg. Without chains the engine skips IK, and Chell's
    # animations already place hands and feet.
    jig = [b for b in result.bones if b.jiggle_length > 0.05]
    for b in jig:
        q += [
            f'$jigglebone "{b.name}"',
            "{",
            "\tis_flexible",
            "\t{",
            f"\t\tlength {_fmt(b.jiggle_length)}",
            "\t\ttip_mass 0",
            "\t\tpitch_stiffness 60",
            "\t\tpitch_damping 8",
            "\t\tyaw_stiffness 60",
            "\t\tyaw_damping 8",
            "\t\talong_stiffness 100",
            "\t\talong_damping 0",
            "\t\tangle_constraint 35",
            "\t}",
            "}",
        ]
    qc_path = os.path.join(out_dir, f"{model_id}.qc")
    with open(qc_path, "w", encoding="ascii", errors="replace", newline="\n") as f:
        f.write("\n".join(line for line in q if line is not None) + "\n")
    return qc_path


def _write_anim_smd(path, names, parents, locals_):
    lines = ["version 1"]
    lines += _nodes_block(names, parents)
    lines += _skeleton_block(locals_)
    with open(path, "w", encoding="ascii", errors="replace", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _v(v) -> str:
    return " ".join(_fmt(float(x)) for x in v)
