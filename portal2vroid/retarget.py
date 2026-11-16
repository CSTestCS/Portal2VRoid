"""Turn a VRM avatar into a skeleton + skinned mesh that plays Chell's animations.

The approach:

1. Put the avatar in Chell's model space (Source units, +X forward, +Z up).
2. Scale it uniformly so its height matches Chell's height.
3. "Repose" the avatar: rotate each limb segment so it points the same way as
   Chell's rest pose (her reference pose is not a T-pose). The mesh follows via
   the avatar's own skin weights. Proportions (segment lengths) are untouched.
4. Build the output skeleton: every Chell bone keeps Chell's name, parent and rest
   *orientation*, but mapped bones take the avatar's joint *positions*. Extra VRM
   bones (hair, clothes, accessories) are appended under their mapped parents.

Because rest orientations match Chell's exactly, Chell's animations (pulled in
with $includemodel) rotate the avatar correctly. Chell's animations also carry
bone positions, so the QC adds an autoplay "proportions" delta sequence that
restores the avatar's own joint positions (the well-known proportion trick).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .bonemap import classify_source_bones, map_vrm_to_source, primary_child, twist_reference
from .mathutil import make_mat4, normalize, rotate_about_point, rotation_between, twist_to_align
from .mdl import MDLSkeleton
from .vrm import VRMModel

MAX_BONES = 128
MAX_WEIGHTS = 3


class ConversionError(Exception):
    pass


@dataclass
class OutBone:
    name: str
    parent: int
    world: np.ndarray  # 4x4, game space
    source: str  # "chell" or "vrm"
    jiggle_length: float = 0.0


@dataclass
class OutMesh:
    name: str
    material: int
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    bones: np.ndarray  # (N, 3) int
    weights: np.ndarray  # (N, 3) float
    indices: np.ndarray  # (M, 3)


@dataclass
class ConversionResult:
    bones: list[OutBone]
    meshes: list[OutMesh]
    chell_world: list[np.ndarray]  # rest pose of Chell's bones (same order as bones[:n_chell])
    n_chell: int
    scale: float
    mapping: dict[str, str]  # VRM slot -> Chell bone name
    warnings: list[str] = field(default_factory=list)


@dataclass
class ConvertOptions:
    height_scale: float = 1.0  # multiplier on top of "match Chell's height"
    jiggle: bool = True
    max_bones: int = MAX_BONES
    chell_height: float | None = None  # legacy: top-of-mesh height when Chell's model is Z-up
    chell_vertices: np.ndarray | None = None  # Chell's mesh vertex positions (from player.vvd), if available
    chell_skin: tuple | None = None  # (positions, bones, weights) of Chell's mesh, for twist-bone weights
    # Experimental (made things worse in game in 1.0.5): copy Chell's twist-bone skinning
    # and place her helper bones along the avatar's limbs. Off by default.
    twist_weights: bool = False


def _sanitize_bone_name(name: str, idx: int, taken: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_.\-]", "_", name).strip("_")
    if not base or not re.search(r"[A-Za-z]", base):
        base = f"node{idx}"
    base = "vrm_" + base[:48]
    cand = base
    k = 1
    while cand.lower() in taken:
        k += 1
        cand = f"{base}_{k}"
    taken.add(cand.lower())
    return cand


def _horizontal(v: np.ndarray, up_axis: int) -> np.ndarray:
    v = np.array(v, dtype=float)
    v[up_axis] = 0.0
    return normalize(v)


def convert(vrm: VRMModel, chell: MDLSkeleton, opts: ConvertOptions | None = None) -> ConversionResult:
    opts = opts or ConvertOptions()
    warnings: list[str] = []

    # ---- Chell skeleton -------------------------------------------------
    c_names = [b.name for b in chell.bones]
    c_parents = [b.parent for b in chell.bones]
    c_world = chell.world_matrices()
    c_pos = np.array([w[:3, 3] for w in c_world])
    slots, spines = classify_source_bones(c_names, c_parents)
    mapping = map_vrm_to_source(vrm.humanoid, slots, spines)
    for req in ("hips", "head", "leftUpperLeg", "rightUpperLeg", "leftUpperArm", "rightUpperArm"):
        if req not in mapping:
            raise ConversionError(
                f"Could not find Chell's '{req}' bone. Bones in the player model: " + ", ".join(c_names[:80])
            )
    missing = sorted(s for s in vrm.humanoid if s not in mapping and not s.endswith("Eye") and s != "jaw")
    if missing:
        warnings.append("VRM bones without a Chell counterpart (kept, but not animated): " + ", ".join(missing))

    # Chell's frame, measured from her skeleton (don't assume Z-up / +X forward).
    feet = [mapping[s] for s in ("leftFoot", "rightFoot", "leftLowerLeg", "rightLowerLeg") if s in mapping][:2]
    base = c_pos[feet].mean(axis=0) if feet else c_pos[[mapping["leftUpperLeg"], mapping["rightUpperLeg"]]].mean(axis=0)
    c_up = normalize(c_pos[mapping["head"]] - base)
    if not c_up.any():
        c_up = np.array([0.0, 0.0, 1.0])
    # snap to the nearest axis if close (models are authored axis-aligned)
    ax = int(np.argmax(np.abs(c_up)))
    if abs(c_up[ax]) > 0.95:
        c_up = np.eye(3)[ax] * np.sign(c_up[ax])
    lr = c_pos[mapping["leftUpperLeg"]] - c_pos[mapping["rightUpperLeg"]]
    c_left = normalize(lr - np.dot(lr, c_up) * c_up)
    if not c_left.any():
        raise ConversionError("Chell's hips are at the same position; cannot work out which way she faces")
    c_fwd = normalize(np.cross(c_left, c_up))
    chell_frame = np.column_stack([c_fwd, c_left, c_up])  # canonical (+X fwd,+Y left,+Z up) -> Chell space

    def up_of(p):
        return float(np.dot(p, c_up))

    low_joints = [mapping[s] for s in ("leftToes", "rightToes", "leftFoot", "rightFoot") if s in mapping]
    head_up = up_of(c_pos[mapping["head"]])
    joint_floor = min(up_of(c_pos[j]) for j in low_joints) if low_joints else 0.0
    # Chell's floor and full height: from her mesh if it is plausible, else from her skeleton.
    c_floor = min(0.0, joint_floor) if abs(joint_floor) < 0.15 * max(abs(head_up), 1.0) else joint_floor
    c_height = None
    verts = opts.chell_vertices
    if verts is not None and len(verts):
        proj = np.asarray(verts, dtype=float) @ c_up
        lo, hi = float(proj.min()), float(proj.max())
        head_h_c = head_up - lo
        if head_h_c > 0 and 1.0 <= (hi - lo) / head_h_c <= 1.6:
            c_floor, c_height = lo, hi - lo
        else:
            warnings.append(
                f"Chell's mesh height looked wrong ({hi - lo:.1f} units vs head joint {head_h_c:.1f}); "
                "estimating her height from the skeleton."
            )
    elif opts.chell_height and opts.chell_height > head_up - c_floor:
        c_height = opts.chell_height
    if c_height is None:
        c_height = (head_up - c_floor) * 1.15  # the head joint sits at ~87% of a person's height
    warnings.append(f"Chell: up axis {np.round(c_up, 3).tolist()}, height {c_height:.1f} units")

    # ---- VRM basis --------------------------------------------------------
    hv = vrm.humanoid
    vpos_gl = {i: n.world[:3, 3] for i, n in enumerate(vrm.nodes)}
    v_left = _horizontal(vpos_gl[hv["leftUpperLeg"]] - vpos_gl[hv["rightUpperLeg"]], 1)
    if not v_left.any():
        raise ConversionError("VRM legs are at the same position; cannot work out which way the avatar faces")
    v_up = np.array([0.0, 1.0, 0.0])
    v_fwd = normalize(np.cross(v_left, v_up))
    basis = np.vstack([v_fwd, v_left, v_up])  # glTF -> canonical
    m3 = chell_frame @ basis

    all_pos = np.concatenate([p.positions for p in vrm.primitives])
    floor = float(all_pos[:, 1].min())
    head_h = float(vpos_gl[hv["head"]][1]) - floor
    if float(all_pos[:, 1].max()) - floor <= 1e-6 or head_h <= 1e-6:
        raise ConversionError("VRM has zero height")
    chell_head = head_up - c_floor
    # Rough scale first (reposing only rotates, so it is scale independent);
    # the exact height match happens after reposing.
    scale = chell_head / head_h
    offset = chell_frame @ np.array([0.0, 0.0, -floor * scale]) + c_up * c_floor

    def to_game_pos(p):
        return scale * (m3 @ p) + offset

    # node rest positions in game space
    n_nodes = len(vrm.nodes)
    node_pos = [to_game_pos(n.world[:3, 3]) for n in vrm.nodes]

    # ---- repose to Chell's rest pose -------------------------------------
    deltas = [np.eye(4) for _ in range(n_nodes)]
    depth = [0] * n_nodes
    for i in range(n_nodes):
        d, p = 0, vrm.nodes[i].parent
        while p >= 0:
            d += 1
            p = vrm.nodes[p].parent
        depth[i] = d

    def subtree(i):
        out, stack = [], [i]
        while stack:
            k = stack.pop()
            out.append(k)
            stack.extend(vrm.nodes[k].children)
        return out

    def cur_pos(i):
        return (deltas[i] @ np.append(node_pos[i], 1.0))[:3]

    spine_chain = [s for s in ("spine", "chest", "upperChest", "neck") if s in mapping]

    def child_slot(slot):
        if slot in spine_chain[:-1]:
            return spine_chain[spine_chain.index(slot) + 1]
        return primary_child(slot)

    order = sorted((s for s in mapping if s != "hips"), key=lambda s: depth[hv[s]])
    for slot in order:
        cs = child_slot(slot)
        if cs is None or cs not in mapping or cs not in hv:
            continue
        b, c = hv[slot], hv[cs]
        p_b = cur_pos(b)
        d_v = cur_pos(c) - p_b
        d_c = c_pos[mapping[cs]] - c_pos[mapping[slot]]
        if np.linalg.norm(d_v) < 1e-6 or np.linalg.norm(d_c) < 1e-6:
            continue
        rot = rotation_between(d_v, d_c)
        tw = twist_reference(slot)
        if tw and tw[0] in mapping and tw[1] in mapping and tw[0] in hv and tw[1] in hv:
            sec_v = rot @ (cur_pos(hv[tw[1]]) - cur_pos(hv[tw[0]]))
            sec_c = c_pos[mapping[tw[1]]] - c_pos[mapping[tw[0]]]
            rot = twist_to_align(d_c, sec_v, sec_c) @ rot
        xf = rotate_about_point(rot, p_b)
        for k in subtree(b):
            deltas[k] = xf @ deltas[k]

    new_pos = [cur_pos(i) for i in range(n_nodes)]

    # pose the mesh with the avatar's own skin weights
    mats = np.stack(deltas)
    posed = []
    for prim in vrm.primitives:
        pos = scale * (prim.positions @ m3.T) + offset
        nrm = prim.normals @ m3.T
        blend = np.einsum("nk,nkij->nij", prim.weights, mats[prim.joints])
        pos = np.einsum("nij,nj->ni", blend[:, :3, :3], pos) + blend[:, :3, 3]
        nrm = np.einsum("nij,nj->ni", blend[:, :3, :3], nrm)
        nl = np.linalg.norm(nrm, axis=1, keepdims=True)
        nl[nl < 1e-12] = 1.0
        posed.append((pos, nrm / nl))

    # exact height match on the posed avatar (Chell's height is measured in her rest pose too)
    zs = np.concatenate([p[0] @ c_up for p in posed])
    p_floor, p_top = float(zs.min()), float(zs.max())
    k = c_height / (p_top - p_floor) * opts.height_scale
    pivot = c_up * p_floor
    floor_pt = c_up * c_floor
    new_pos = [k * (p - pivot) + floor_pt for p in new_pos]
    posed = [(k * (pos - pivot) + floor_pt, nrm) for pos, nrm in posed]
    scale *= k

    # ---- output skeleton: Chell bones ------------------------------------
    n_chell = len(chell.bones)
    chell_of_node: dict[int, int] = {}
    for slot, ci in mapping.items():
        chell_of_node[hv[slot]] = ci
    node_of_chell = {ci: n for n, ci in chell_of_node.items()}

    c_children: list[list[int]] = [[] for _ in range(n_chell)]
    for i, p in enumerate(c_parents):
        if 0 <= p < n_chell and p != i:
            c_children[p].append(i)

    def nearest_mapped_descendant(i):
        frontier = list(c_children[i])
        while frontier:
            hits = [k for k in frontier if k in node_of_chell]
            if hits:
                return min(hits, key=lambda k: np.linalg.norm(c_pos[k] - c_pos[i]))
            frontier = [g for k in frontier for g in c_children[k]]
        return None

    def mapped_ancestor(i):
        p = c_parents[i]
        while p >= 0:
            if p in node_of_chell:
                return p
            p = c_parents[p]
        return None

    out_world: list[np.ndarray | None] = [None] * n_chell
    order_c = sorted(range(n_chell), key=lambda i: _chain_depth(c_parents, i))
    for i in order_c:
        rot = c_world[i][:3, :3]
        if i == mapping["hips"] and up_of(c_pos[i]) - c_floor < 0.3 * chell_head:
            # A "root" bone that sits at the floor (not at the pelvis) keeps Chell's position,
            # otherwise the proportion delta would lift the whole avatar off the ground.
            pos = c_pos[i].copy()
        elif i in node_of_chell:
            pos = new_pos[node_of_chell[i]]
        else:
            anc, desc = mapped_ancestor(i), nearest_mapped_descendant(i)
            par = c_parents[i]
            par_child = _mapped_child_of(par, c_children, node_of_chell) if 0 <= par < n_chell and par in node_of_chell else None
            if opts.twist_weights and desc is None and par_child is not None and out_world[par] is not None:
                # Helper bone hanging off a mapped limb bone (e.g. a twist joint): keep its
                # relative place along the limb, scaled to the avatar's limb length.
                seg_c = c_pos[par_child] - c_pos[par]
                L2 = float(np.dot(seg_c, seg_c))
                rel = c_pos[i] - c_pos[par]
                t = float(np.dot(rel, seg_c) / L2) if L2 > 1e-9 else 0.0
                perp = rel - t * seg_c
                seg_o = new_pos[node_of_chell[par_child]] - out_world[par][:3, 3]
                pos = out_world[par][:3, 3] + t * seg_o + perp
            elif anc is not None and desc is not None:
                a, d = c_pos[anc], c_pos[desc]
                la, ld = np.linalg.norm(c_pos[i] - a), np.linalg.norm(d - c_pos[i])
                t = la / (la + ld) if la + ld > 1e-6 else 0.5
                pos = (1 - t) * new_pos[node_of_chell[anc]] + t * new_pos[node_of_chell[desc]]
            elif 0 <= par < n_chell and out_world[par] is not None:
                pos = out_world[par][:3, 3] + (c_pos[i] - c_pos[par])
            else:
                pos = c_pos[i].copy()
        out_world[i] = make_mat4(rot, pos)

    bones = [OutBone(c_names[i], c_parents[i], out_world[i], "chell") for i in range(n_chell)]  # type: ignore[arg-type]

    # ---- which VRM nodes become extra bones -------------------------------
    used = set()
    for prim in vrm.primitives:
        used.update(np.unique(prim.joints[prim.weights > 1e-4]).tolist())
    # Only bones that move vertices (and their ancestors) are kept: spring bones with
    # nothing attached would just use up Source's 128-bone budget.
    needed = used - set(chell_of_node)
    extra_set = set()
    for n in needed:
        k = n
        while k >= 0 and k not in chell_of_node and k not in extra_set:
            extra_set.add(k)
            k = vrm.nodes[k].parent

    # Drop pure-structure roots above everything (e.g. "Root"/armature) that carry no weights.
    hips_node = hv["hips"]
    ancestors_of_hips = set()
    k = vrm.nodes[hips_node].parent
    while k >= 0:
        ancestors_of_hips.add(k)
        k = vrm.nodes[k].parent
    extra_set -= {k for k in ancestors_of_hips if k not in used}

    # Weight totals for collapse priority
    weight_sum = np.zeros(n_nodes)
    for prim in vrm.primitives:
        np.add.at(weight_sum, prim.joints.reshape(-1), prim.weights.reshape(-1))

    # Collapse leaf extras until we fit the bone budget.
    budget = opts.max_bones - n_chell
    if budget < 0:
        raise ConversionError(f"Chell's skeleton alone has {n_chell} bones (limit {opts.max_bones})")
    merged_into: dict[int, int] = {}

    def extra_parent(k):
        p = vrm.nodes[k].parent
        while p >= 0 and p not in extra_set and p not in chell_of_node:
            p = vrm.nodes[p].parent
        return p

    if len(extra_set) > budget:
        warnings.append(f"Avatar has {len(extra_set)} extra bones; merging the smallest into their parents to fit Source's {opts.max_bones}-bone limit.")
    while len(extra_set) > budget:
        leaves = [k for k in extra_set if not any(c in extra_set for c in vrm.nodes[k].children)]
        victim = min(leaves, key=lambda k: (weight_sum[k], -depth[k]))
        extra_set.discard(victim)
        merged_into[victim] = extra_parent(victim)

    extras = sorted(extra_set, key=lambda k: (depth[k], k))
    taken = {n.lower() for n in c_names}
    out_of_node: dict[int, int] = dict(chell_of_node)
    root_fallback = mapping["hips"]
    for k in extras:
        name = _sanitize_bone_name(vrm.nodes[k].name, k, taken)
        p = extra_parent(k)
        parent_out = out_of_node.get(p, root_fallback) if p >= 0 else root_fallback
        out_of_node[k] = len(bones)
        bones.append(OutBone(name, parent_out, make_mat4(np.eye(3), new_pos[k]), "vrm"))

    # Orient extra bones: local +Z towards their first child (Source jigglebones use +Z).
    for k in extras:
        bi = out_of_node[k]
        par_rot = bones[bones[bi].parent].world[:3, :3]
        kids = [c for c in vrm.nodes[k].children if c in out_of_node and bones[out_of_node[c]].source == "vrm"]
        if kids:
            d = new_pos[kids[0]] - new_pos[k]
            rot = rotation_between(par_rot @ np.array([0.0, 0.0, 1.0]), d) @ par_rot
            if opts.jiggle and k in vrm.spring_joints:
                bones[bi].jiggle_length = float(np.linalg.norm(d))
        else:
            rot = par_rot
        bones[bi].world[:3, :3] = rot

    # ---- meshes ----------------------------------------------------------
    def resolve(node: int) -> int:
        seen = 0
        while node not in out_of_node:
            if node in merged_into:
                node = merged_into[node]
            else:
                node = vrm.nodes[node].parent if node >= 0 else -1
            seen += 1
            if node < 0 or seen > 1000:
                return root_fallback
        return out_of_node[node]

    resolve_cache: dict[int, int] = {}
    meshes: list[OutMesh] = []
    twist_profiles = (
        _twist_profiles(opts.chell_skin, c_pos, c_children, node_of_chell, c_names, warnings) if opts.twist_weights else {}
    )
    for pi, prim in enumerate(vrm.primitives):
        pos, nrm = posed[pi]

        # remap joints to output bones, merge duplicates, keep the top 3
        n = len(pos)
        ob = np.zeros((n, 4), dtype=np.int64)
        for j in range(4):
            col = prim.joints[:, j]
            for u in np.unique(col):
                u = int(u)
                if u not in resolve_cache:
                    resolve_cache[u] = resolve(u)
                ob[col == u, j] = resolve_cache[u]
        bw = prim.weights.copy()
        for a in range(4):
            for b in range(a + 1, 4):
                same = (ob[:, a] == ob[:, b]) & (bw[:, b] > 0)
                bw[same, a] += bw[same, b]
                bw[same, b] = 0.0
        order_w = np.argsort(-bw, axis=1)[:, :MAX_WEIGHTS]
        top_b = np.take_along_axis(ob, order_w, axis=1)
        top_w = np.take_along_axis(bw, order_w, axis=1)
        s = top_w.sum(axis=1, keepdims=True)
        zero = s[:, 0] <= 1e-8
        top_w[zero] = [1.0, 0.0, 0.0]
        top_b[zero, 0] = root_fallback
        s[zero] = 1.0
        top_w = top_w / s
        if twist_profiles:
            top_b, top_w = _apply_twist_profiles(pos, top_b, top_w, twist_profiles, bones)
        meshes.append(OutMesh(prim.mesh_name, prim.material, pos, nrm, prim.uvs.copy(), top_b, top_w, prim.indices.copy()))

    return ConversionResult(
        bones=bones,
        meshes=meshes,
        chell_world=c_world,
        n_chell=n_chell,
        scale=scale,
        mapping={slot: c_names[ci] for slot, ci in sorted(mapping.items())},
        warnings=warnings,
    )


def _chain_depth(parents: list[int], i: int) -> int:
    d = 0
    p = parents[i]
    while p >= 0 and d < 512:
        d += 1
        p = parents[p]
    return d


def _mapped_child_of(i, c_children, node_of_chell):
    kids = [k for k in c_children[i] if k in node_of_chell]
    return kids[0] if kids else None


TWIST_BINS = 12


def _twist_profiles(chell_skin, c_pos, c_children, node_of_chell, c_names, warnings):
    """How Chell's own skin is shared between a limb bone and its twist helper bones.

    Returns {limb bone: (child bone, [family bones], shares[TWIST_BINS, len(family)])}
    where shares are measured along the limb (0 = limb bone joint, 1 = child joint).
    """
    if chell_skin is None:
        return {}
    pos, bones, weights = chell_skin
    out = {}
    for b, kids in enumerate(c_children):
        if b not in node_of_chell:
            continue
        child = _mapped_child_of(b, c_children, node_of_chell)
        helpers = [k for k in kids if k not in node_of_chell and not c_children[k]]
        if child is None or not helpers:
            continue
        fam = [b] + helpers
        famw = np.zeros((len(pos), len(fam)))
        for j, fb in enumerate(fam):
            famw[:, j] = (weights * (bones == fb)).sum(axis=1)
        tot = famw.sum(axis=1)
        sel = tot > 0.05
        if not np.any(famw[sel][:, 1:].sum(axis=1) > 0.05):
            continue  # helpers carry no skin: nothing to copy
        seg = c_pos[child] - c_pos[b]
        L2 = float(np.dot(seg, seg))
        if L2 < 1e-9:
            continue
        t = np.clip((pos[sel] - c_pos[b]) @ seg / L2, 0.0, 0.999)
        bins = (t * TWIST_BINS).astype(int)
        share = np.zeros((TWIST_BINS, len(fam)))
        for k in range(TWIST_BINS):
            m = bins == k
            if m.any():
                fw = famw[sel][m]
                share[k] = fw.sum(axis=0) / max(fw.sum(), 1e-9)
        # fill empty bins from the nearest filled one
        filled = [k for k in range(TWIST_BINS) if share[k].sum() > 0]
        if not filled:
            continue
        for k in range(TWIST_BINS):
            if share[k].sum() == 0:
                share[k] = share[min(filled, key=lambda f: abs(f - k))]
        out[b] = (child, fam, share)
    if out:
        warnings.append("Twist bones copied from Chell: " + ", ".join(
            f"{c_names[b]}+{'/'.join(c_names[h] for h in fam[1:])}" for b, (_, fam, _) in out.items()))
    return out


def _apply_twist_profiles(pos, top_b, top_w, profiles, bones):
    """Split weights on limb bones between the limb bone and its twist helpers like Chell's skin."""
    n = len(pos)
    extra_b = [top_b[:, k] for k in range(top_b.shape[1])]
    extra_w = [top_w[:, k] for k in range(top_w.shape[1])]
    cols_b, cols_w = [], []
    for k in range(top_b.shape[1]):
        bk, wk = top_b[:, k].copy(), top_w[:, k].copy()
        for limb, (child, fam, share) in profiles.items():
            m = (bk == limb) & (wk > 0)
            if not m.any():
                continue
            p0 = bones[limb].world[:3, 3]
            seg = bones[child].world[:3, 3] - p0
            L2 = float(np.dot(seg, seg))
            if L2 < 1e-9:
                continue
            t = np.clip((pos[m] - p0) @ seg / L2, 0.0, 0.999)
            sh = share[(t * TWIST_BINS).astype(int)]  # (m, len(fam))
            base_w = wk[m].copy()
            wk[m] = base_w * sh[:, 0]
            for j, fb in enumerate(fam[1:], start=1):
                nb = np.full(n, fb)
                nw = np.zeros(n)
                nw[m] = base_w * sh[:, j]
                cols_b.append(nb)
                cols_w.append(nw)
        extra_b[k], extra_w[k] = bk, wk
    if not cols_b:
        return top_b, top_w
    allb = np.stack(extra_b + cols_b, axis=1)
    allw = np.stack(extra_w + cols_w, axis=1)
    # merge duplicates, keep the strongest three, renormalize
    for a in range(allb.shape[1]):
        for c in range(a + 1, allb.shape[1]):
            same = (allb[:, a] == allb[:, c]) & (allw[:, c] > 0)
            allw[same, a] += allw[same, c]
            allw[same, c] = 0.0
    order = np.argsort(-allw, axis=1)[:, :MAX_WEIGHTS]
    nb = np.take_along_axis(allb, order, axis=1)
    nw = np.take_along_axis(allw, order, axis=1)
    s = nw.sum(axis=1, keepdims=True)
    s[s <= 1e-8] = 1.0
    return nb, nw / s
