"""Synthetic test data: a ValveBiped-like Chell .mdl/.vvd and VRM 0.x / 1.0 avatars."""

from __future__ import annotations

import io
import json
import math
import struct

import numpy as np

from portal2vroid.mathutil import make_mat4, mat3_to_quat, rigid_inverse, rot_z, rotation_between

# --------------------------------------------------------------------------- Chell


def _bone_specs():
    """(name, parent name, world position) for an A-posed ValveBiped skeleton (+X fwd, +Y left)."""
    s = []

    def add(name, parent, pos):
        s.append((name, parent, np.array(pos, dtype=float)))

    vb = "ValveBiped.Bip01_"
    add(vb + "Pelvis", None, (0, 0, 38))
    add(vb + "Spine", vb + "Pelvis", (0, 0, 41))
    add(vb + "Spine1", vb + "Spine", (0.3, 0, 45))
    add(vb + "Spine2", vb + "Spine1", (0.5, 0, 49))
    add(vb + "Spine4", vb + "Spine2", (0.4, 0, 54))
    add(vb + "Neck1", vb + "Spine4", (0.2, 0, 59))
    add(vb + "Head1", vb + "Neck1", (0.6, 0, 62))
    for side, sy in (("L", 1), ("R", -1)):
        add(vb + f"{side}_Clavicle", vb + "Spine4", (0, sy * 1.0, 57))
        add(vb + f"{side}_UpperArm", vb + f"{side}_Clavicle", (0, sy * 6.5, 57))
        # A-pose: arm 45 degrees down
        el = np.array([0, sy * 6.5, 57]) + 10.5 * np.array([0, sy * math.cos(math.radians(45)), -math.sin(math.radians(45))])
        add(vb + f"{side}_Forearm", vb + f"{side}_UpperArm", el)
        wr = el + 10 * np.array([0, sy * math.cos(math.radians(45)), -math.sin(math.radians(45))])
        add(vb + f"{side}_Hand", vb + f"{side}_Forearm", wr)
        add(vb + f"{side}_Ulna", vb + f"{side}_Forearm", (el + wr) / 2)  # helper
        for f in range(5):
            base = wr + np.array([1.2 - f * 0.6, sy * 2.0, -2.0])
            d = np.array([0, sy * 0.7, -0.7])
            add(vb + f"{side}_Finger{f}", vb + f"{side}_Hand", base)
            add(vb + f"{side}_Finger{f}1", vb + f"{side}_Finger{f}", base + d * 1.5)
            add(vb + f"{side}_Finger{f}2", vb + f"{side}_Finger{f}1", base + d * 2.6)
        add(vb + f"{side}_Thigh", vb + "Pelvis", (0, sy * 3.8, 37))
        add(vb + f"{side}_Calf", vb + f"{side}_Thigh", (0.4, sy * 3.9, 20))
        add(vb + f"{side}_Foot", vb + f"{side}_Calf", (0, sy * 4.0, 3.5))
        add(vb + f"{side}_Toe0", vb + f"{side}_Foot", (4.5, sy * 4.1, 0.8))
    add("ValveBiped.weapon_bone", vb + "R_Hand", s[-1][2] * 0 + np.array([0, -18, 36]))
    return s


def chell_world_bones():
    specs = _bone_specs()
    names = [n for n, _, _ in specs]
    parents = [names.index(p) if p else -1 for _, p, _ in specs]
    children = {i: [j for j, p in enumerate(parents) if p == i] for i in range(len(specs))}
    worlds = []
    for i, (n, _, pos) in enumerate(specs):
        kids = children[i]
        if kids:
            d = specs[kids[0]][2] - pos
            rot = rotation_between([1, 0, 0], d)
        else:
            rot = worlds[parents[i]][:3, :3] if parents[i] >= 0 else np.eye(3)
        # add some twist so orientations are non-trivial
        rot = rot @ rot_z(0.3) if "Finger" not in n else rot
        worlds.append(make_mat4(rot, pos))
    return names, parents, worlds


def make_chell_mdl() -> bytes:
    names, parents, worlds = chell_world_bones()
    nb = len(names)
    header_size = 408
    bone_index = header_size
    bones_end = bone_index + nb * 216
    # ik chains after bones
    chains = [("lfoot", "L"), ("rfoot", "R"), ("lhand", "L"), ("rhand", "R")]
    ik_index = bones_end
    ik_size = len(chains) * 16
    link_index = ik_index + ik_size
    links_size = len(chains) * 3 * 28
    inc_index = link_index + links_size
    str_start = inc_index + 8
    strings = bytearray()
    str_off = {}

    def sref(s):
        if s not in str_off:
            str_off[s] = len(strings)
            strings.extend(s.encode() + b"\0")
        return str_start + str_off[s]

    data = bytearray(str_start)
    struct.pack_into("<4si", data, 0, b"IDST", 49)
    data[12 : 12 + 13] = b"player/chell\0"
    struct.pack_into("<3f", data, 80, 0, 0, 64)
    struct.pack_into("<3f", data, 92, 0, 0, 36)
    struct.pack_into("<3f", data, 104, -16, -16, 0)
    struct.pack_into("<3f", data, 116, 16, 16, 72)
    struct.pack_into("<ii", data, 156, nb, bone_index)
    struct.pack_into("<i", data, 188, 20)
    struct.pack_into("<ii", data, 284, len(chains), ik_index)
    struct.pack_into("<ii", data, 336, 1, inc_index)
    for i in range(nb):
        base = bone_index + i * 216
        p = parents[i]
        local = rigid_inverse(worlds[p]) @ worlds[i] if p >= 0 else worlds[i]
        struct.pack_into("<ii", data, base, sref(names[i]) - base, p)
        struct.pack_into("<3f", data, base + 32, *local[:3, 3])
        struct.pack_into("<4f", data, base + 44, *mat3_to_quat(local[:3, :3]))
        p2b = rigid_inverse(worlds[i])
        struct.pack_into("<12f", data, base + 96, *p2b[:3, :].reshape(-1))
    for ci, (cname, side) in enumerate(chains):
        base = ik_index + ci * 16
        lb = link_index + ci * 3 * 28
        if "foot" in cname:
            bns = [f"ValveBiped.Bip01_{side}_Thigh", f"ValveBiped.Bip01_{side}_Calf", f"ValveBiped.Bip01_{side}_Foot"]
        else:
            bns = [f"ValveBiped.Bip01_{side}_UpperArm", f"ValveBiped.Bip01_{side}_Forearm", f"ValveBiped.Bip01_{side}_Hand"]
        struct.pack_into("<iiii", data, base, sref(cname) - base, 0, 3, lb - base)
        for j, bn in enumerate(bns):
            struct.pack_into("<i3f3f", data, lb + j * 28, names.index(bn), 0.7 if j == 0 else 0, 0.7 if j == 0 else 0, 0, 0, 0, 0)
    struct.pack_into("<ii", data, inc_index, 0, sref("models/player/chell/anims.mdl") - inc_index)
    data += strings
    struct.pack_into("<i", data, 76, len(data))
    return bytes(data)


def make_chell_vvd(top: float = 70.0) -> bytes:
    verts = np.zeros((3, 12), dtype=np.float32)
    verts[:, 6] = [0.0, 35.0, top]
    data = bytearray(64)
    struct.pack_into("<4si", data, 0, b"IDSV", 4)
    struct.pack_into("<ii", data, 56, 64, 64 + verts.nbytes)
    return bytes(data) + verts.tobytes()


# --------------------------------------------------------------------------- VRM


def _png(color=(200, 120, 160, 255), size=(64, 32), alpha_hole=False) -> bytes:
    from PIL import Image

    img = Image.new("RGBA", size, color)
    if alpha_hole:
        for x in range(10):
            for y in range(10):
                img.putpixel((x, y), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# humanoid joints in canonical avatar space: x = avatar's left, y = up, z = forward (meters)
def _vrm_joints():
    j = {}

    def add(name, parent, pos, human=None):
        j[name] = (parent, np.array(pos, dtype=float), human)

    add("Root", None, (0, 0, 0))
    add("J_Hips", "Root", (0, 0.80, 0), "hips")
    add("J_Spine", "J_Hips", (0, 0.88, 0), "spine")
    add("J_Chest", "J_Spine", (0, 0.98, 0), "chest")
    add("J_UpperChest", "J_Chest", (0, 1.08, 0), "upperChest")
    add("J_Neck", "J_UpperChest", (0, 1.20, 0), "neck")
    add("J_Head", "J_Neck", (0, 1.27, 0), "head")
    add("J_Hair1", "J_Head", (0, 1.40, -0.08))
    add("J_Hair2", "J_Hair1", (0, 1.25, -0.12))
    add("J_Hair3", "J_Hair2", (0, 1.10, -0.13))
    for side, sx in (("L", 1), ("R", -1)):
        hs = "left" if side == "L" else "right"
        add(f"J_{side}_Shoulder", "J_UpperChest", (sx * 0.03, 1.16, 0), f"{hs}Shoulder")
        add(f"J_{side}_UpperArm", f"J_{side}_Shoulder", (sx * 0.12, 1.16, 0), f"{hs}UpperArm")
        add(f"J_{side}_LowerArm", f"J_{side}_UpperArm", (sx * 0.36, 1.16, 0), f"{hs}LowerArm")
        add(f"J_{side}_Hand", f"J_{side}_LowerArm", (sx * 0.58, 1.16, 0), f"{hs}Hand")
        for fi, fname in enumerate(("Thumb", "Index", "Middle", "Ring", "Little")):
            segs = ("Metacarpal", "Proximal", "Distal") if fi == 0 else ("Proximal", "Intermediate", "Distal")
            base = np.array([sx * 0.63, 1.16, 0.03 - fi * 0.015])
            if fi == 0:
                base = np.array([sx * 0.60, 1.15, 0.035])
            prev = f"J_{side}_Hand"
            for k, seg in enumerate(segs):
                nm = f"J_{side}_{fname}{k + 1}"
                add(nm, prev, base + np.array([sx * 0.025 * k, 0, 0]), f"{hs}{fname}{seg}")
                prev = nm
        add(f"J_{side}_UpperLeg", "J_Hips", (sx * 0.09, 0.78, 0), f"{hs}UpperLeg")
        add(f"J_{side}_LowerLeg", f"J_{side}_UpperLeg", (sx * 0.09, 0.44, 0), f"{hs}LowerLeg")
        add(f"J_{side}_Foot", f"J_{side}_LowerLeg", (sx * 0.09, 0.07, 0), f"{hs}Foot")
        add(f"J_{side}_Toes", f"J_{side}_Foot", (sx * 0.09, 0.01, 0.11), f"{hs}Toes")
        add(f"J_{side}_Skirt", "J_Hips", (sx * 0.12, 0.72, 0.05))
    return j


VRM0_THUMB = {"Metacarpal": "Proximal", "Proximal": "Intermediate"}


class _Bin:
    def __init__(self):
        self.data = bytearray()
        self.views = []
        self.accessors = []

    def add(self, arr: np.ndarray, comp: int, typ: str, target=None, minmax=False):
        while len(self.data) % 4:
            self.data.append(0)
        off = len(self.data)
        raw = np.ascontiguousarray(arr).tobytes()
        self.data += raw
        view = {"buffer": 0, "byteOffset": off, "byteLength": len(raw)}
        if target:
            view["target"] = target
        self.views.append(view)
        acc = {"bufferView": len(self.views) - 1, "componentType": comp, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            acc["min"] = arr.min(axis=0).tolist()
            acc["max"] = arr.max(axis=0).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def add_raw(self, raw: bytes):
        while len(self.data) % 4:
            self.data.append(0)
        off = len(self.data)
        self.data += raw
        self.views.append({"buffer": 0, "byteOffset": off, "byteLength": len(raw)})
        return len(self.views) - 1


def _box(center, half):
    c = np.array(center, dtype=float)
    h = np.array(half, dtype=float)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float) * h + c
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return corners, np.array(faces)


def make_vrm(version: str = "1.0", scale: float = 1.0) -> bytes:
    """A small but complete VRM avatar. VRM 1.0 faces +Z, VRM 0.x faces -Z."""
    joints = _vrm_joints()
    names = list(joints)
    flip = -1.0 if version == "0.x" else 1.0  # rotate 180 deg about Y for VRM0

    def to_gl(p):
        return np.array([p[0] * flip, p[1], p[2] * flip]) * scale

    nodes = []
    for n in names:
        parent, pos, _ = joints[n]
        ppos = to_gl(joints[parent][1]) if parent else np.zeros(3)
        nodes.append({"name": n, "translation": (to_gl(pos) - ppos).tolist(), "children": []})
    for i, n in enumerate(names):
        parent = joints[n][0]
        if parent:
            nodes[names.index(parent)]["children"].append(i)

    # mesh: a box around each segment, skinned to the segment's bone
    b = _Bin()
    pos_l, uv_l, j_l, w_l, idx_l = [], [], [], [], []
    hair_idx = []
    vcount = 0
    joint_list = [i for i, n in enumerate(names) if n != "Root"]
    for n in names:
        if n == "Root":
            continue
        parent, p, _ = joints[n]
        kids = [k for k in names if joints[k][0] == n]
        tip = joints[kids[0]][1] if kids else p + np.array([0, 0.05, 0])
        center = (p + tip) / 2
        half = np.maximum(np.abs(tip - p) / 2, 0.02)
        corners, faces = _box(center, half)
        jl = joint_list.index(names.index(n))
        pos_l.append(np.array([to_gl(c) for c in corners]))
        uv_l.append(np.tile([[0.25, 0.75]], (8, 1)))
        jj = np.zeros((8, 4), dtype=np.uint16)
        ww = np.zeros((8, 4), dtype=np.float32)
        jj[:, 0] = jl
        ww[:, 0] = 1.0
        if parent and parent != "Root":
            # soft blend with parent on half the verts
            jj[:4, 1] = joint_list.index(names.index(parent))
            ww[:4, 0] = 0.7
            ww[:4, 1] = 0.3
        j_l.append(jj)
        w_l.append(ww)
        tgt = hair_idx if n.startswith("J_Hair") else idx_l
        tgt.append(faces + vcount)
        vcount += 8
    P = np.concatenate(pos_l).astype(np.float32)
    UV = np.concatenate(uv_l).astype(np.float32)
    J = np.concatenate(j_l)
    W = np.concatenate(w_l)
    I_body = np.concatenate(idx_l).reshape(-1).astype(np.uint32)
    I_hair = np.concatenate(hair_idx).reshape(-1).astype(np.uint16)
    a_pos = b.add(P, 5126, "VEC3", 34962, minmax=True)
    a_uv = b.add(UV, 5126, "VEC2", 34962)
    a_j = b.add(J, 5123, "VEC4", 34962)
    a_w = b.add(W, 5126, "VEC4", 34962)
    a_ib = b.add(I_body, 5125, "SCALAR", 34963)
    a_ih = b.add(I_hair, 5123, "SCALAR", 34963)
    ibm = []
    for ji in joint_list:
        # world of each joint in glTF space (translation only)
        w = np.eye(4)
        w[:3, 3] = to_gl(joints[names[ji]][1])
        ibm.append(np.linalg.inv(w).T.reshape(-1))  # column-major
    a_ibm = b.add(np.array(ibm, dtype=np.float32), 5126, "MAT4")
    img0 = b.add_raw(_png())
    img1 = b.add_raw(_png((40, 40, 60, 255), alpha_hole=True))

    mesh_node = len(nodes)
    nodes.append({"name": "Body", "mesh": 0, "skin": 0})
    gltf = {
        "asset": {"version": "2.0", "generator": "portal2vroid-tests"},
        "scene": 0,
        "scenes": [{"nodes": [0, mesh_node]}],
        "nodes": nodes,
        "meshes": [
            {
                "name": "Body",
                "primitives": [
                    {"attributes": {"POSITION": a_pos, "TEXCOORD_0": a_uv, "JOINTS_0": a_j, "WEIGHTS_0": a_w}, "indices": a_ib, "material": 0},
                    {"attributes": {"POSITION": a_pos, "TEXCOORD_0": a_uv, "JOINTS_0": a_j, "WEIGHTS_0": a_w}, "indices": a_ih, "material": 1},
                ],
            }
        ],
        "skins": [{"joints": joint_list, "inverseBindMatrices": a_ibm}],
        "materials": [
            {"name": "Body_Skin", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}},
            {"name": "Hair (cutout)", "alphaMode": "MASK", "alphaCutoff": 0.4, "doubleSided": True, "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}, "baseColorFactor": [0.9, 0.8, 0.7, 1.0]}},
        ],
        "textures": [{"source": 0}, {"source": 1}],
        "images": [{"bufferView": img0, "mimeType": "image/png"}, {"bufferView": img1, "mimeType": "image/png"}],
        "bufferViews": b.views,
        "accessors": b.accessors,
        "buffers": [{"byteLength": len(b.data)}],
    }
    human = {joints[n][2]: names.index(n) for n in names if joints[n][2]}
    hair = [names.index("J_Hair1"), names.index("J_Hair2"), names.index("J_Hair3")]
    if version == "1.0":
        gltf["extensionsUsed"] = ["VRMC_vrm", "VRMC_springBone"]
        gltf["extensions"] = {
            "VRMC_vrm": {"specVersion": "1.0", "meta": {"name": "Test Avatar", "authors": ["tester"]}, "humanoid": {"humanBones": {k: {"node": v} for k, v in human.items()}}},
            "VRMC_springBone": {"specVersion": "1.0", "springs": [{"joints": [{"node": h} for h in hair]}]},
        }
    else:
        hb = []
        for k, v in human.items():
            name = k
            for side in ("left", "right"):
                for a, bb in (("ThumbProximal", "ThumbIntermediate"), ("ThumbMetacarpal", "ThumbProximal")):
                    if k == f"{side}{a}":
                        name = f"{side}{bb}"
            hb.append({"bone": name, "node": v})
        gltf["extensionsUsed"] = ["VRM"]
        gltf["extensions"] = {
            "VRM": {
                "meta": {"title": "Test Avatar Zero", "author": "tester"},
                "humanoid": {"humanBones": hb},
                "secondaryAnimation": {"boneGroups": [{"bones": [hair[0]]}]},
                "materialProperties": [{"name": "Body_Skin", "shader": "VRM/MToon"}, {"name": "Hair", "shader": "VRM/MToon"}],
            }
        }
    js = json.dumps(gltf).encode()
    while len(js) % 4:
        js += b" "
    binc = bytes(b.data)
    while len(binc) % 4:
        binc += b"\0"
    total = 12 + 8 + len(js) + 8 + len(binc)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(js), 0x4E4F534A)
        + js
        + struct.pack("<II", len(binc), 0x004E4942)
        + binc
    )


# --------------------------------------------------------------------------- menus

INGAME_MENU_RES = r'''"Resource/UI/BaseModUI/InGameMainMenu.res"
{
	"InGameMainMenu"
	{
		"ControlName"		"Frame"
		"fieldName"		"InGameMainMenu"
		"xpos"		"0"
		"ypos"		"0"
		"wide"		"f0"
		"tall"		"f0"
	}
	"BtnReturnToGame"
	{
		"ControlName"		"BaseModHybridButton"
		"fieldName"		"BtnReturnToGame"
		"xpos"		"100"
		"ypos"		"200"
		"wide"		"220"
		"tall"		"20"
		"navUp"		"BtnExitToMainMenu"
		"navDown"		"BtnOptions"
		"labelText"		"#PORTAL2_InGameMainMenu_ReturnToGame"
		"style"		"MainMenuButton"
		"command"		"ReturnToGame"
		"ActivationType"	"1"
	}
	"BtnOptions"
	{
		"ControlName"		"BaseModHybridButton"
		"fieldName"		"BtnOptions"
		"xpos"		"100"
		"ypos"		"220"
		"wide"		"220"
		"tall"		"20"
		"navUp"		"BtnReturnToGame"
		"navDown"		"BtnExitToMainMenu"
		"tooltiptext"		"#L4D360UI_MainMenu_Options_Tip"
		"labelText"		"#PORTAL2_MainMenu_Options"
		"style"		"MainMenuButton"
		"command"		"FlmOptionsFlyout"
	}
	"BtnExitToMainMenu"
	{
		"ControlName"		"BaseModHybridButton"
		"fieldName"		"BtnExitToMainMenu"
		"xpos"		"100"
		"ypos"		"240"		[$WIN32]
		"wide"		"220"
		"tall"		"20"
		"navUp"		"BtnOptions"
		"navDown"		"BtnReturnToGame"
		"labelText"		"#PORTAL2_InGameMainMenu_ExitToMainMenu"
		"style"		"MainMenuButton"
		"command"		"ExitToMainMenu"
	}
	"FlmOptionsFlyout"
	{
		"ControlName"		"FlyoutMenu"
		"fieldName"		"FlmOptionsFlyout"
		"visible"		"0"
		"wide"		"0"
		"tall"		"0"
		"zpos"		"3"
		"InitialFocus"		"BtnVideo"
		"ResourceFile"		"resource/UI/basemodui/OptionsFlyout.res"
	}
}
'''

OPTIONS_FLYOUT_RES = r'''"Resource/UI/OptionsFlyout.res"
{
	"PnlBackground"
	{
		"ControlName"		"Panel"
		"fieldName"		"PnlBackground"
		"xpos"		"0"
		"ypos"		"0"
		"zpos"		"-1"
		"wide"		"160"
		"tall"		"70"
		"visible"		"1"
		"enabled"		"1"
		"paintbackground"		"1"
		"paintborder"		"1"
	}
	"BtnVideo"
	{
		"ControlName"		"BaseModHybridButton"
		"fieldName"		"BtnVideo"
		"xpos"		"0"
		"ypos"		"5"
		"wide"		"150"
		"tall"		"20"
		"labelText"		"#GameUI_Video"
		"style"		"FlyoutMenuButton"
		"command"		"Video"
	}
	"BtnAudio"
	{
		"ControlName"		"BaseModHybridButton"
		"fieldName"		"BtnAudio"
		"xpos"		"0"
		"ypos"		"25"
		"wide"		"150"
		"tall"		"20"
		"labelText"		"#GameUI_Audio"
		"style"		"FlyoutMenuButton"
		"command"		"Audio"
	}
}
'''

KB_ACT = '"blank"\t\t"=========================="\n"blank"\t\t"#Valve_Movement_Title"\n"+forward"\t\t"#Valve_Move_Forward"\n'
