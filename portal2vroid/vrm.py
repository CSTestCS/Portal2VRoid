"""Load VRM 0.x / 1.0 avatars (binary glTF) into plain numpy structures."""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass, field

import numpy as np

from .mathutil import trs_to_mat4

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

_COMPONENT = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}

# Canonical humanoid bone names (VRM 1.0 spelling). VRM 0.x thumb names are remapped.
VRM0_THUMB_RENAME = {
    "leftThumbProximal": "leftThumbMetacarpal",
    "leftThumbIntermediate": "leftThumbProximal",
    "rightThumbProximal": "rightThumbMetacarpal",
    "rightThumbIntermediate": "rightThumbProximal",
}


@dataclass
class Node:
    index: int
    name: str
    parent: int
    children: list[int]
    local: np.ndarray
    world: np.ndarray = field(default_factory=lambda: np.eye(4))
    mesh: int | None = None
    skin: int | None = None


@dataclass
class Primitive:
    positions: np.ndarray  # (N,3) skinned rest pose, glTF space (meters)
    normals: np.ndarray  # (N,3)
    uvs: np.ndarray  # (N,2)
    joints: np.ndarray  # (N,4) node indices
    weights: np.ndarray  # (N,4)
    indices: np.ndarray  # (M,3)
    material: int
    mesh_name: str


@dataclass
class Material:
    name: str
    image: int | None
    base_color: tuple[float, float, float, float]
    alpha_mode: str  # OPAQUE / MASK / BLEND
    alpha_cutoff: float
    double_sided: bool


@dataclass
class VRMModel:
    name: str
    version: str  # "0.x" or "1.0"
    nodes: list[Node]
    humanoid: dict[str, int]
    primitives: list[Primitive]
    materials: list[Material]
    images: list[tuple[bytes, str]]  # (data, mime)
    spring_joints: set[int]
    title: str = ""
    author: str = ""

    def children_of(self, i: int) -> list[int]:
        return self.nodes[i].children


class GLTFReader:
    def __init__(self, data: bytes):
        magic, version, _length = struct.unpack_from("<III", data, 0)
        if magic != GLB_MAGIC:
            raise ValueError("not a binary glTF/VRM file")
        if version != 2:
            raise ValueError(f"unsupported glTF version {version}")
        off = 12
        self.json = None
        self.bin = b""
        while off + 8 <= len(data):
            clen, ctype = struct.unpack_from("<II", data, off)
            chunk = data[off + 8 : off + 8 + clen]
            if ctype == CHUNK_JSON:
                self.json = json.loads(chunk.decode("utf-8"))
            elif ctype == CHUNK_BIN and not self.bin:
                self.bin = chunk
            off += 8 + clen
        if self.json is None:
            raise ValueError("glTF JSON chunk missing")

    def buffer_view(self, idx: int) -> bytes:
        bv = self.json["bufferViews"][idx]
        if bv.get("buffer", 0) != 0:
            raise ValueError("external buffers are not supported (use a .vrm/.glb file)")
        start = bv.get("byteOffset", 0)
        return self.bin[start : start + bv["byteLength"]]

    def accessor(self, idx: int) -> np.ndarray:
        acc = self.json["accessors"][idx]
        dtype = np.dtype(_COMPONENT[acc["componentType"]])
        ncomp = _NCOMP[acc["type"]]
        count = acc["count"]
        if "bufferView" in acc:
            bv = self.json["bufferViews"][acc["bufferView"]]
            raw = self.buffer_view(acc["bufferView"])
            start = acc.get("byteOffset", 0)
            stride = bv.get("byteStride", 0) or dtype.itemsize * ncomp
            if stride == dtype.itemsize * ncomp:
                arr = np.frombuffer(raw, dtype=dtype, count=count * ncomp, offset=start).reshape(count, ncomp)
            else:
                arr = np.empty((count, ncomp), dtype=dtype)
                for i in range(count):
                    arr[i] = np.frombuffer(raw, dtype=dtype, count=ncomp, offset=start + i * stride)
            arr = arr.copy()
        else:
            arr = np.zeros((count, ncomp), dtype=dtype)
        sparse = acc.get("sparse")
        if sparse:
            n = sparse["count"]
            ii = sparse["indices"]
            idt = np.dtype(_COMPONENT[ii["componentType"]])
            ind = np.frombuffer(self.buffer_view(ii["bufferView"]), dtype=idt, count=n, offset=ii.get("byteOffset", 0))
            vv = sparse["values"]
            vals = np.frombuffer(
                self.buffer_view(vv["bufferView"]), dtype=dtype, count=n * ncomp, offset=vv.get("byteOffset", 0)
            ).reshape(n, ncomp)
            arr[ind.astype(np.int64)] = vals
        if acc.get("normalized") and dtype.kind in "iu":
            maxv = float(np.iinfo(dtype).max)
            arr = np.maximum(arr.astype(np.float64) / maxv, -1.0)
        if ncomp == 1:
            arr = arr.reshape(count)
        return arr


def _node_local(n: dict) -> np.ndarray:
    if "matrix" in n:
        return np.array(n["matrix"], dtype=float).reshape(4, 4).T  # glTF is column-major
    return trs_to_mat4(n.get("translation"), n.get("rotation"), n.get("scale"))


def load_vrm(path_or_bytes) -> VRMModel:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
        fname = "avatar"
    else:
        with open(path_or_bytes, "rb") as f:
            data = f.read()
        fname = str(path_or_bytes)
    g = GLTFReader(data)
    js = g.json
    ext = js.get("extensions", {})

    # --- nodes -----------------------------------------------------------
    raw_nodes = js.get("nodes", [])
    nodes = [
        Node(i, n.get("name") or f"node{i}", -1, list(n.get("children", [])), _node_local(n), mesh=n.get("mesh"), skin=n.get("skin"))
        for i, n in enumerate(raw_nodes)
    ]
    for n in nodes:
        for c in n.children:
            nodes[c].parent = n.index

    def compute_world(i: int, parent_world: np.ndarray):
        nodes[i].world = parent_world @ nodes[i].local
        for c in nodes[i].children:
            compute_world(c, nodes[i].world)

    for n in nodes:
        if n.parent < 0:
            compute_world(n.index, np.eye(4))

    # --- humanoid ----------------------------------------------------------
    humanoid: dict[str, int] = {}
    version = ""
    title = author = ""
    if "VRMC_vrm" in ext:
        version = "1.0"
        vrm = ext["VRMC_vrm"]
        for bone, info in vrm.get("humanoid", {}).get("humanBones", {}).items():
            if isinstance(info, dict) and "node" in info:
                humanoid[bone] = int(info["node"])
        meta = vrm.get("meta", {})
        title = meta.get("name", "")
        authors = meta.get("authors") or []
        author = authors[0] if authors else ""
    elif "VRM" in ext:
        version = "0.x"
        vrm = ext["VRM"]
        for hb in vrm.get("humanoid", {}).get("humanBones", []):
            bone = hb.get("bone")
            node = hb.get("node", -1)
            if bone and node is not None and node >= 0:
                humanoid[VRM0_THUMB_RENAME.get(bone, bone)] = int(node)
        meta = vrm.get("meta", {})
        title = meta.get("title", "")
        author = meta.get("author", "")
    else:
        raise ValueError("this file has no VRM humanoid data (VRM 0.x or 1.0 required)")
    for req in ("hips", "spine", "head", "leftUpperLeg", "rightUpperLeg", "leftUpperArm", "rightUpperArm"):
        if req not in humanoid:
            raise ValueError(f"VRM humanoid is missing required bone '{req}'")

    # --- spring bones -------------------------------------------------------
    spring: set[int] = set()
    if "VRMC_springBone" in ext:
        for s in ext["VRMC_springBone"].get("springs", []):
            for j in s.get("joints", []):
                if "node" in j:
                    spring.add(int(j["node"]))
    if "VRM" in ext:
        for grp in ext["VRM"].get("secondaryAnimation", {}).get("boneGroups", []):
            stack = list(grp.get("bones", []))
            while stack:
                b = stack.pop()
                if b in spring or not (0 <= b < len(nodes)):
                    continue
                spring.add(b)
                stack.extend(nodes[b].children)
    spring -= set(humanoid.values())

    # --- materials / images -----------------------------------------------
    textures = js.get("textures", [])
    images = []
    for img in js.get("images", []):
        if "bufferView" in img:
            images.append((g.buffer_view(img["bufferView"]), img.get("mimeType", "image/png")))
        else:
            images.append((b"", img.get("mimeType", "")))
    vrm0_props = ext.get("VRM", {}).get("materialProperties", []) if version == "0.x" else []
    materials = []
    for mi, m in enumerate(js.get("materials", [])):
        pbr = m.get("pbrMetallicRoughness", {})
        tex = pbr.get("baseColorTexture", {}).get("index")
        image = None
        if tex is not None and 0 <= tex < len(textures):
            image = textures[tex].get("source")
        if image is None and mi < len(vrm0_props):
            t = vrm0_props[mi].get("textureProperties", {}).get("_MainTex")
            if t is not None and 0 <= t < len(textures):
                image = textures[t].get("source")
        color = tuple(pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]))
        materials.append(
            Material(
                name=m.get("name") or f"material{mi}",
                image=image,
                base_color=color,  # type: ignore[arg-type]
                alpha_mode=m.get("alphaMode", "OPAQUE"),
                alpha_cutoff=float(m.get("alphaCutoff", 0.5)),
                double_sided=bool(m.get("doubleSided", False)),
            )
        )
    if not materials:
        materials.append(Material("default", None, (1, 1, 1, 1), "OPAQUE", 0.5, False))

    # --- meshes ------------------------------------------------------------
    primitives: list[Primitive] = []
    skins = js.get("skins", [])
    meshes = js.get("meshes", [])
    for node in nodes:
        if node.mesh is None:
            continue
        mesh = meshes[node.mesh]
        skin = skins[node.skin] if node.skin is not None and node.skin < len(skins) else None
        joint_nodes = np.array(skin["joints"], dtype=np.int64) if skin else None
        if skin is not None:
            if "inverseBindMatrices" in skin:
                ibm = g.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1).astype(float)
            else:
                ibm = np.repeat(np.eye(4)[None], len(joint_nodes), axis=0)
            skin_mats = np.array([nodes[j].world for j in joint_nodes]) @ ibm
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            pos = g.accessor(attrs["POSITION"]).astype(float)
            n = len(pos)
            nrm = g.accessor(attrs["NORMAL"]).astype(float) if "NORMAL" in attrs else None
            uv = g.accessor(attrs["TEXCOORD_0"]).astype(float) if "TEXCOORD_0" in attrs else np.zeros((n, 2))
            if "indices" in prim:
                idx = g.accessor(prim["indices"]).astype(np.int64)
            else:
                idx = np.arange(n, dtype=np.int64)
            idx = idx[: len(idx) - len(idx) % 3].reshape(-1, 3)
            if skin is not None and "JOINTS_0" in attrs and "WEIGHTS_0" in attrs:
                jl = g.accessor(attrs["JOINTS_0"]).astype(np.int64)
                wt = g.accessor(attrs["WEIGHTS_0"]).astype(float)
                jl = np.clip(jl, 0, len(joint_nodes) - 1)
                s = wt.sum(axis=1, keepdims=True)
                s[s <= 1e-8] = 1.0
                wt = wt / s
                # skin to rest pose (glTF spec: node transform of skinned mesh is ignored)
                mats = np.einsum("nk,nkij->nij", wt, skin_mats[jl])
                p4 = np.concatenate([pos, np.ones((n, 1))], axis=1)
                pos = np.einsum("nij,nj->ni", mats, p4)[:, :3]
                if nrm is not None:
                    nrm = np.einsum("nij,nj->ni", mats[:, :3, :3], nrm)
                joints = joint_nodes[jl]
            else:
                # rigid mesh: attach to its own node (or nearest ancestor)
                w = node.world
                pos = (w[:3, :3] @ pos.T).T + w[:3, 3]
                if nrm is not None:
                    nrm = (w[:3, :3] @ nrm.T).T
                owner = node.index
                joints = np.full((n, 4), owner, dtype=np.int64)
                wt = np.zeros((n, 4))
                wt[:, 0] = 1.0
            if nrm is None:
                nrm = _vertex_normals(pos, idx)
            nl = np.linalg.norm(nrm, axis=1, keepdims=True)
            nl[nl < 1e-12] = 1.0
            nrm = nrm / nl
            primitives.append(
                Primitive(pos, nrm, uv, joints, wt, idx, int(prim.get("material", 0)) if materials else 0, mesh.get("name", f"mesh{node.mesh}"))
            )
    if not primitives:
        raise ValueError("VRM contains no triangle meshes")

    name = title or fname.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return VRMModel(name, version, nodes, humanoid, primitives, materials, images, spring, title, author)


def _vertex_normals(pos: np.ndarray, idx: np.ndarray) -> np.ndarray:
    nrm = np.zeros_like(pos)
    a, b, c = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(nrm, idx[:, k], fn)
    return nrm


def decode_image(data: bytes):
    """Decode PNG/JPEG bytes into a PIL RGBA image."""
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGBA")
