"""Map VRM humanoid bones onto the bones of a Source skeleton (Chell / ValveBiped)."""

from __future__ import annotations

import re

SIDES = ("left", "right")
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")
THUMB_SEGS = ("Metacarpal", "Proximal", "Distal")
FINGER_SEGS = ("Proximal", "Intermediate", "Distal")

LIMB_SLOTS = ("Shoulder", "UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot", "Toes")


def finger_slot(side: str, finger: int, seg: int) -> str:
    segs = THUMB_SEGS if finger == 0 else FINGER_SEGS
    return f"{side}{FINGERS[finger]}{segs[seg]}"


# For reposing: slot -> slot whose joint the bone should point at.
def primary_child(slot: str) -> str | None:
    table = {
        "neck": "head",
    }
    if slot in table:
        return table[slot]
    for side in SIDES:
        if slot == f"{side}Shoulder":
            return f"{side}UpperArm"
        if slot == f"{side}UpperArm":
            return f"{side}LowerArm"
        if slot == f"{side}LowerArm":
            return f"{side}Hand"
        if slot == f"{side}Hand":
            return f"{side}MiddleProximal"
        if slot == f"{side}UpperLeg":
            return f"{side}LowerLeg"
        if slot == f"{side}LowerLeg":
            return f"{side}Foot"
        if slot == f"{side}Foot":
            return f"{side}Toes"
        for fi in range(5):
            for seg in range(2):
                if slot == finger_slot(side, fi, seg):
                    return finger_slot(side, fi, seg + 1)
    return None


def twist_reference(slot: str) -> tuple[str, str] | None:
    """Pair of slots whose joint-to-joint vector fixes the bone's twist (hands only)."""
    for side in SIDES:
        if slot == f"{side}Hand":
            return (f"{side}LittleProximal", f"{side}IndexProximal")
    return None


_SIDE_PATTERNS = [
    (re.compile(r"(^|[_\s.\-])(l|left)([_\s.\-]|$)"), "left"),
    (re.compile(r"(^|[_\s.\-])(r|right)([_\s.\-]|$)"), "right"),
]
_HELPER = re.compile(r"helper|twist|ulna|wrist|bicep|trap|pect|elbow|hlp|_jnt_end|_end$|nub|jiggle|_jig|dummy|attach|weapon|prop")


def _normalize(name: str) -> str:
    n = name.lower()
    for prefix in ("valvebiped.", "bip01_", "bip01 ", "bip_", "bip01.", "bip."):
        if n.startswith(prefix):
            n = n[len(prefix) :]
    n = n.replace("valvebiped.", "").replace("bip01_", "")
    return n


def _side_and_core(name: str) -> tuple[str | None, str]:
    n = _normalize(name)
    for pat, side in _SIDE_PATTERNS:
        m = pat.search(n)
        if m:
            core = (n[: m.start()] + "_" + n[m.end() :]).strip("_ .-")
            return side, core
    if n.startswith("left"):
        return "left", n[4:].strip("_ .-")
    if n.startswith("right"):
        return "right", n[5:].strip("_ .-")
    return None, n


def _depths(parents: list[int]) -> list[int]:
    d = []
    for i, p in enumerate(parents):
        d.append(d[p] + 1 if 0 <= p < i else 0)
    return d


def _is_descendant(parents: list[int], child: int, ancestor: int) -> bool:
    i = parents[child]
    while i >= 0:
        if i == ancestor:
            return True
        i = parents[i]
    return False


def classify_source_bones(names: list[str], parents: list[int]) -> tuple[dict[str, int], list[int]]:
    """Return (slot -> bone index, ordered spine bone indices) for a Source skeleton."""
    depth = _depths(parents)
    cands: dict[str, list[tuple[int, int]]] = {}  # slot -> [(score, idx)]
    spines: list[int] = []

    def add(slot: str, idx: int, score: int):
        cands.setdefault(slot, []).append((score, idx))

    for i, name in enumerate(names):
        side, core = _side_and_core(name)
        low = name.lower()
        score = depth[i] * 2 + len(name) // 8
        if "bip" in low:
            score -= 100
        if _HELPER.search(core):
            continue
        if side is None:
            if re.fullmatch(r"(pelvis|hips?|hip_?root)", core):
                add("hips", i, score)
            elif re.fullmatch(r"spine_?\d*|chest|upper_?chest|torso_?\d*", core):
                spines.append(i)
            elif re.fullmatch(r"neck_?\d*", core):
                add("neck", i, score)
            elif re.fullmatch(r"head_?\d*", core):
                add("head", i, score)
            continue
        if re.fullmatch(r"clavicle|collar(bone)?", core):
            add(f"{side}Shoulder", i, score)
        elif re.fullmatch(r"shoulder", core):
            add(f"{side}Shoulder", i, score + 50)  # weaker: often a helper in HL2 rigs
        elif re.fullmatch(r"upper_?arm|up_?arm|arm", core):
            add(f"{side}UpperArm", i, score)
        elif re.fullmatch(r"fore_?arm|lower_?arm|low_?arm", core):
            add(f"{side}LowerArm", i, score)
        elif re.fullmatch(r"hand", core):
            add(f"{side}Hand", i, score)
        elif re.fullmatch(r"thigh|upper_?leg|up_?leg", core):
            add(f"{side}UpperLeg", i, score)
        elif re.fullmatch(r"hip", core):  # TF2-style rigs: bip_hip_L
            add(f"{side}UpperLeg", i, score + 30)
        elif re.fullmatch(r"calf|lower_?leg|low_?leg|shin|leg", core):
            add(f"{side}LowerLeg", i, score)
        elif re.fullmatch(r"knee", core):  # TF2-style rigs: bip_knee_L
            add(f"{side}LowerLeg", i, score + 30)
        elif re.fullmatch(r"foot|ankle", core):
            add(f"{side}Foot", i, score)
        elif re.fullmatch(r"toe_?\d*|toe_?base|toes|ball", core):
            add(f"{side}Toes", i, score)
        else:
            m = re.fullmatch(r"finger_?(\d)(\d?)", core)
            if m:
                fi, seg = int(m.group(1)), int(m.group(2) or 0)
                if fi <= 4 and seg <= 2:
                    add(finger_slot(side, fi, seg), i, score)
                continue
            m = re.fullmatch(r"(thumb|index|middle|mid|ring|pinky|little)_?(\d)", core)
            if m:
                fname = {"thumb": 0, "index": 1, "middle": 2, "mid": 2, "ring": 3, "pinky": 4, "little": 4}[m.group(1)]
                cands.setdefault(f"__{side}{fname}", []).append((int(m.group(2)), i))

    out: dict[str, int] = {}
    for slot, lst in cands.items():
        if slot.startswith("__"):
            continue
        lst.sort()
        out[slot] = lst[0][1]
    # numbered finger names (thumb_01, thumb_02...) -> ordered segments
    for key, lst in cands.items():
        if not key.startswith("__"):
            continue
        side = "left" if key.startswith("__left") else "right"
        fi = int(key[-1])
        lst.sort()
        for seg, (_, idx) in enumerate(lst[:3]):
            out.setdefault(finger_slot(side, fi, seg), idx)

    spines.sort(key=lambda i: depth[i])
    hips = out.get("hips")
    if hips is not None:
        spines = [s for s in spines if _is_descendant(parents, s, hips)]

    # Hierarchy validation: drop mappings that are not descendants of their parent slot.
    chain_parent = {}
    for side in SIDES:
        chain_parent[f"{side}UpperArm"] = f"{side}Shoulder"
        chain_parent[f"{side}LowerArm"] = f"{side}UpperArm"
        chain_parent[f"{side}Hand"] = f"{side}LowerArm"
        chain_parent[f"{side}LowerLeg"] = f"{side}UpperLeg"
        chain_parent[f"{side}Foot"] = f"{side}LowerLeg"
        chain_parent[f"{side}Toes"] = f"{side}Foot"
        for fi in range(5):
            chain_parent[finger_slot(side, fi, 0)] = f"{side}Hand"
            chain_parent[finger_slot(side, fi, 1)] = finger_slot(side, fi, 0)
            chain_parent[finger_slot(side, fi, 2)] = finger_slot(side, fi, 1)
    for slot in list(out):
        par = chain_parent.get(slot)
        if par and par in out and not _is_descendant(parents, out[slot], out[par]):
            del out[slot]
    return out, spines


def map_vrm_to_source(vrm_humanoid: dict[str, int], src_slots: dict[str, int], src_spines: list[int]) -> dict[str, int]:
    """VRM humanoid slot -> Source bone index."""
    mapping: dict[str, int] = {}
    for slot, idx in src_slots.items():
        if slot in vrm_humanoid:
            mapping[slot] = idx
    vchain = [s for s in ("spine", "chest", "upperChest") if s in vrm_humanoid]
    m = len(src_spines)
    n = len(vchain)
    if m and n:
        if n == 1 or m == 1:
            mapping[vchain[0]] = src_spines[0]
            if n > 1 and m == 1:
                pass
        else:
            used = set()
            for i, slot in enumerate(vchain):
                if m >= n:
                    j = round(i * (m - 1) / (n - 1))
                else:
                    j = i if i < m - 1 else m - 1
                    if i == n - 1:
                        j = m - 1
                    elif i >= m - 1:
                        continue
                if j in used:
                    continue
                used.add(j)
                mapping[slot] = src_spines[j]
    return mapping
