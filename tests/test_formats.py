import math
import os
import struct

import numpy as np
import pytest

from portal2vroid import keyvalues as kv
from portal2vroid.mathutil import euler_to_mat3, mat3_to_euler, mat3_to_quat, quat_to_mat3, rotation_between
from portal2vroid.mdl import parse_mdl, vvd_height
from portal2vroid.menu_patch import FLYOUT_ITEMS, build_flyout, patch_ingame_menu, patch_kb_act
from portal2vroid.vpk import VPK, write_vpk
from portal2vroid.vrm import load_vrm
from portal2vroid.vtf import FMT_BGRA8888, FMT_DXT1, FMT_DXT5, build_vtf, toon_ramp_vtf

from . import fixtures


def test_euler_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(200):
        e = rng.uniform(-3, 3, 3)
        e[1] = rng.uniform(-1.5, 1.5)
        m = euler_to_mat3(*e)
        assert np.allclose(euler_to_mat3(*mat3_to_euler(m)), m, atol=1e-9)


def test_quat_roundtrip_and_source_convention():
    rng = np.random.default_rng(2)
    for _ in range(100):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        m = quat_to_mat3(q)
        q2 = mat3_to_quat(m)
        assert np.allclose(quat_to_mat3(q2), m, atol=1e-9)
    # Source AngleQuaternion(RadianEuler) for z=90deg is (0,0,sin45,cos45)
    m = euler_to_mat3(0, 0, math.pi / 2)
    assert np.allclose(mat3_to_quat(m), [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)])


def test_rotation_between():
    for a, b in (([1, 0, 0], [0, 1, 0]), ([0, 0, 1], [0, 0, -1]), ([1, 2, 3], [-3, 1, 0.5])):
        r = rotation_between(a, b)
        assert np.allclose(r @ (np.array(a) / np.linalg.norm(a)), np.array(b) / np.linalg.norm(b), atol=1e-9)


def test_vpk_roundtrip(tmp_path):
    files = {"scripts/vscripts/mapspawn.nut": b"printl(1)", "models/player/chell/player.mdl": b"\x01" * 1000, "readme": b"x"}
    p = tmp_path / "pak01_dir.vpk"
    write_vpk(str(p), files)
    v = VPK(str(p))
    for k, d in files.items():
        assert v.read(k) == d


def test_mdl_parse():
    skel = parse_mdl(fixtures.make_chell_mdl())
    names, parents, worlds = fixtures.chell_world_bones()
    assert [b.name for b in skel.bones] == names
    assert [b.parent for b in skel.bones] == parents
    for a, b in zip(skel.world_matrices(), worlds):
        assert np.allclose(a, b, atol=1e-4)
    assert [c.name for c in skel.ikchains] == ["lfoot", "rfoot", "lhand", "rhand"]
    assert skel.include_models == ["models/player/chell/anims.mdl"]
    assert vvd_height(fixtures.make_chell_vvd(70.0)) == pytest.approx(70.0)


def _decode_dxt(data: bytes, w: int, h: int, fmt: int) -> np.ndarray:
    out = np.zeros((h, w, 4), dtype=np.float32)
    bs = 8 if fmt == FMT_DXT1 else 16
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    for by in range(bh):
        for bx in range(bw):
            blk = data[(by * bw + bx) * bs : (by * bw + bx + 1) * bs]
            alpha = np.full(16, 255.0)
            if fmt == FMT_DXT5:
                a0, a1 = blk[0], blk[1]
                bits = int.from_bytes(blk[2:8], "little")
                lv = [a0, a1] + [((8 - k) * a0 + (k - 1) * a1) / 7 for k in range(2, 8)]
                alpha = np.array([lv[(bits >> (3 * i)) & 7] for i in range(16)], dtype=float)
                blk = blk[8:]
            c0, c1, bits = struct.unpack("<HHI", blk)

            def rgb(c):
                return np.array([((c >> 11) & 31) * 255 / 31, ((c >> 5) & 63) * 255 / 63, (c & 31) * 255 / 31])

            p0, p1 = rgb(c0), rgb(c1)
            pal = [p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3]
            for i in range(16):
                y, x = by * 4 + i // 4, bx * 4 + i % 4
                if y < h and x < w:
                    out[y, x, :3] = pal[(bits >> (2 * i)) & 3]
                    out[y, x, 3] = alpha[i]
    return out


@pytest.mark.parametrize("fmt", [FMT_DXT1, FMT_DXT5])
def test_vtf_dxt_quality(fmt):
    from PIL import Image

    yy, xx = np.mgrid[0:64, 0:64]
    arr = np.stack([xx * 4, yy * 4, (xx + yy) * 2, 255 - yy * 3], axis=-1).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGBA")
    data = build_vtf(img, fmt)
    assert data[:4] == b"VTF\0"
    w, h = struct.unpack_from("<HH", data, 16)
    assert (w, h) == (64, 64)
    mips = data[56]
    assert mips == 7
    # largest mip is stored last
    size = 64 * 64 // (2 if fmt == FMT_DXT1 else 1)
    dec = _decode_dxt(data[-size:], 64, 64, fmt)
    err = np.abs(dec[..., :3] - arr[..., :3]).mean()
    assert err < 6.0
    if fmt == FMT_DXT5:
        assert np.abs(dec[..., 3] - arr[..., 3]).mean() < 3.0


def test_vtf_sizes_and_ramp():
    from PIL import Image

    img = Image.new("RGBA", (8, 4), (10, 20, 30, 255))
    data = build_vtf(img, FMT_BGRA8888)
    # header 80 + lowres dxt1 (8x4 -> 2 blocks*8) + mips 8x4,4x2,2x1,1x1 *4 bytes
    assert len(data) == 80 + 16 + (32 + 8 + 2 + 1) * 4
    ramp = toon_ramp_vtf()
    assert ramp[:4] == b"VTF\0"


def test_keyvalues_roundtrip_and_conditions():
    root = kv.parse(fixtures.INGAME_MENU_RES)
    top = root.blocks()[0]
    exit_btn = top.find("BtnExitToMainMenu")
    ypos = exit_btn.find("ypos")
    assert ypos.value == "240" and ypos.cond == "[$WIN32]"
    again = kv.parse(kv.dump(root))
    assert kv.dump(again) == kv.dump(root)


def test_menu_patch():
    text, tpl = patch_ingame_menu(fixtures.INGAME_MENU_RES)
    assert tpl == "resource/UI/basemodui/OptionsFlyout.res"
    top = kv.parse(text).blocks()[0]
    btn = top.find("BtnVRoid")
    assert btn.get("ypos") == "240"
    assert btn.get("command") == "FlmVRoidFlyout"
    assert btn.get("navUp") == "BtnOptions" and btn.get("navDown") == "BtnExitToMainMenu"
    assert top.find("BtnExitToMainMenu").get("ypos") == "260"
    assert top.find("BtnOptions").get("navDown") == "BtnVRoid"
    assert top.find("BtnExitToMainMenu").get("navUp") == "BtnVRoid"
    fly = top.find("FlmVRoidFlyout")
    assert fly.get("ControlName") == "FlyoutMenu"
    assert fly.get("ResourceFile") == "resource/ui/basemodui/vroid_flyout.res"
    # button order: right after options
    keys = [c.key for c in top.blocks()]
    assert keys.index("BtnVRoid") == keys.index("BtnOptions") + 1
    # re-patching an already patched file is refused
    assert patch_ingame_menu(text) is None


def test_flyout_build():
    for tpl in (fixtures.OPTIONS_FLYOUT_RES, None):
        top = kv.parse(build_flyout(tpl)).blocks()[0]
        btns = [c for c in top.blocks() if c.key.startswith("BtnVRoid")]
        assert len(btns) == len(FLYOUT_ITEMS)
        assert btns[0].get("command") == "engine vroid_tp_toggle"
        assert btns[1].get("ypos") == ("25" if tpl else "20")
        bg = top.find("PnlBackground")
        assert int(bg.get("tall")) >= int(btns[-1].get("ypos")) + 20


def test_kb_act_patch():
    out = patch_kb_act(fixtures.KB_ACT)
    assert out.startswith(fixtures.KB_ACT.rstrip("\n"))
    assert '"vroid_tp_toggle"' in out
    assert patch_kb_act(out) == out


@pytest.mark.parametrize("version", ["1.0", "0.x"])
def test_vrm_load(version):
    vrm = load_vrm(fixtures.make_vrm(version))
    assert vrm.version == version
    assert "hips" in vrm.humanoid and "leftThumbMetacarpal" in vrm.humanoid
    assert len(vrm.spring_joints) == 3
    assert len(vrm.primitives) == 2
    assert vrm.materials[1].alpha_mode == "MASK"
    # skinning to rest pose with identity bind leaves the head box near 1.3m
    allp = np.concatenate([p.positions for p in vrm.primitives])
    assert 1.3 < allp[:, 1].max() < 1.5
    assert abs(allp[:, 1].min()) < 0.05
