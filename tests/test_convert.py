import os
import re

import numpy as np
import pytest

from portal2vroid.mathutil import euler_to_mat3, make_mat4, normalize, rot_z
from portal2vroid.mdl import parse_mdl
from portal2vroid.retarget import ConvertOptions, convert
from portal2vroid.source_export import write_model_sources
from portal2vroid.vrm import load_vrm
from portal2vroid.materials import write_materials

from . import fixtures


def _setup(version="1.0", scale=1.0, **kw):
    chell = parse_mdl(fixtures.make_chell_mdl())
    vrm = load_vrm(fixtures.make_vrm(version, scale))
    res = convert(vrm, chell, ConvertOptions(chell_height=70.0, **kw))
    return chell, vrm, res


def _bone(res, name):
    for i, b in enumerate(res.bones):
        if b.name == name:
            return i, b
    raise KeyError(name)


@pytest.mark.parametrize("version", ["1.0", "0.x"])
@pytest.mark.parametrize("vscale", [1.0, 0.6])
def test_height_facing_and_mapping(version, vscale):
    chell, vrm, res = _setup(version, vscale)
    vb = "ValveBiped.Bip01_"
    assert res.mapping["hips"] == vb + "Pelvis"
    assert res.mapping["leftUpperArm"] == vb + "L_UpperArm"
    assert res.mapping["leftThumbMetacarpal"] == vb + "L_Finger0"
    assert res.mapping["leftIndexDistal"] == vb + "L_Finger12"
    assert res.mapping["spine"] == vb + "Spine"
    assert res.mapping["upperChest"] == vb + "Spine4"
    # height matches Chell regardless of the avatar's own size
    allp = np.concatenate([m.positions for m in res.meshes])
    assert allp[:, 2].max() == pytest.approx(70.0, abs=0.05)
    assert allp[:, 2].min() == pytest.approx(0.0, abs=0.05)
    # facing: avatar's left hand is on +Y (Chell's left), toes point +X
    _, lh = _bone(res, vb + "L_Hand")
    _, rh = _bone(res, vb + "R_Hand")
    assert lh.world[1, 3] > 5 and rh.world[1, 3] < -5
    _, toe = _bone(res, vb + "L_Toe0")
    _, foot = _bone(res, vb + "L_Foot")
    assert toe.world[0, 3] > foot.world[0, 3] + 1


def test_rest_orientation_matches_chell_and_arms_are_reposed():
    chell, vrm, res = _setup()
    cw = chell.world_matrices()
    for i in range(res.n_chell):
        assert np.allclose(res.bones[i].world[:3, :3], cw[i][:3, :3], atol=1e-9)
    vb = "ValveBiped.Bip01_"
    for side in ("L", "R"):
        iu, up = _bone(res, vb + f"{side}_UpperArm")
        il, lo = _bone(res, vb + f"{side}_Forearm")
        d_out = normalize(lo.world[:3, 3] - up.world[:3, 3])
        d_ch = normalize(cw[il][:3, 3] - cw[iu][:3, 3])
        assert np.dot(d_out, d_ch) > 0.999  # T-pose avatar now in Chell's A-pose
    # proportions are the avatar's: upper-arm length differs from Chell's
    iu, up = _bone(res, vb + "L_UpperArm")
    il, lo = _bone(res, vb + "L_Forearm")
    out_len = np.linalg.norm(lo.world[:3, 3] - up.world[:3, 3])
    ch_len = np.linalg.norm(cw[il][:3, 3] - cw[iu][:3, 3])
    expected = 0.24 * res.scale
    assert out_len == pytest.approx(expected, rel=1e-3)
    assert abs(out_len - ch_len) > 0.5


def test_mesh_follows_repose():
    chell, vrm, res = _setup()
    # vertices of the forearm box (weighted fully to the forearm) should lie along Chell's arm direction
    vb = "ValveBiped.Bip01_"
    il, lo = _bone(res, vb + "L_Forearm")
    ih, hand = _bone(res, vb + "L_Hand")
    mesh = res.meshes[0]
    sel = (mesh.bones[:, 0] == il) & (mesh.weights[:, 0] > 0.99)
    assert sel.any()
    pts = mesh.positions[sel]
    axis = normalize(hand.world[:3, 3] - lo.world[:3, 3])
    rel = pts - lo.world[:3, 3]
    along = rel @ axis
    perp = np.linalg.norm(rel - np.outer(along, axis), axis=1)
    assert along.max() > 3 and perp.max() < 3.0


def test_weights_and_extra_bones():
    chell, vrm, res = _setup()
    n = len(res.bones)
    assert n <= 128
    for m in res.meshes:
        assert m.bones.shape[1] == 3
        assert np.all((m.bones >= 0) & (m.bones < n))
        assert np.allclose(m.weights.sum(axis=1), 1.0)
    extra = [b.name for b in res.bones[res.n_chell :]]
    assert "vrm_J_Hair1" in extra and "vrm_J_L_Skirt" in extra
    assert "vrm_Root" not in extra  # unweighted armature root is dropped
    _, hair1 = _bone(res, "vrm_J_Hair1")
    assert res.bones[hair1.parent].name == "ValveBiped.Bip01_Head1"
    assert hair1.jiggle_length > 0
    # jigglebone axis (+Z) points at the next hair joint
    _, hair2 = _bone(res, "vrm_J_Hair2")
    d = normalize(hair2.world[:3, 3] - hair1.world[:3, 3])
    assert np.dot(hair1.world[:3, 2], d) > 0.999


def test_bone_budget_collapses_leaves():
    chell, vrm, res = _setup(max_bones=len(parse_mdl(fixtures.make_chell_mdl()).bones) + 2)
    extra = res.bones[res.n_chell :]
    assert len(extra) == 2
    assert res.warnings


def _read_smd_skeleton(path):
    text = open(path).read()
    nodes = re.search(r"nodes\n(.*?)\nend", text, re.S).group(1).splitlines()
    names, parents = [], []
    for ln in nodes:
        m = re.match(r'(\d+) "(.*)" (-?\d+)', ln)
        names.append(m.group(2))
        parents.append(int(m.group(3)))
    sk = re.search(r"skeleton\ntime 0\n(.*?)\nend", text, re.S).group(1).splitlines()
    worlds = []
    for i, ln in enumerate(sk):
        v = [float(x) for x in ln.split()[1:]]
        local = make_mat4(euler_to_mat3(*v[3:]), v[:3])
        if parents[i] >= 0:
            worlds.append(worlds[parents[i]] @ local)
        else:
            # undo studiomdl's default +90 deg Z rotation pre-compensation
            worlds.append(make_mat4(rot_z(np.pi / 2)) @ local)
    return names, parents, worlds, text


def test_smd_qc_output(tmp_path):
    chell, vrm, res = _setup()
    vmts = write_materials(vrm, str(tmp_path), "test", log=lambda *_: None)
    assert vmts == ["body_skin", "hair_cutout"]
    hair_vmt = open(tmp_path / "materials/models/vroid/test/hair_cutout.vmt").read()
    assert '"$alphatest" "1"' in hair_vmt and '"$nocull" "1"' in hair_vmt
    assert os.path.isfile(tmp_path / "materials/models/vroid/shared/toon_ramp.vtf")
    qc = write_model_sources(str(tmp_path / "src"), "test", res, chell, vmts)
    qtext = open(qc).read()
    assert '$includemodel "player/chell/player.mdl"' in qtext
    assert '$sequence "vroid_proportions" "a_vroid_proportions" predelta autoplay' in qtext
    assert '$ikchain "lfoot" "ValveBiped.Bip01_L_Foot" knee 0.700000 0.700000 0.000000' in qtext
    assert '$jigglebone "vrm_J_Hair1"' in qtext

    bodies = re.findall(r'\$body "[^"]+" "([^"]+)"', qtext)
    assert bodies
    names, parents, worlds, text = _read_smd_skeleton(tmp_path / "src" / bodies[0])
    assert names == [b.name for b in res.bones]
    for w, b in zip(worlds, res.bones):
        assert np.allclose(w, b.world, atol=1e-4)
    # reference skeleton reproduces Chell's own rest pose
    _, _, ref_worlds, _ = _read_smd_skeleton(tmp_path / "src" / "chell_reference.smd")
    cw = chell.world_matrices()
    for i in range(res.n_chell):
        assert np.allclose(ref_worlds[i], cw[i], atol=1e-4)
    # triangles reference only valid materials and bones
    tri = text.split("triangles\n", 1)[1]
    mats = {ln for ln in tri.splitlines()[::4] if ln != "end"}
    assert mats <= set(vmts)
