import os
import re

import pytest

from portal2vroid import compiler, installer
from portal2vroid import keyvalues as kv
from portal2vroid.gamefs import GameFS, find_mod_dir
from portal2vroid.installer import ModManager
from portal2vroid.vpk import VPK, write_vpk

from . import fixtures


@pytest.fixture
def game(tmp_path):
    root = tmp_path / "Portal 2"
    (root / "portal2").mkdir(parents=True)
    (root / "portal2" / "gameinfo.txt").write_text('"GameInfo" { game "Portal 2" }')
    (root / "portal2_dlc1").mkdir()
    (root / "portal2_dlc2").mkdir()
    (root / "bin").mkdir()
    write_vpk(
        str(root / "portal2" / "pak01_dir.vpk"),
        {
            "models/player/chell/player.mdl": fixtures.make_chell_mdl(),
            "models/player/chell/player.vvd": fixtures.make_chell_vvd(70.0),
            "resource/ui/basemodui/ingamemainmenu.res": fixtures.INGAME_MENU_RES.encode(),
            "resource/ui/basemodui/optionsflyout.res": fixtures.OPTIONS_FLYOUT_RES.encode(),
            "scripts/vscripts/mapspawn.nut": b"printl(\"mapspawn\")\n",
        },
    )
    (root / "portal2" / "scripts").mkdir()
    (root / "portal2" / "scripts" / "kb_act.lst").write_text(fixtures.KB_ACT)
    return root


def fake_compile(game_root, qc_path, model_id, mod_dir, log=print, studiomdl=None):
    text = open(qc_path).read()
    assert f'$modelname "vroid/{model_id}.mdl"' in text
    out = os.path.join(mod_dir, "models", "vroid")
    os.makedirs(out, exist_ok=True)
    paths = []
    for ext in (".mdl", ".vvd", ".dx90.vtx"):
        p = os.path.join(out, model_id + ext)
        open(p, "wb").write(b"IDST")
        paths.append(p)
    return paths


def test_install_add_uninstall(game, tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "compile_model", fake_compile)
    m = ModManager(str(game), log=lambda *_: None)
    mod = m.install()
    assert os.path.basename(mod) == "portal2_dlc3"
    assert find_mod_dir(str(game)) == mod

    mapspawn = open(os.path.join(mod, "scripts/vscripts/mapspawn.nut")).read()
    assert mapspawn.startswith('printl("mapspawn")')
    assert 'IncludeScript("vroid/vroid_main")' in mapspawn
    assert os.path.isfile(os.path.join(mod, "scripts/vscripts/vroid/vroid_main.nut"))
    assert os.path.isfile(os.path.join(mod, "cfg/vroid/vroid_aliases.cfg"))
    menu = kv.parse(open(os.path.join(mod, "resource/ui/basemodui/ingamemainmenu.res")).read())
    assert menu.blocks()[0].find("BtnVRoid") is not None
    fly = kv.parse(open(os.path.join(mod, "resource/ui/basemodui/vroid_flyout.res")).read())
    assert fly.blocks()[0].find("BtnVRoidTP").get("ypos") == "5"  # took the Options flyout's layout
    assert '"vroid_tp_toggle"' in open(os.path.join(mod, "scripts/kb_act.lst")).read()
    binds = open(os.path.join(mod, "cfg/vroid/vroid_binds.cfg")).read()
    assert 'bind "v" "vroid_tp_toggle"' in binds

    # GameFS must ignore our own folder when reading originals
    fs = GameFS(str(game), exclude_dir=mod)
    assert fs.read("scripts/vscripts/mapspawn.nut") == b'printl("mapspawn")\n'
    # re-install is idempotent (still one entry, patched from the original)
    m.install()
    menu_text = open(os.path.join(mod, "resource/ui/basemodui/ingamemainmenu.res")).read()
    assert [c.key for c in kv.parse(menu_text).blocks()[0].blocks()].count("BtnVRoid") == 1

    vrm_path = tmp_path / "Seed San.vrm"
    vrm_path.write_bytes(fixtures.make_vrm("1.0"))
    av = m.add_avatar(str(vrm_path))
    assert av.id == 1 and av.name == "Test Avatar" and av.model_id == "a1_test_avatar"
    assert os.path.isfile(os.path.join(mod, "materials/models/vroid/a1_test_avatar/body_skin.vmt"))
    nut = open(os.path.join(mod, "scripts/vscripts/vroid/vroid_avatars.nut")).read()
    assert 'model = "models/vroid/a1_test_avatar.mdl", short = "vroid/a1_test_avatar"' in nut
    assert re.search(r"::VRoidDefaults <- \{ avatar = 1,", nut)  # first avatar becomes the default
    assert not os.path.exists(os.path.join(mod, "_build"))

    v = VPK(os.path.join(mod, "pak01_dir.vpk"))
    assert "scripts/vscripts/vroid/vroid_avatars.nut" in v
    assert v.read("scripts/vscripts/vroid/vroid_avatars.nut").decode() == nut

    av2 = m.add_avatar(str(vrm_path), name="Second")
    assert av2.id == 2
    m.remove_avatar(1)
    assert not os.path.exists(os.path.join(mod, "models/vroid/a1_test_avatar.mdl"))
    nut = open(os.path.join(mod, "scripts/vscripts/vroid/vroid_avatars.nut")).read()
    assert "a1_test_avatar" not in nut and "avatar = 2," in nut

    m.uninstall()
    assert not os.path.exists(mod)
    assert find_mod_dir(str(game)) is None


def test_uninstall_keeps_numbering(game, monkeypatch):
    m = ModManager(str(game), log=lambda *_: None)
    mod = m.install()
    later = game / "portal2_dlc4"
    later.mkdir()
    m.uninstall()
    assert os.path.isdir(mod) and os.listdir(mod) == ["placeholder.txt"]


def test_compile_requires_studiomdl(game, tmp_path):
    with pytest.raises(compiler.CompileError):
        compiler.compile_model(str(game), str(tmp_path / "x.qc"), "x", str(tmp_path))
