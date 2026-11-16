import os
import re

import pytest

from portal2vroid import compiler, installer
from portal2vroid import keyvalues as kv
from portal2vroid.gamefs import GameFS, find_mod_dir
from portal2vroid.installer import ModManager
from portal2vroid.vpk import VPK, write_vpk

from . import fixtures


GAMEINFO = """"GameInfo"
{
	game	"Portal 2"
	FileSystem
	{
		SteamAppId	620
		SearchPaths
		{
			Game	|gameinfo_path|.
			Game	portal2
		}
	}
}
"""


@pytest.fixture
def game(tmp_path):
    root = tmp_path / "Portal 2"
    (root / "portal2").mkdir(parents=True)
    (root / "portal2" / "gameinfo.txt").write_text(GAMEINFO)
    (root / "portal2_dlc1").mkdir()
    (root / "portal2_dlc2").mkdir()
    (root / "bin").mkdir()
    (root / "bin" / "studiomdl.exe").write_bytes(b"MZ")
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
    assert os.path.basename(mod) == "portal2_vroid"
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

    gi = (game / "portal2" / "gameinfo.txt").read_text()
    assert "\t\t\tGame\t\t\t\t|gameinfo_path|../portal2_vroid // Portal2VRoid\n" in gi
    assert gi.index("portal2_vroid") < gi.index("|gameinfo_path|.\n")  # first search path
    assert (game / "portal2" / "gameinfo.txt.vroid_backup").read_text() == GAMEINFO
    assert "exec vroid/vroid_boot" in (game / "portal2" / "cfg" / "autoexec.cfg").read_text()
    assert not [ln for ln in m.diagnose() if ln.startswith("PROBLEM") and "avatar" not in ln and "studiomdl" not in ln]

    # GameFS must ignore our own folder when reading originals
    fs = GameFS(str(game), exclude_dir=mod)
    assert fs.read("scripts/vscripts/mapspawn.nut") == b'printl("mapspawn")\n'
    # re-install is idempotent (still one entry, patched from the original)
    m.install()
    assert (game / "portal2" / "gameinfo.txt").read_text().count("Portal2VRoid") == 1
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
    assert os.path.isfile(os.path.join(mod, "_build", "a1_test_avatar", "a1_test_avatar.qc"))  # kept for debugging
    import zipfile

    zp = m.export_debug(str(tmp_path / "debug.zip"))
    names = zipfile.ZipFile(zp).namelist()
    assert "game/models/player/chell/player.mdl" in names and "game/models/player/chell/anims.mdl" not in names
    assert "build/a1_test_avatar/a1_test_avatar.qc" in names and "vroid_settings.json" in names

    v = VPK(os.path.join(mod, "pak01_dir.vpk"))
    assert "scripts/vscripts/vroid/vroid_avatars.nut" in v
    assert "scripts/vscripts/mapspawn.nut" in v and "resource/ui/basemodui/ingamemainmenu.res" in v
    assert "models/vroid/a1_test_avatar.mdl" not in v  # big files stay loose
    assert os.path.isfile(os.path.join(mod, "models/vroid/a1_test_avatar.mdl"))
    assert "vroid_settings.json" not in v
    assert v.read("scripts/vscripts/vroid/vroid_avatars.nut").decode() == nut

    av2 = m.add_avatar(str(vrm_path), name="Second")
    assert av2.id == 2
    m.remove_avatar(1)
    assert not os.path.exists(os.path.join(mod, "models/vroid/a1_test_avatar.mdl"))
    nut = open(os.path.join(mod, "scripts/vscripts/vroid/vroid_avatars.nut")).read()
    assert "a1_test_avatar" not in nut and "avatar = 2," in nut

    (game / "portal2" / "cfg" / "autoexec.cfg").write_text("bind x jump\n" + (game / "portal2" / "cfg" / "autoexec.cfg").read_text())
    m.uninstall()
    assert not os.path.exists(mod)
    assert (game / "portal2" / "gameinfo.txt").read_text() == GAMEINFO
    assert (game / "portal2" / "cfg" / "autoexec.cfg").read_text() == "bind x jump\n"
    assert find_mod_dir(str(game)) is None


def test_migrates_old_dlc_install(game, monkeypatch):
    """Older versions installed into portal2_dlc3 (mounted twice: as DLC and via gameinfo)."""
    old = game / "portal2_dlc3"
    (old / "models" / "vroid").mkdir(parents=True)
    (old / "vroid_mod.txt").write_text("old")
    (old / "models" / "vroid" / "a1_miku.mdl").write_bytes(b"IDST")
    (old / "vroid_avatars.json").write_text('[{"id": 1, "name": "Miku", "model_id": "a1_miku", "source": "x"}]')
    (game / "portal2_dlc4").mkdir()  # someone else's DLC folder after ours
    gi = game / "portal2" / "gameinfo.txt"
    gi.write_text(gi.read_text().replace("SearchPaths\n\t\t{\n", "SearchPaths\n\t\t{\n\t\t\tGame\t|gameinfo_path|../portal2_dlc3 // Portal2VRoid\n"))

    m = ModManager(str(game), log=lambda *_: None)
    assert os.path.basename(m.mod_dir) == "portal2_dlc3"
    mod = m.install()
    assert os.path.basename(mod) == "portal2_vroid"
    assert os.path.isfile(game / "portal2_vroid" / "models" / "vroid" / "a1_miku.mdl")
    assert [a.name for a in m.avatars()] == ["Miku"]
    assert os.listdir(old) == ["placeholder.txt"]  # keeps portal2_dlc4 mounting
    text = gi.read_text()
    assert "portal2_dlc3" not in text and text.count("Portal2VRoid") == 1 and "../portal2_vroid" in text
    assert find_mod_dir(str(game)) == mod
    m.uninstall()
    assert not (game / "portal2_vroid").exists() and gi.read_text() == GAMEINFO


def test_compile_requires_studiomdl(game, tmp_path):
    (game / "bin" / "studiomdl.exe").unlink()
    with pytest.raises(compiler.CompileError):
        compiler.compile_model(str(game), str(tmp_path / "x.qc"), "x", str(tmp_path))


def test_portal2_chell_with_animation_include(tmp_path, monkeypatch):
    """player.mdl without IK chains whose animations (and IK) live in player_animations.mdl."""
    monkeypatch.setattr(installer, "compile_model", fake_compile)
    root = tmp_path / "Portal 2"
    (root / "portal2").mkdir(parents=True)
    (root / "portal2" / "gameinfo.txt").write_text(GAMEINFO)
    (root / "bin").mkdir()
    (root / "bin" / "studiomdl.exe").write_bytes(b"MZ")
    anims = "models/player/chell/player_animations.mdl"
    write_vpk(str(root / "portal2" / "pak01_dir.vpk"), {
        "models/player/chell/player.mdl": fixtures.make_chell_mdl("portal2", with_ik=False, includes=(anims,)),
        "models/player/chell/player.vvd": fixtures.make_chell_vvd(13.8),
        anims: fixtures.make_chell_mdl("portal2", includes=()),
    })
    logs = []
    m = ModManager(str(root), log=logs.append)
    vrm_path = tmp_path / "miku.vrm"
    vrm_path.write_bytes(fixtures.make_vrm("1.0"))
    av = m.add_avatar(str(vrm_path))
    assert any("IK chains from models/player/chell/player_animations.mdl: lfoot, rfoot, lhand, rhand" in ln for ln in logs)
    assert av.id == 1


def test_qc_includes_chell_animation_models(tmp_path, monkeypatch):
    captured = {}

    def grab(game_root, qc_path, model_id, mod_dir, log=print, studiomdl=None):
        captured["qc"] = open(qc_path).read()
        return fake_compile(game_root, qc_path, model_id, mod_dir, log, studiomdl)

    monkeypatch.setattr(installer, "compile_model", grab)
    root = tmp_path / "Portal 2"
    (root / "portal2").mkdir(parents=True)
    (root / "portal2" / "gameinfo.txt").write_text(GAMEINFO)
    (root / "bin").mkdir()
    (root / "bin" / "studiomdl.exe").write_bytes(b"MZ")
    anims = "models/player/chell/player_animations.mdl"
    write_vpk(str(root / "portal2" / "pak01_dir.vpk"), {
        "models/player/chell/player.mdl": fixtures.make_chell_mdl("portal2", with_ik=False, includes=(anims,)),
        anims: fixtures.make_chell_mdl("portal2", includes=()),
    })
    vrm_path = tmp_path / "miku.vrm"
    vrm_path.write_bytes(fixtures.make_vrm("0.x"))
    ModManager(str(root), log=lambda *_: None).add_avatar(str(vrm_path))
    qc = captured["qc"]
    assert '$includemodel "player/chell/player.mdl"' in qc
    assert '$includemodel "player/chell/player_animations.mdl"' in qc
    assert '$ikchain "lfoot" "ankle_L"' in qc


FAKE_STUDIOMDL = r"""#!/bin/sh
# Mimics Portal 2's studiomdl: writes <game>/models/<modelname> but never creates folders.
game="$2"
qc="$6"
name=$(sed -n 's/^\$modelname "\(.*\)"/\1/p' "$qc")
out="$game/models/$name"
if [ ! -d "$(dirname "$out")" ]; then
  echo "Error opening $game\\models/$name! (Check for write enable)"
  exit 1
fi
for ext in mdl vvd dx90.vtx; do printf 'IDST' > "${out%.mdl}.$ext"; done
echo "Completed \"$qc\""
"""


def test_real_compile_path_creates_output_folder(game, tmp_path):
    import stat
    import sys

    if sys.platform.startswith("win"):
        pytest.skip("shell-script stand-in for studiomdl")
    (game / "bin" / "studiomdl.exe").unlink()
    fake = game / "bin" / "studiomdl"
    fake.write_text(FAKE_STUDIOMDL)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    m = ModManager(str(game), log=lambda *_: None)
    m.install()
    vrm_path = tmp_path / "miku.vrm"
    vrm_path.write_bytes(fixtures.make_vrm("1.0"))
    av = m.add_avatar(str(vrm_path))
    mdir = os.path.join(m.mod_dir, "models", "vroid")
    assert sorted(os.listdir(mdir)) == [f"{av.model_id}.dx90.vtx", f"{av.model_id}.mdl", f"{av.model_id}.vvd"]
    # nothing left behind in Valve's folder
    assert not os.path.exists(game / "portal2" / "models" / "vroid")
    assert not os.path.exists(game / "portal2" / "materials" / "models" / "vroid")


def test_install_cleans_up_old_saved_cfg(game):
    cfg = game / "portal2" / "cfg"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "vroid_saved.cfg").write_text("Unknown command\n" * 1000)
    (cfg / "config.cfg").write_text('bind "w" "+forward"\ncon_logfile "cfg/vroid_saved.cfg"\nsensitivity "3"\n')
    m = ModManager(str(game), log=lambda *_: None)
    m.install()
    assert not (cfg / "vroid_saved.cfg").exists()
    assert (cfg / "config.cfg").read_text() == 'bind "w" "+forward"\nsensitivity "3"\n'
    boot = open(os.path.join(m.mod_dir, "cfg", "vroid", "vroid_boot.cfg")).read()
    assert 'con_logfile ""' in boot
    nut = open(os.path.join(m.mod_dir, "scripts", "vscripts", "vroid", "vroid_main.nut")).read()
    assert "con_logfile" not in nut and "exec vroid_saved" not in nut
