"""Install / update / uninstall the mod and manage converted avatars."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field

from . import __version__
from .compiler import compile_model
from .gamefs import MOD_FOLDER, MOD_MARKER, GameFS, dlc_dirs, find_mod_dir, next_free_dlc_dir
from .materials import write_materials
from .mdl import parse_mdl, vvd_positions
from .menu_patch import FLYOUT_RES, build_flyout, patch_ingame_menu, patch_kb_act
from .retarget import ConvertOptions, convert
from .source_export import write_model_sources
from .vpk import write_vpk
from .vrm import load_vrm

GAME_FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_files")
CHELL_MODEL = "models/player/chell/player.mdl"
SETTINGS_FILE = "vroid_settings.json"
AVATARS_FILE = "vroid_avatars.json"
SAVED_CFG = "vroid_saved.cfg"
HOOK_TAG = "// Portal2VRoid"
STUDIOMDL_HELP = (
    "studiomdl.exe (Valve's model compiler) is not installed. It is part of the free "
    "'Portal 2 Authoring Tools - Beta':\n\n"
    "1. In Steam, open Library and switch the filter from Games to Tools.\n"
    "2. Install 'Portal 2 Authoring Tools - Beta'. It installs into your Portal 2 folder "
    "(Portal 2\\bin\\studiomdl.exe).\n"
    "3. Click 'Add VRM avatar...' again.\n\n"
    "If you have studiomdl.exe elsewhere, set it with 'studiomdl.exe...' in the app."
)

BIND_ACTIONS = [
    ("vroid_tp_toggle", "Toggle third-person camera"),
    ("vroid_avatar_next", "Next avatar"),
    ("vroid_avatar_prev", "Previous avatar"),
    ("vroid_cam_dist_more", "Camera distance +"),
    ("vroid_cam_dist_less", "Camera distance -"),
    ("vroid_cam_up_more", "Camera height +"),
    ("vroid_cam_up_less", "Camera height -"),
    ("vroid_cam_right_more", "Camera right"),
    ("vroid_cam_right_less", "Camera left"),
]


@dataclass
class Settings:
    default_avatar: int = 0
    third_person: bool = True
    cam_dist: int = 110
    cam_up: int = 0
    cam_right: int = 20
    step_dist: int = 10
    step_up: int = 4
    step_right: int = 4
    manage_cheats: bool = True
    apply_binds: bool = True
    binds: dict = field(default_factory=lambda: {"vroid_tp_toggle": "v", "vroid_avatar_next": "", "vroid_avatar_prev": ""})
    max_texture: int = 2048
    toon_shading: bool = True
    jiggle_bones: bool = True
    studiomdl_path: str = ""


@dataclass
class Avatar:
    id: int
    name: str
    model_id: str
    source: str
    height_scale: float = 1.0
    added: float = 0.0

    @property
    def model_path(self) -> str:
        return f"models/vroid/{self.model_id}.mdl"


class ModManager:
    def __init__(self, game_root: str, log=print):
        self.game_root = os.path.abspath(game_root)
        self.log = log
        self.mod_dir = find_mod_dir(self.game_root)

    # ---------------------------------------------------------------- state
    @property
    def installed(self) -> bool:
        return self.mod_dir is not None

    def _path(self, *parts) -> str:
        assert self.mod_dir
        return os.path.join(self.mod_dir, *parts)

    def load_settings(self) -> Settings:
        s = Settings()
        if self.mod_dir and os.path.isfile(self._path(SETTINGS_FILE)):
            try:
                with open(self._path(SETTINGS_FILE), encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(s, k):
                        setattr(s, k, v)
            except (OSError, ValueError):
                pass
        return s

    def save_settings(self, s: Settings) -> None:
        with open(self._path(SETTINGS_FILE), "w", encoding="utf-8") as f:
            json.dump(asdict(s), f, indent=2)
        self.regenerate()

    def avatars(self) -> list[Avatar]:
        if not self.mod_dir or not os.path.isfile(self._path(AVATARS_FILE)):
            return []
        try:
            with open(self._path(AVATARS_FILE), encoding="utf-8") as f:
                return [Avatar(**a) for a in json.load(f)]
        except (OSError, ValueError, TypeError):
            return []

    def _save_avatars(self, avatars: list[Avatar]) -> None:
        with open(self._path(AVATARS_FILE), "w", encoding="utf-8") as f:
            json.dump([asdict(a) for a in avatars], f, indent=2)

    # -------------------------------------------------------------- install
    def install(self) -> str:
        """Create or repair the mod folder. Returns its path."""
        # Never edit gameinfo.txt: the first search path listed there becomes the game's
        # main (MOD / write) folder, which broke startup. Remove the line older versions added.
        self._unhook_gameinfo()
        legacy = os.path.join(self.game_root, MOD_FOLDER)
        if self.mod_dir is not None and os.path.normcase(os.path.abspath(self.mod_dir)) == os.path.normcase(legacy):
            self.mod_dir = self._migrate_from_legacy_folder(legacy)
        if self.mod_dir is None:
            self.mod_dir = next_free_dlc_dir(self.game_root)
            os.makedirs(self.mod_dir, exist_ok=True)
            self.log(f"Creating mod folder {self.mod_dir}")
        else:
            self.log(f"Updating mod folder {self.mod_dir}")
        with open(self._path(MOD_MARKER), "w", encoding="utf-8") as f:
            f.write(
                f"Portal 2 VRoid Avatar mod {__version__}\n"
                "This folder is managed by Portal2VRoid. Use the Portal2VRoid app to uninstall.\n"
            )
        fs = GameFS(self.game_root, exclude_dir=self.mod_dir)

        # static files
        for dirpath, _, files in os.walk(GAME_FILES):
            for fn in files:
                src = os.path.join(dirpath, fn)
                rel = os.path.relpath(src, GAME_FILES)
                dst = self._path(rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)

        # mapspawn.nut: keep the game's own content, then load our script
        orig = fs.read("scripts/vscripts/mapspawn.nut")
        text = orig.decode("utf-8", errors="replace") if orig else ""
        text = text.rstrip() + (
            "\n\n// --- Portal 2 VRoid Avatar mod ---\n"
            "try { IncludeScript(\"vroid/vroid_main\"); }\n"
            "catch (e) { printl(\"[VRoid] failed to load: \" + e); }\n"
        )
        self._write_text("scripts/vscripts/mapspawn.nut", text)

        self._patch_menus(fs)

        kb = fs.read("scripts/kb_act.lst")
        if kb:
            enc = "utf-16" if kb[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
            try:
                patched = patch_kb_act(kb.decode(enc, errors="replace").lstrip("﻿"))
                self._write_text("scripts/kb_act.lst", patched, encoding=enc)
                self.log("Added VRoid actions to Options > Keyboard.")
            except UnicodeError:
                self.log("! Could not patch kb_act.lst (unknown encoding); keybinds still work from the console.")
        else:
            self.log("! scripts/kb_act.lst not found; keybinds still work, but won't be listed in Options.")

        if not os.path.isfile(self._path(SETTINGS_FILE)):
            self.save_settings(Settings())
        else:
            self.regenerate()
        self._hook_autoexec()
        self._remove_saved_cfg()
        for line in self.diagnose():
            self.log("  " + line)
        self.log("Install complete. Restart Portal 2 if it is running.")
        return self.mod_dir

    # ------------------------------------------------------------- hooks
    def _gameinfo_path(self) -> str:
        return os.path.join(self.game_root, "portal2", "gameinfo.txt")

    # Files and folders inside the mod folder that belong to the mod.
    OWN_PATHS = (
        MOD_MARKER, SETTINGS_FILE, AVATARS_FILE, "pak01_dir.vpk", "_build", "models/vroid",
        "materials/models/vroid", "scripts/vscripts/mapspawn.nut", "scripts/vscripts/vroid",
        "scripts/kb_act.lst", "cfg/vroid", "cfg/vroid.cfg", "resource/ui/basemodui",
    )

    def _migrate_from_legacy_folder(self, legacy: str) -> str:
        """Move the mod out of portal2_vroid (used with the gameinfo.txt line) into portal2_dlcN.

        While that folder was the game's write folder, Portal 2 may have saved games and
        settings into it: save games are moved back to portal2/SAVE, caches are dropped.
        """
        target = next_free_dlc_dir(self.game_root)
        os.makedirs(target, exist_ok=True)
        for rel in self.OWN_PATHS:
            src = os.path.join(legacy, *rel.split("/"))
            dst = os.path.join(target, *rel.split("/"))
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif os.path.isfile(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
        save_src = os.path.join(legacy, "SAVE")
        if os.path.isdir(save_src):
            save_dst = os.path.join(self.game_root, "portal2", "SAVE")
            moved = 0
            for dirpath, _, fns in os.walk(save_src):
                for fn in fns:
                    s = os.path.join(dirpath, fn)
                    d = os.path.join(save_dst, os.path.relpath(s, save_src))
                    if not os.path.exists(d):
                        os.makedirs(os.path.dirname(d), exist_ok=True)
                        shutil.copy2(s, d)
                        moved += 1
            if moved:
                self.log(f"Moved {moved} save file(s) back to portal2\\SAVE.")
        shutil.rmtree(legacy)
        self.log(f"Moved the mod from {MOD_FOLDER} to {os.path.basename(target)}.")
        return target

    def _unhook_gameinfo(self) -> None:
        path = self._gameinfo_path()
        if not os.path.isfile(path):
            return
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        new = _strip_tagged_lines(text)
        if new != text:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(new)
            self.log("Removed the line older versions added to portal2/gameinfo.txt.")

    def _hook_autoexec(self) -> None:
        """Run our boot cfg at game start (keys + console commands work even before a map loads)."""
        path = os.path.join(self.game_root, "portal2", "cfg", "autoexec.cfg")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        text = ""
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        text = _strip_tagged_lines(text)
        if text and not text.endswith("\n"):
            text += "\n"
        text += f"exec vroid/vroid_boot {HOOK_TAG}\n"
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)

    def _remove_saved_cfg(self) -> None:
        """Clean up after the removed in-game "Save as Default" feature.

        It wrote cfg/vroid_saved.cfg through con_logfile. If logging stayed switched on,
        that file kept collecting console output and was replayed at every start.
        """
        for d in [os.path.join(self.game_root, x) for x in os.listdir(self.game_root)]:
            cfg = os.path.join(d, "cfg", SAVED_CFG)
            if os.path.isfile(cfg):
                size = os.path.getsize(cfg)
                os.remove(cfg)
                self.log(f"Removed {cfg} ({size} bytes) left by an older version.")
        config = os.path.join(self.game_root, "portal2", "cfg", "config.cfg")
        if os.path.isfile(config):
            with open(config, encoding="utf-8", errors="replace") as f:
                text = f.read()
            new = "".join(ln for ln in text.splitlines(keepends=True) if "vroid_saved" not in ln)
            if new != text:
                with open(config, "w", encoding="utf-8", newline="") as f:
                    f.write(new)
                self.log("Turned off console logging into vroid_saved.cfg in config.cfg.")

    def _unhook(self) -> None:
        for path in (self._gameinfo_path(), os.path.join(self.game_root, "portal2", "cfg", "autoexec.cfg")):
            if os.path.isfile(path):
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
                new = _strip_tagged_lines(text)
                if new != text:
                    with open(path, "w", encoding="utf-8", newline="") as f:
                        f.write(new)

    def diagnose(self) -> list[str]:
        """Human-readable checklist of everything the mod needs."""
        from .compiler import find_studiomdl

        out = []

        def check(ok, msg):
            out.append(("OK    " if ok else "PROBLEM ") + msg)

        if not self.mod_dir:
            check(False, "mod is not installed (click Install / Repair)")
            return out
        check(True, f"mod folder: {self.mod_dir}")
        try:
            with open(self._gameinfo_path(), encoding="utf-8", errors="replace") as f:
                gi = f.read()
            check(HOOK_TAG not in gi, "portal2/gameinfo.txt is unmodified" + ("" if HOOK_TAG not in gi else
                  " (an older version's line is still there - click Install / Repair)"))
        except OSError:
            check(False, "cannot read portal2/gameinfo.txt")
        ae = os.path.join(self.game_root, "portal2", "cfg", "autoexec.cfg")
        ok = os.path.isfile(ae) and HOOK_TAG in open(ae, encoding="utf-8", errors="replace").read()
        check(ok, "autoexec.cfg loads the mod's keys and commands")
        vpk_path = self._path("pak01_dir.vpk")
        try:
            from .vpk import VPK

            v = VPK(vpk_path)
            for rel in ("scripts/vscripts/mapspawn.nut", "scripts/vscripts/vroid/vroid_main.nut", "cfg/vroid/vroid_boot.cfg"):
                check(rel in v, f"pak01_dir.vpk contains {rel}")
        except (OSError, ValueError):
            check(False, "pak01_dir.vpk missing or unreadable (click Install / Repair)")
        avatars = self.avatars()
        check(bool(avatars), f"{len(avatars)} avatar(s) converted")
        for a in avatars:
            check(os.path.isfile(self._path("models", "vroid", a.model_id + ".mdl")), f"model file for '{a.name}' ({a.model_path})")
        custom = self.load_settings().studiomdl_path
        have = (custom and os.path.isfile(custom)) or find_studiomdl(self.game_root) is not None
        check(have, "studiomdl.exe found (needed to convert avatars)" + ("" if have else
              " - install 'Portal 2 Authoring Tools - Beta' from Steam Library > Tools"))
        return out

    def _patch_menus(self, fs: GameFS) -> None:
        menus = fs.list(lambda p: "ingamemainmenu" in p and p.endswith(".res"), prefix="resource/ui")
        template_text = None
        patched_any = False
        for rel in menus:
            data = fs.read(rel)
            if not data:
                continue
            result = patch_ingame_menu(data.decode("utf-8", errors="replace"))
            if result is None:
                self.log(f"! Could not find the button list in {rel}; left unchanged.")
                continue
            text, tpl = result
            self._write_text(rel, text)
            patched_any = True
            self.log(f"Added the VRoid entry to the pause menu ({rel}).")
            if tpl and template_text is None:
                tdata = fs.read(tpl.replace("\\", "/").lower())
                if tdata:
                    template_text = tdata.decode("utf-8", errors="replace")
        if not patched_any:
            self.log("! Pause menu layout not found. Use the keybinds / console commands instead (see README).")
        self._write_text(FLYOUT_RES, build_flyout(template_text))

    def _write_text(self, rel: str, text: str, encoding: str = "utf-8") -> None:
        dst = self._path(*rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding=encoding, newline="") as f:
            f.write(text)

    # ----------------------------------------------------------- generated
    def regenerate(self) -> None:
        s = self.load_settings()
        avatars = self.avatars()
        ids = {a.id for a in avatars}
        default = s.default_avatar if s.default_avatar in ids else 0
        lines = ["// Generated by Portal2VRoid - edit avatars with the Portal2VRoid app", "::VRoidAvatars <- ["]
        for a in avatars:
            name = _squirrel_str(a.name)
            lines.append(
                f'\t{{ id = {a.id}, name = "{name}", model = "{a.model_path}", short = "vroid/{a.model_id}" }},'
            )
        lines.append("];")
        lines.append(
            "::VRoidDefaults <- { "
            f"avatar = {default}, thirdperson = {1 if s.third_person else 0}, "
            f"dist = {int(s.cam_dist)}, up = {int(s.cam_up)}, right = {int(s.cam_right)}, "
            f"step_dist = {int(s.step_dist)}, step_up = {int(s.step_up)}, step_right = {int(s.step_right)}, "
            f"manage_cheats = {1 if s.manage_cheats else 0} }};"
        )
        self._write_text("scripts/vscripts/vroid/vroid_avatars.nut", "\n".join(lines) + "\n")

        binds = ["// Generated by Portal2VRoid (key bindings)"]
        if s.apply_binds:
            for action, key in (s.binds or {}).items():
                key = str(key).strip().lower()
                if key and re.fullmatch(r"[a-z0-9_]+", key) and action in dict(BIND_ACTIONS):
                    binds.append(f'bind "{key}" "{action}"')
        self._write_text("cfg/vroid/vroid_binds.cfg", "\n".join(binds) + "\n")
        self._pack_vpk()

    def _pack_vpk(self) -> None:
        """Pack the small override files (scripts, cfg, menus) into pak01_dir.vpk.

        Portal 2 searches VPK archives before loose folders, so loose overrides
        (mapspawn.nut, the pause menu, kb_act.lst) would be hidden by Valve's VPK.
        Models and textures have unique names and stay loose.
        """
        files = {}
        skip = {MOD_MARKER, SETTINGS_FILE, AVATARS_FILE, "pak01_dir.vpk", "placeholder.txt"}
        for dirpath, dirnames, fns in os.walk(self.mod_dir):
            dirnames[:] = [d for d in dirnames if not d.startswith("_")]
            for fn in fns:
                if fn in skip or not fn.lower().endswith((".nut", ".cfg", ".res", ".lst")):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, self.mod_dir).replace("\\", "/")
                with open(full, "rb") as f:
                    files[rel] = f.read()
        write_vpk(self._path("pak01_dir.vpk"), files)

    # ------------------------------------------------------------ uninstall
    def uninstall(self) -> None:
        if not self.mod_dir:
            self.log("Mod is not installed.")
            return
        mod_dir = self.mod_dir
        if not os.path.isfile(os.path.join(mod_dir, MOD_MARKER)):
            raise RuntimeError(f"{mod_dir} is not a Portal2VRoid folder; refusing to delete it.")
        self._unhook()
        self._remove_saved_cfg()
        self._remove_mod_folder(mod_dir)
        for cfg in (
            os.path.join(self.game_root, "portal2", "cfg", SAVED_CFG),
            os.path.join(self.game_root, "update", "cfg", SAVED_CFG),
        ):
            if os.path.isfile(cfg):
                os.remove(cfg)
        self.mod_dir = None
        self.log("Uninstalled.")

    def _remove_mod_folder(self, path: str) -> None:
        dlcs = [os.path.normcase(os.path.abspath(d)) for d in dlc_dirs(self.game_root)]
        here = os.path.normcase(os.path.abspath(path))
        shutil.rmtree(path)
        if here in dlcs and dlcs[-1] != here:
            # Later portal2_dlcN folders only mount if the numbering has no gaps.
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "placeholder.txt"), "w", encoding="utf-8") as f:
                f.write("Left empty on purpose so the next portal2_dlc folders keep loading.\n")

    # -------------------------------------------------------------- avatars
    def load_chell(self, fs: GameFS | None = None):
        fs = fs or GameFS(self.game_root, exclude_dir=self.mod_dir)
        data = fs.read(CHELL_MODEL)
        path = CHELL_MODEL
        if data is None:
            cands = fs.list(lambda p: p.endswith(".mdl") and "chell" in p and "player" in p, prefix="models")
            if not cands:
                raise RuntimeError("Could not find Chell's player model in the game files.")
            path = cands[0]
            data = fs.read(path)
        skel = parse_mdl(data)
        vvd = fs.read(path[:-4] + ".vvd")
        verts = vvd_positions(vvd) if vvd else None
        # Chell's animations live in included models (e.g. player_animations.mdl). Their
        # IK chains must be declared on our model too, so read them if player.mdl has none.
        for inc in skel.include_models:
            rel = inc.replace("\\", "/").lower()
            if not rel.startswith("models/"):
                rel = "models/" + rel
            inc_data = fs.read(rel)
            if inc_data is None:
                self.log(f"  ! Chell's included model {rel} not found")
                continue
            try:
                inc_skel = parse_mdl(inc_data)
            except ValueError as e:
                self.log(f"  ! could not read {rel}: {e}")
                continue
            if not skel.ikchains and inc_skel.ikchains:
                names = {b.name: i for i, b in enumerate(skel.bones)}
                remapped = []
                for ch in inc_skel.ikchains:
                    links = [names.get(inc_skel.bones[b].name, -1) for b in ch.links]
                    if all(x >= 0 for x in links):
                        ch.links = links
                        remapped.append(ch)
                skel.ikchains = remapped
                self.log(f"  IK chains from {rel}: {', '.join(c.name for c in remapped) or 'none'}")
        return skel, path, verts

    def add_avatar(self, vrm_path: str, name: str | None = None, height_scale: float = 1.0) -> Avatar:
        if not self.mod_dir:
            self.install()
        log = self.log
        s = self.load_settings()
        avatars = self.avatars()
        new_id = max([a.id for a in avatars] + [0]) + 1

        log(f"Reading {os.path.basename(vrm_path)} ...")
        vrm = load_vrm(vrm_path)
        display = (name or vrm.title or os.path.splitext(os.path.basename(vrm_path))[0]).strip()
        if not display.isascii():
            display = os.path.splitext(os.path.basename(vrm_path))[0]
            if not display.isascii():
                display = f"Avatar {new_id}"
        slug = re.sub(r"[^a-z0-9]+", "_", display.lower()).strip("_")[:24] or "avatar"
        model_id = f"a{new_id}_{slug}"
        log(f"  VRM {vrm.version}, {len(vrm.nodes)} nodes, {sum(len(p.indices) for p in vrm.primitives)} triangles")

        from .compiler import CompileError, find_studiomdl

        if not (s.studiomdl_path and os.path.isfile(s.studiomdl_path)) and find_studiomdl(self.game_root) is None:
            raise CompileError(STUDIOMDL_HELP)

        chell, chell_path, chell_verts = self.load_chell()
        log(f"  Chell: {len(chell.bones)} bones, {len(chell.ikchains)} IK chains, includes {chell.include_models or 'none'}")
        diffs = chell.rest_vs_bind_report()
        log(f"  Chell rest vs bind pose: {len(diffs)} bone(s) differ")
        for d in diffs[:40]:
            log("    " + d)
        result = convert(
            vrm,
            chell,
            ConvertOptions(height_scale=height_scale, jiggle=s.jiggle_bones, chell_vertices=chell_verts),
        )
        log("  bone mapping: " + ", ".join(f"{k}={v}" for k, v in result.mapping.items()))
        for w in result.warnings:
            log("  ! " + w)
        log(f"  scale {result.scale:.2f}, {len(result.bones)} bones ({len(result.bones) - result.n_chell} from the avatar)")

        build_dir = self._path("_build", model_id)
        try:
            log("Converting materials ...")
            vmts = write_materials(vrm, self.mod_dir, model_id, s.max_texture, s.toon_shading, log=log)
            include = chell_path[len("models/") :] if chell_path.startswith("models/") else chell_path
            extra = [i[len("models/"):] if i.lower().startswith("models/") else i for i in chell.include_models]
            qc = write_model_sources(build_dir, model_id, result, chell, vmts, [include] + extra)
            compile_model(self.game_root, qc, model_id, self.mod_dir, log=log, studiomdl=s.studiomdl_path or None)
        except Exception:
            self._remove_model_files(model_id)
            raise
        # Keep the generated sources (not packed into the game) for "Export debug files".

        av = Avatar(new_id, display, model_id, os.path.abspath(vrm_path), height_scale, time.time())
        avatars.append(av)
        self._save_avatars(avatars)
        if s.default_avatar == 0:
            s.default_avatar = new_id
            self.save_settings(s)
        else:
            self.regenerate()
        log(f"Added avatar '{display}' (id {new_id}).")
        return av

    def export_debug(self, zip_path: str, log_file: str | None = None) -> str:
        """Zip Chell's model files, the mod settings and the generated model sources."""
        import zipfile

        fs = GameFS(self.game_root, exclude_dir=self.mod_dir)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            skel_data = fs.read(CHELL_MODEL)
            wanted = [CHELL_MODEL, CHELL_MODEL[:-4] + ".vvd"]
            if skel_data:
                try:
                    for inc in parse_mdl(skel_data).include_models:
                        rel = inc.replace("\\", "/").lower()
                        wanted.append(rel if rel.startswith("models/") else "models/" + rel)
                except ValueError:
                    pass
            for rel in wanted:
                data = fs.read(rel)
                if data:
                    z.writestr("game/" + rel, data)
            if self.mod_dir:
                for fn in (SETTINGS_FILE, AVATARS_FILE):
                    if os.path.isfile(self._path(fn)):
                        z.write(self._path(fn), fn)
                build = self._path("_build")
                for dirpath, _, fns in os.walk(build):
                    for fn in fns:
                        full = os.path.join(dirpath, fn)
                        z.write(full, "build/" + os.path.relpath(full, build).replace("\\", "/"))
                for rel in ("scripts/vscripts/vroid/vroid_avatars.nut",):
                    if os.path.isfile(self._path(*rel.split("/"))):
                        z.write(self._path(*rel.split("/")), "mod/" + rel)
            if log_file and os.path.isfile(log_file):
                z.write(log_file, "Portal2VRoid.log")
        self.log(f"Debug files written to {zip_path}")
        return zip_path

    def remove_avatar(self, avatar_id: int) -> None:
        avatars = self.avatars()
        keep = [a for a in avatars if a.id != avatar_id]
        for a in avatars:
            if a.id == avatar_id:
                self._remove_model_files(a.model_id)
        self._save_avatars(keep)
        s = self.load_settings()
        if s.default_avatar == avatar_id:
            s.default_avatar = keep[0].id if keep else 0
        self.save_settings(s)

    def _remove_model_files(self, model_id: str) -> None:
        mdir = self._path("models", "vroid")
        if os.path.isdir(mdir):
            for fn in os.listdir(mdir):
                if fn.split(".", 1)[0] == model_id:
                    os.remove(os.path.join(mdir, fn))
        shutil.rmtree(self._path("materials", "models", "vroid", model_id), ignore_errors=True)
        shutil.rmtree(self._path("_build", model_id), ignore_errors=True)


def _strip_tagged_lines(text: str) -> str:
    return "".join(ln for ln in text.splitlines(keepends=True) if HOOK_TAG not in ln)


def _squirrel_str(s: str) -> str:
    return re.sub(r'["\\\x00-\x1f]', "", s)[:48]


def _prune(path: str) -> None:
    try:
        if os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)
    except OSError:
        pass
