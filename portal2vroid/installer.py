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
from .gamefs import MOD_MARKER, GameFS, dlc_dirs, find_mod_dir, next_free_dlc_dir
from .materials import write_materials
from .mdl import parse_mdl, vvd_height
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
        self.log("Install complete.")
        return self.mod_dir

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
        """Duplicate the small text files into pak01_dir.vpk (belt and braces for mounting)."""
        files = {}
        for dirpath, dirnames, fns in os.walk(self.mod_dir):
            dirnames[:] = [d for d in dirnames if not d.startswith("_")]
            for fn in fns:
                if not fn.lower().endswith((".nut", ".cfg", ".res", ".lst")):
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
        dlcs = [os.path.normcase(os.path.abspath(d)) for d in dlc_dirs(self.game_root)]
        here = os.path.normcase(os.path.abspath(mod_dir))
        is_last = bool(dlcs) and dlcs[-1] == here
        shutil.rmtree(mod_dir)
        if not is_last:
            # Later portal2_dlcN folders only mount if the numbering has no gaps.
            os.makedirs(mod_dir, exist_ok=True)
            with open(os.path.join(mod_dir, "placeholder.txt"), "w", encoding="utf-8") as f:
                f.write("Left empty on purpose so the next portal2_dlc folders keep loading.\n")
        for cfg in (
            os.path.join(self.game_root, "portal2", "cfg", SAVED_CFG),
            os.path.join(self.game_root, "update", "cfg", SAVED_CFG),
        ):
            if os.path.isfile(cfg):
                os.remove(cfg)
        self.mod_dir = None
        self.log("Uninstalled.")

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
        height = vvd_height(vvd) if vvd else None
        return skel, path, height

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

        chell, chell_path, chell_height = self.load_chell()
        log(f"  Chell: {len(chell.bones)} bones, height {chell_height or 0:.1f} units, {len(chell.ikchains)} IK chains")
        result = convert(
            vrm,
            chell,
            ConvertOptions(height_scale=height_scale, jiggle=s.jiggle_bones, chell_height=chell_height),
        )
        for w in result.warnings:
            log("  ! " + w)
        log(f"  scale {result.scale:.2f}, {len(result.bones)} bones ({len(result.bones) - result.n_chell} from the avatar)")

        build_dir = self._path("_build", model_id)
        try:
            log("Converting materials ...")
            vmts = write_materials(vrm, self.mod_dir, model_id, s.max_texture, s.toon_shading, log=log)
            include = chell_path[len("models/") :] if chell_path.startswith("models/") else chell_path
            qc = write_model_sources(build_dir, model_id, result, chell, vmts, include)
            compile_model(self.game_root, qc, model_id, self.mod_dir, log=log)
        except Exception:
            self._remove_model_files(model_id)
            raise
        shutil.rmtree(build_dir, ignore_errors=True)
        _prune(self._path("_build"))

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


def _squirrel_str(s: str) -> str:
    return re.sub(r'["\\\x00-\x1f]', "", s)[:48]


def _prune(path: str) -> None:
    try:
        if os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)
    except OSError:
        pass
