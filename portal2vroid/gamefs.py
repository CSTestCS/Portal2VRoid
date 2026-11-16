"""Locate Portal 2 and read original game files (loose files or VPKs)."""

from __future__ import annotations

import os
import re
import sys

from .vpk import VPK

PORTAL2_APPID = "620"
MOD_MARKER = "vroid_mod.txt"
MOD_FOLDER = "portal2_vroid"  # loaded through a gameinfo.txt search path, not as DLC


def _steam_roots() -> list[str]:
    roots: list[str] = []
    if sys.platform.startswith("win"):
        try:
            import winreg  # type: ignore

            for hive, key, value in (
                (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath"),
            ):
                try:
                    with winreg.OpenKey(hive, key) as k:
                        roots.append(winreg.QueryValueEx(k, value)[0])
                except OSError:
                    pass
        except ImportError:
            pass
        for env in ("ProgramFiles(x86)", "ProgramFiles"):
            base = os.environ.get(env)
            if base:
                roots.append(os.path.join(base, "Steam"))
    else:
        home = os.path.expanduser("~")
        roots += [
            os.path.join(home, ".steam", "steam"),
            os.path.join(home, ".local", "share", "Steam"),
            os.path.join(home, "Library", "Application Support", "Steam"),
        ]
    seen, out = set(), []
    for r in roots:
        r = os.path.normpath(r)
        if r not in seen and os.path.isdir(r):
            seen.add(r)
            out.append(r)
    return out


def _library_folders(steam_root: str) -> list[str]:
    libs = [steam_root]
    vdf = os.path.join(steam_root, "steamapps", "libraryfolders.vdf")
    try:
        with open(vdf, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for m in re.finditer(r'"path"\s+"([^"]+)"', text):
            libs.append(m.group(1).replace("\\\\", "\\"))
        # very old format: "1" "D:\\Games\\Steam"
        for m in re.finditer(r'"\d+"\s+"([A-Za-z]:\\\\[^"]+)"', text):
            libs.append(m.group(1).replace("\\\\", "\\"))
    except OSError:
        pass
    return libs


def is_portal2_dir(path: str) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "portal2", "gameinfo.txt"))


def find_portal2() -> str | None:
    for root in _steam_roots():
        for lib in _library_folders(root):
            cand = os.path.join(lib, "steamapps", "common", "Portal 2")
            if is_portal2_dir(cand):
                return os.path.normpath(cand)
    return None


def dlc_dirs(game_root: str) -> list[str]:
    """Existing portal2_dlcN folders in mount order (contiguous from 1)."""
    out = []
    n = 1
    while True:
        p = os.path.join(game_root, f"portal2_dlc{n}")
        if not os.path.isdir(p):
            break
        out.append(p)
        n += 1
    return out


def find_mod_dir(game_root: str) -> str | None:
    """The mod folder: portal2_vroid, or a portal2_dlcN folder from older versions."""
    own = os.path.join(game_root, MOD_FOLDER)
    if os.path.isfile(os.path.join(own, MOD_MARKER)):
        return own
    for d in dlc_dirs(game_root):
        if os.path.isfile(os.path.join(d, MOD_MARKER)):
            return d
    return None


def next_free_dlc_dir(game_root: str) -> str:
    return os.path.join(game_root, f"portal2_dlc{len(dlc_dirs(game_root)) + 1}")


class GameFS:
    """Read files the way the game would, ignoring our own mod folder.

    Priority (highest first): update/, portal2_dlcN..1 (except ours), portal2/.
    Each search path checks loose files first, then its pak01_dir.vpk.
    """

    def __init__(self, game_root: str, exclude_dir: str | None = None):
        self.game_root = game_root
        exclude = os.path.normcase(os.path.abspath(exclude_dir)) if exclude_dir else None
        paths = [os.path.join(game_root, "update")]
        paths += list(reversed(dlc_dirs(game_root)))
        paths.append(os.path.join(game_root, "portal2"))
        self.search: list[tuple[str, VPK | None]] = []
        for p in paths:
            if not os.path.isdir(p):
                continue
            if exclude and os.path.normcase(os.path.abspath(p)) == exclude:
                continue
            if os.path.isfile(os.path.join(p, MOD_MARKER)):
                continue
            vpk = None
            vpk_path = os.path.join(p, "pak01_dir.vpk")
            if os.path.isfile(vpk_path):
                try:
                    vpk = VPK(vpk_path)
                except (OSError, ValueError):
                    vpk = None
            self.search.append((p, vpk))

    def read(self, rel: str) -> bytes | None:
        rel = rel.replace("\\", "/").lower()
        for base, vpk in self.search:
            loose = os.path.join(base, *rel.split("/"))
            if os.path.isfile(loose):
                with open(loose, "rb") as f:
                    return f.read()
            # case-insensitive loose lookup (Linux installs)
            alt = _ci_lookup(base, rel)
            if alt:
                with open(alt, "rb") as f:
                    return f.read()
            if vpk is not None and rel in vpk:
                return vpk.read(rel)
        return None

    def exists(self, rel: str) -> bool:
        return self.read(rel) is not None

    def list(self, predicate, prefix: str = "") -> list[str]:
        """All file paths (lowercase, '/' separated) under ``prefix`` matching ``predicate``."""
        found: set[str] = set()
        prefix = prefix.strip("/").lower()
        for base, vpk in self.search:
            if vpk is not None:
                found.update(vpk.find(lambda p: p.startswith(prefix) and predicate(p)))
            root = os.path.join(base, *prefix.split("/")) if prefix else base
            for dirpath, _, files in os.walk(root):
                for fn in files:
                    rel = os.path.relpath(os.path.join(dirpath, fn), base).replace("\\", "/").lower()
                    if predicate(rel):
                        found.add(rel)
        return sorted(found)


def _ci_lookup(base: str, rel: str) -> str | None:
    if sys.platform.startswith("win"):
        return None
    cur = base
    for part in rel.split("/"):
        try:
            names = os.listdir(cur)
        except OSError:
            return None
        match = next((n for n in names if n.lower() == part), None)
        if match is None:
            return None
        cur = os.path.join(cur, match)
    return cur if os.path.isfile(cur) else None
