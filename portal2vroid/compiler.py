"""Run Portal 2's own studiomdl.exe and move the compiled model into the mod folder."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys


class CompileError(Exception):
    pass


MODEL_EXTS = (".mdl", ".vvd", ".dx90.vtx", ".dx80.vtx", ".sw.vtx", ".vtx", ".phy", ".ani")


def find_studiomdl(game_root: str) -> str | None:
    for rel in ("bin/studiomdl.exe", "bin/win64/studiomdl.exe", "bin/studiomdl"):
        p = os.path.join(game_root, *rel.split("/"))
        if os.path.isfile(p):
            return p
    return None


def compile_model(
    game_root: str,
    qc_path: str,
    model_id: str,
    mod_dir: str,
    log=print,
    studiomdl: str | None = None,
) -> list[str]:
    """Compile ``qc_path`` and move models/vroid/<model_id>.* into ``mod_dir``. Returns new file paths."""
    exe = studiomdl or find_studiomdl(game_root)
    if not exe:
        raise CompileError(
            "studiomdl.exe was not found in the Portal 2 'bin' folder. It ships with Portal 2 "
            "(Portal 2 Authoring Tools). Verify the game files in Steam and try again."
        )
    game_dir = os.path.join(game_root, "portal2")

    # studiomdl only sees gameinfo.txt search paths (not the DLC folder), so give it
    # temporary copies of our materials to read material flags from.
    tmp_mat = os.path.join(game_dir, "materials", "models", "vroid", model_id)
    src_mat = os.path.join(mod_dir, "materials", "models", "vroid", model_id)
    copied_mats = False
    if os.path.isdir(src_mat) and not os.path.exists(tmp_mat):
        shutil.copytree(src_mat, tmp_mat)
        copied_mats = True

    cmd = [exe, "-game", game_dir, "-nop4", "-nox360", "-verbose", qc_path]
    if not sys.platform.startswith("win") and exe.lower().endswith(".exe"):
        raise CompileError("studiomdl.exe can only be run on Windows.")
    env = dict(os.environ)
    env.setdefault("SteamAppId", "620")
    log("Running studiomdl ...")
    try:
        proc = subprocess.run(
            cmd,
            cwd=os.path.dirname(qc_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            timeout=1800,
        )
    finally:
        if copied_mats:
            shutil.rmtree(tmp_mat, ignore_errors=True)
            _prune_empty(os.path.join(game_dir, "materials", "models", "vroid"))
    output = proc.stdout or ""
    with open(os.path.join(os.path.dirname(qc_path), "studiomdl.log"), "w", encoding="utf-8") as f:
        f.write(output)
    errors = [ln for ln in output.splitlines() if "ERROR" in ln.upper() and "0 errors" not in ln.lower()]

    out_dir = os.path.join(mod_dir, "models", "vroid")
    os.makedirs(out_dir, exist_ok=True)
    moved = []
    for search in (os.path.join(game_dir, "models", "vroid"),):
        for path in glob.glob(os.path.join(search, model_id + ".*")):
            if path.lower().endswith(MODEL_EXTS):
                dst = os.path.join(out_dir, os.path.basename(path))
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(path, dst)
                moved.append(dst)
        _prune_empty(search)
    if proc.returncode != 0 or not any(p.lower().endswith(".mdl") for p in moved):
        tail = "\n".join((errors or output.splitlines())[-25:])
        raise CompileError(f"studiomdl failed (exit code {proc.returncode}).\n{tail}")
    for w in errors:
        log("  " + w)
    log(f"Compiled {len(moved)} model files.")
    return moved


def _prune_empty(path: str) -> None:
    try:
        if os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)
    except OSError:
        pass
