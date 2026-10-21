"""Command-line interface. Run without arguments to open the GUI."""

from __future__ import annotations

import argparse
import os
import sys

from .gamefs import find_portal2, is_portal2_dir


def _game_root(arg: str | None) -> str:
    root = arg or find_portal2()
    if not root or not is_portal2_dir(root):
        sys.exit("Portal 2 was not found. Pass --game \"C:\\...\\steamapps\\common\\Portal 2\".")
    return root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="Portal2VRoid", description="Portal 2 VRoid avatar mod manager")
    p.add_argument("--game", help="Portal 2 install folder (auto-detected if omitted)")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("gui", help="open the graphical manager (default)")
    sub.add_parser("install", help="install or repair the mod")
    sub.add_parser("uninstall", help="remove the mod")
    sub.add_parser("list", help="list installed avatars")
    a = sub.add_parser("add", help="convert and install a .vrm avatar")
    a.add_argument("vrm")
    a.add_argument("--name")
    a.add_argument("--scale", type=float, default=1.0, help="height multiplier (1.0 = Chell's height)")
    r = sub.add_parser("remove", help="remove an avatar by id")
    r.add_argument("id", type=int)
    d = sub.add_parser("default", help="choose the avatar used at startup (0 = Chell)")
    d.add_argument("id", type=int)
    c = sub.add_parser("convert", help="offline: write SMD/QC/materials without installing or compiling")
    c.add_argument("vrm")
    c.add_argument("--chell-mdl", required=True, help="path to an extracted player.mdl (Chell)")
    c.add_argument("--chell-vvd", help="path to the matching player.vvd (for exact height)")
    c.add_argument("--out", required=True)
    c.add_argument("--scale", type=float, default=1.0)
    args = p.parse_args(argv)

    if args.cmd in (None, "gui"):
        from .gui import run_gui

        return run_gui(args.game)

    if args.cmd == "convert":
        return _offline_convert(args)

    from .installer import ModManager

    mgr = ModManager(_game_root(args.game))
    if args.cmd == "install":
        mgr.install()
    elif args.cmd == "uninstall":
        mgr.uninstall()
    elif args.cmd == "list":
        s = mgr.load_settings() if mgr.installed else None
        if not mgr.installed:
            print("Mod not installed.")
        for av in mgr.avatars():
            star = "*" if s and s.default_avatar == av.id else " "
            print(f"{star} [{av.id}] {av.name}  ({av.model_path})")
    elif args.cmd == "add":
        mgr.add_avatar(args.vrm, args.name, args.scale)
    elif args.cmd == "remove":
        mgr.remove_avatar(args.id)
    elif args.cmd == "default":
        s = mgr.load_settings()
        s.default_avatar = args.id
        mgr.save_settings(s)
    return 0


def _offline_convert(args) -> int:
    from .materials import write_materials
    from .mdl import parse_mdl, vvd_height
    from .retarget import ConvertOptions, convert
    from .source_export import write_model_sources
    from .vrm import load_vrm

    with open(args.chell_mdl, "rb") as f:
        chell = parse_mdl(f.read())
    height = None
    if args.chell_vvd:
        with open(args.chell_vvd, "rb") as f:
            height = vvd_height(f.read())
    vrm = load_vrm(args.vrm)
    res = convert(vrm, chell, ConvertOptions(height_scale=args.scale, chell_height=height))
    for w in res.warnings:
        print("!", w)
    model_id = "avatar"
    vmts = write_materials(vrm, args.out, model_id)
    qc = write_model_sources(os.path.join(args.out, "src"), model_id, res, chell, vmts)
    print(f"Wrote {qc} (scale {res.scale:.3f}, {len(res.bones)} bones)")
    for slot, bone in res.mapping.items():
        print(f"  {slot:24s} -> {bone}")
    return 0
