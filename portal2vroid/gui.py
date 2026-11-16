"""Tkinter manager window: install the mod, import VRM avatars, set camera defaults and keys."""

from __future__ import annotations

import os
import queue
import threading
import traceback

from . import __version__
from .gamefs import find_portal2, is_portal2_dir
from .installer import BIND_ACTIONS, ModManager, Settings

HELP_TEXT = (
    "1. Click 'Install / Repair'.\n"
    "2. Click 'Add VRM avatar...' and pick a .vrm file (VRoid Studio export).\n"
    "3. Start Portal 2 and load a single-player map.\n"
    "   Pause menu > 'VRoid Avatar / Camera' switches avatars and the camera.\n"
    "   Default key: V toggles third person.\n"
    "Note: avatar switching and third person turn on sv_cheats (achievements are\n"
    "paused while it is on). Chell + first person turns it back off."
)


def run_gui(game_arg: str | None = None) -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError:
        print("tkinter is not available; use the command line (Portal2VRoid --help).")
        return 1

    root = tk.Tk()
    root.title(f"Portal 2 VRoid Avatar Mod {__version__}")
    root.minsize(760, 640)
    logq: queue.Queue = queue.Queue()  # str = log line, callable = run on the UI thread
    state = {"mgr": None, "busy": False}
    log_path = _log_file_path()

    def log(msg: str):
        msg = str(msg)
        logq.put(msg)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass

    def ui(fn):
        """Run fn on the Tk thread (Tk must not be touched from worker threads)."""
        logq.put(fn)

    # ------------------------------------------------------------ layout
    main = ttk.Frame(root, padding=10)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=1)

    game_var = tk.StringVar(value=game_arg or find_portal2() or "")
    f_game = ttk.LabelFrame(main, text="Portal 2 folder", padding=6)
    f_game.grid(row=0, column=0, sticky="ew")
    f_game.columnconfigure(0, weight=1)
    ttk.Entry(f_game, textvariable=game_var).grid(row=0, column=0, sticky="ew")

    def browse_game():
        d = filedialog.askdirectory(title="Select the 'Portal 2' folder (contains portal2\\gameinfo.txt)")
        if d:
            game_var.set(d)
            refresh()

    ttk.Button(f_game, text="Browse...", command=browse_game).grid(row=0, column=1, padx=4)
    status_var = tk.StringVar()
    ttk.Label(f_game, textvariable=status_var).grid(row=1, column=0, sticky="w", pady=(4, 0))
    btn_install = ttk.Button(f_game, text="Install / Repair")
    btn_uninstall = ttk.Button(f_game, text="Uninstall")
    btn_install.grid(row=1, column=1, padx=4, pady=(4, 0))
    btn_uninstall.grid(row=1, column=2, padx=4, pady=(4, 0))

    # avatars
    f_av = ttk.LabelFrame(main, text="Avatars", padding=6)
    f_av.grid(row=1, column=0, sticky="nsew", pady=6)
    f_av.columnconfigure(0, weight=1)
    av_list = tk.Listbox(f_av, height=6, exportselection=False)
    av_list.grid(row=0, column=0, rowspan=5, sticky="nsew")
    btn_add = ttk.Button(f_av, text="Add VRM avatar...")
    btn_remove = ttk.Button(f_av, text="Remove")
    btn_default = ttk.Button(f_av, text="Use at startup")
    btn_chell = ttk.Button(f_av, text="Start as Chell")
    for i, b in enumerate((btn_add, btn_remove, btn_default, btn_chell)):
        b.grid(row=i, column=1, sticky="ew", padx=4, pady=1)
    hs = ttk.Frame(f_av)
    hs.grid(row=4, column=1, sticky="ew", padx=4)
    ttk.Label(hs, text="Height x").pack(side="left")
    height_var = tk.DoubleVar(value=1.0)
    ttk.Spinbox(hs, from_=0.5, to=2.0, increment=0.05, width=6, textvariable=height_var).pack(side="left")

    # camera + options
    f_cam = ttk.LabelFrame(main, text="Third-person camera (defaults)", padding=6)
    f_cam.grid(row=2, column=0, sticky="ew")
    f_cam.columnconfigure(1, weight=1)
    tp_var = tk.BooleanVar()
    dist_var, up_var, right_var = tk.IntVar(), tk.IntVar(), tk.IntVar()
    ttk.Checkbutton(f_cam, text="Start in third person", variable=tp_var).grid(row=0, column=0, columnspan=3, sticky="w")
    for r, (label, var, lo, hi) in enumerate(
        (("Distance", dist_var, 20, 400), ("Height", up_var, -64, 96), ("Horizontal (right +)", right_var, -96, 96)), start=1
    ):
        ttk.Label(f_cam, text=label).grid(row=r, column=0, sticky="w")
        ttk.Scale(f_cam, from_=lo, to=hi, variable=var, command=lambda v, var=var: var.set(int(float(v)))).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Label(f_cam, textvariable=var, width=5).grid(row=r, column=2)

    f_keys = ttk.LabelFrame(main, text="Keys (Source key names, e.g. v, f5, mouse4; blank = unbound)", padding=6)
    f_keys.grid(row=3, column=0, sticky="ew", pady=6)
    key_vars = {}
    for i, (action, label) in enumerate(BIND_ACTIONS):
        r, c = divmod(i, 3)
        ttk.Label(f_keys, text=label).grid(row=r, column=c * 2, sticky="e", padx=(6, 2))
        v = tk.StringVar()
        ttk.Entry(f_keys, textvariable=v, width=8).grid(row=r, column=c * 2 + 1, sticky="w")
        key_vars[action] = v

    f_opt = ttk.LabelFrame(main, text="Options", padding=6)
    f_opt.grid(row=4, column=0, sticky="ew")
    toon_var, jig_var, cheat_var, binds_var = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
    tex_var = tk.IntVar()
    ttk.Checkbutton(f_opt, text="Anime (toon) shading", variable=toon_var).grid(row=0, column=0, sticky="w")
    ttk.Checkbutton(f_opt, text="Hair/clothes physics (jigglebones)", variable=jig_var).grid(row=0, column=1, sticky="w")
    ttk.Checkbutton(f_opt, text="Turn sv_cheats off when not needed", variable=cheat_var).grid(row=1, column=0, sticky="w")
    ttk.Checkbutton(f_opt, text="Apply key bindings at game start", variable=binds_var).grid(row=1, column=1, sticky="w")
    ttk.Label(f_opt, text="Max texture size").grid(row=0, column=2, sticky="e", padx=6)
    ttk.Combobox(f_opt, textvariable=tex_var, values=(512, 1024, 2048, 4096), width=6, state="readonly").grid(row=0, column=3)
    btn_save = ttk.Button(f_opt, text="Save settings")
    btn_save.grid(row=1, column=3, sticky="e")
    btn_check = ttk.Button(f_opt, text="Check installation")
    btn_check.grid(row=1, column=2, sticky="e", padx=6)
    btn_studiomdl = ttk.Button(f_opt, text="studiomdl.exe...")
    btn_studiomdl.grid(row=2, column=3, sticky="e", pady=(4, 0))

    f_log = ttk.LabelFrame(main, text="Log", padding=6)
    f_log.grid(row=5, column=0, sticky="nsew", pady=(6, 0))
    main.rowconfigure(5, weight=1)
    txt = tk.Text(f_log, height=10, wrap="word")
    txt.pack(fill="both", expand=True)
    txt.insert("end", HELP_TEXT + "\n\n")

    # ------------------------------------------------------------ logic
    def mgr() -> ModManager | None:
        g = game_var.get().strip()
        if not is_portal2_dir(g):
            return None
        m = state["mgr"]
        if m is None or os.path.normcase(m.game_root) != os.path.normcase(os.path.abspath(g)):
            m = ModManager(g, log=log)
            state["mgr"] = m
        return m

    def load_settings_into_ui(s: Settings):
        tp_var.set(bool(s.third_person))
        dist_var.set(int(s.cam_dist))
        up_var.set(int(s.cam_up))
        right_var.set(int(s.cam_right))
        for a, v in key_vars.items():
            v.set((s.binds or {}).get(a, ""))
        toon_var.set(bool(s.toon_shading))
        jig_var.set(bool(s.jiggle_bones))
        cheat_var.set(bool(s.manage_cheats))
        binds_var.set(bool(s.apply_binds))
        tex_var.set(int(s.max_texture))

    def ui_settings(m: ModManager) -> Settings:
        s = m.load_settings()
        s.third_person = tp_var.get()
        s.cam_dist, s.cam_up, s.cam_right = dist_var.get(), up_var.get(), right_var.get()
        s.binds = {a: v.get().strip().lower() for a, v in key_vars.items()}
        s.toon_shading, s.jiggle_bones = toon_var.get(), jig_var.get()
        s.manage_cheats, s.apply_binds = cheat_var.get(), binds_var.get()
        s.max_texture = int(tex_var.get())
        return s

    def refresh():
        m = mgr()
        av_list.delete(0, "end")
        if m is None:
            status_var.set("Portal 2 not found - pick its folder.")
            load_settings_into_ui(Settings())
        else:
            if m.installed:
                status_var.set(f"Installed in {os.path.basename(m.mod_dir)}")
                s = m.load_settings()
                for a in m.avatars():
                    star = "  (startup)" if a.id == s.default_avatar else ""
                    av_list.insert("end", f"[{a.id}] {a.name}{star}")
                load_settings_into_ui(s)
            else:
                status_var.set("Not installed")
                load_settings_into_ui(Settings())
        busy = state["busy"]
        inst = m is not None and m.installed
        for b in (btn_uninstall, btn_remove, btn_default, btn_chell, btn_save):
            b.state(["!disabled"] if inst and not busy else ["disabled"])
        for b in (btn_install, btn_add, btn_check):
            b.state(["!disabled"] if m is not None and not busy else ["disabled"])

    def run_bg(fn, done_msg=None):
        state["busy"] = True
        refresh()

        def worker():
            try:
                fn()
                if done_msg:
                    log(done_msg)
            except Exception as e:  # noqa: BLE001 - shown to the user
                log("ERROR: " + str(e))
                log(traceback.format_exc())
                msg = str(e)[:1500]
                ui(lambda: messagebox.showerror("Portal2VRoid", msg + f"\n\nFull log: {log_path}"))
            finally:
                state["busy"] = False
                ui(refresh)

        threading.Thread(target=worker, daemon=True).start()

    def selected_id():
        sel = av_list.curselection()
        if not sel:
            return None
        text = av_list.get(sel[0])
        return int(text[1 : text.index("]")])

    def do_install():
        m = mgr()
        if m:
            run_bg(m.install, "Done. Start Portal 2 to use the mod.")

    def do_uninstall():
        m = mgr()
        if m and messagebox.askyesno("Uninstall", "Remove the mod and all converted avatars?"):
            run_bg(m.uninstall)

    def do_add():
        m = mgr()
        if not m:
            return
        path = filedialog.askopenfilename(title="Choose a VRoid avatar", filetypes=[("VRM avatar", "*.vrm"), ("glTF binary", "*.glb"), ("All files", "*.*")])
        if not path:
            return
        scale = float(height_var.get() or 1.0)
        if m.installed:
            m.save_settings(ui_settings(m))
        def work():
            log(f"=== Adding {path}")
            m.install()  # always refresh the mod files and game hooks first
            av = m.add_avatar(path, None, scale)
            names = ", ".join(a.name for a in m.avatars())
            log(f"Avatar list is now: {names}")
            ui(lambda: messagebox.showinfo(
                "Portal2VRoid",
                f"'{av.name}' was converted and installed.\n\nRestart Portal 2, load a single-player map, "
                "then press V (third person) or use Pause menu > VRoid Avatar / Camera.",
            ))

        run_bg(work)

    def do_remove():
        m, i = mgr(), selected_id()
        if m and i is not None and messagebox.askyesno("Remove avatar", "Delete this converted avatar?"):
            run_bg(lambda: m.remove_avatar(i))

    def set_default(i):
        m = mgr()
        if m and i is not None:
            s = ui_settings(m)
            s.default_avatar = i
            m.save_settings(s)
            log("Startup avatar saved.")
            refresh()

    def do_save():
        m = mgr()
        if m:
            m.save_settings(ui_settings(m))
            log("Settings saved. They apply the next time you start Portal 2.")

    btn_install.configure(command=do_install)
    btn_uninstall.configure(command=do_uninstall)
    btn_add.configure(command=do_add)
    btn_remove.configure(command=do_remove)
    btn_default.configure(command=lambda: set_default(selected_id()))
    btn_chell.configure(command=lambda: set_default(0))
    btn_save.configure(command=do_save)

    def do_check():
        m = mgr()
        if not m:
            messagebox.showwarning("Portal2VRoid", "Portal 2 folder not found.")
            return
        lines = m.diagnose()
        log("=== Installation check")
        for ln in lines:
            log("  " + ln)
        bad = [ln for ln in lines if ln.startswith("PROBLEM")]
        messagebox.showinfo("Installation check", "\n".join(lines) + (
            "\n\nEverything looks right." if not bad else "\n\nClick Install / Repair to fix problems."
        ) + f"\n\nLog file: {log_path}")

    btn_check.configure(command=do_check)

    def do_studiomdl():
        m = mgr()
        if not m or not m.installed:
            messagebox.showwarning("Portal2VRoid", "Click Install / Repair first.")
            return
        path = filedialog.askopenfilename(title="Locate studiomdl.exe", filetypes=[("studiomdl", "studiomdl.exe"), ("All files", "*.*")])
        if path:
            s = ui_settings(m)
            s.studiomdl_path = path
            m.save_settings(s)
            log(f"Using studiomdl: {path}")

    btn_studiomdl.configure(command=do_studiomdl)
    game_var.trace_add("write", lambda *_: refresh())

    def pump():
        try:
            while True:
                item = logq.get_nowait()
                if callable(item):
                    try:
                        item()
                    except Exception as e:  # noqa: BLE001
                        txt.insert("end", f"UI error: {e}\n")
                    continue
                txt.insert("end", item + "\n")
                txt.see("end")
        except queue.Empty:
            pass
        root.after(100, pump)

    refresh()
    pump()
    root.mainloop()
    return 0


def _log_file_path() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "Portal2VRoid")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = os.getcwd()
    return os.path.join(d, "Portal2VRoid.log")
