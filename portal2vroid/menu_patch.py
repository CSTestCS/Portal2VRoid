"""Add a "VRoid Avatar" entry (with a flyout sub-menu) to Portal 2's pause menu.

Portal 2's pause menu is a BaseModUI layout (resource/ui/basemodui/ingamemainmenu*.res).
We never ship Valve's files: the installer reads the player's own copy, inserts
one button + one FlyoutMenu control, and writes the patched copy into the mod folder.
"""

from __future__ import annotations

import statistics

from . import keyvalues as kv

FLYOUT_NAME = "FlmVRoidFlyout"
BUTTON_NAME = "BtnVRoid"
FLYOUT_RES = "resource/ui/basemodui/vroid_flyout.res"
BUTTON_LABEL = "VRoid Avatar / Camera"

# (control name, label, console command)
FLYOUT_ITEMS = [
    ("BtnVRoidTP", "Toggle Third-Person Camera", "vroid_tp_toggle"),
    ("BtnVRoidNext", "Next Avatar", "vroid_avatar_next"),
    ("BtnVRoidPrev", "Previous Avatar", "vroid_avatar_prev"),
    ("BtnVRoidChell", "Use Chell", "vroid_avatar_chell"),
    ("BtnVRoidDistP", "Camera Distance +", "vroid_cam_dist_more"),
    ("BtnVRoidDistM", "Camera Distance -", "vroid_cam_dist_less"),
    ("BtnVRoidUpP", "Camera Height +", "vroid_cam_up_more"),
    ("BtnVRoidUpM", "Camera Height -", "vroid_cam_up_less"),
    ("BtnVRoidRightP", "Camera Right", "vroid_cam_right_more"),
    ("BtnVRoidRightM", "Camera Left", "vroid_cam_right_less"),
    ("BtnVRoidReset", "Reset Camera", "vroid_cam_reset"),
    ("BtnVRoidSave", "Save as Default", "vroid_save"),
]


def _int(v) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _name(ctrl: kv.KV) -> str:
    return ctrl.get("fieldName") or ctrl.key


def _control_type(ctrl: kv.KV) -> str:
    return (ctrl.get("ControlName") or "").lower()


def patch_ingame_menu(text: str) -> tuple[str, str | None] | None:
    """Return (patched text, ResourceFile of the flyout we used as a template) or None."""
    root = kv.parse(text)
    tops = root.blocks()
    if not tops:
        return None
    top = tops[0]
    controls = top.blocks()
    if any(_name(c).lower() == BUTTON_NAME.lower() for c in controls):
        return None  # already patched

    buttons = [
        c
        for c in controls
        if "button" in _control_type(c) and c.get("command") and _int(c.get("ypos")) is not None
    ]
    if not buttons:
        return None
    # Main column = most common xpos among buttons.
    xs = [str(c.get("xpos")) for c in buttons]
    main_x = max(set(xs), key=xs.count)
    column = sorted((c for c in buttons if str(c.get("xpos")) == main_x), key=lambda c: _int(c.get("ypos")))
    if not column:
        return None

    def is_options(c):
        s = (str(c.get("command")) + " " + _name(c)).lower()
        return "option" in s

    anchor = next((c for c in column if is_options(c)), None)
    if anchor is None:
        anchor = column[-2] if len(column) >= 2 else column[-1]
    ys = [_int(c.get("ypos")) for c in column]
    diffs = [b - a for a, b in zip(ys, ys[1:]) if b > a]
    spacing = int(statistics.median(diffs)) if diffs else (_int(anchor.get("tall")) or 20)
    anchor_y = _int(anchor.get("ypos"))

    # Shift everything below the anchor in the main column down by one slot.
    for c in controls:
        y = _int(c.get("ypos"))
        if c is not anchor and y is not None and y > anchor_y and str(c.get("xpos")) == main_x:
            c.set("ypos", str(y + spacing))

    new_btn = anchor.clone()
    new_btn.key = BUTTON_NAME
    new_btn.set("fieldName", BUTTON_NAME)
    new_btn.set("ypos", str(anchor_y + spacing))
    new_btn.set("labelText", BUTTON_LABEL)
    new_btn.set("command", FLYOUT_NAME)
    for k in ("tooltiptext", "disabled_tooltiptext", "tabPosition", "ActivationType"):
        new_btn.remove(k)
    if new_btn.find("style") is None:
        new_btn.set("style", "MainMenuButton")

    # Navigation: anchor <-> new <-> whatever followed the anchor.
    old_down = anchor.get("navDown")
    new_btn.set("navUp", _name(anchor))
    if old_down:
        new_btn.set("navDown", old_down)
        for c in controls:
            if _name(c) == old_down and c.get("navUp"):
                c.set("navUp", BUTTON_NAME)
    anchor.set("navDown", BUTTON_NAME)

    # Flyout control: copy an existing FlyoutMenu entry if there is one.
    flyouts = [c for c in controls if "flyout" in _control_type(c)]
    template = next((c for c in flyouts if _name(c) == anchor.get("command")), flyouts[0] if flyouts else None)
    template_res = template.get("ResourceFile") if template is not None else None
    if template is not None:
        fly = template.clone()
    else:
        fly = kv.KV(FLYOUT_NAME, None, [])
        for k, v in (("ControlName", "FlyoutMenu"), ("xpos", "0"), ("ypos", "0"), ("zpos", "4"), ("wide", "0"), ("tall", "0"), ("visible", "0"), ("enabled", "1")):
            fly.set(k, v)
    fly.key = FLYOUT_NAME
    fly.set("fieldName", FLYOUT_NAME)
    fly.set("ResourceFile", FLYOUT_RES)
    fly.set("InitialFocus", FLYOUT_ITEMS[0][0])

    # Insert the button right after the anchor, the flyout at the end.
    idx = top.children.index(anchor)
    top.children.insert(idx + 1, new_btn)
    top.children.append(fly)
    return kv.dump(root), template_res


DEFAULT_FLYOUT_BUTTON = {
    "ControlName": "BaseModHybridButton",
    "xpos": "0",
    "ypos": "0",
    "wide": "180",
    "tall": "20",
    "autoResize": "1",
    "pinCorner": "0",
    "visible": "1",
    "enabled": "1",
    "tabPosition": "0",
    "wrap": "1",
    "style": "FlyoutMenuButton",
}


def build_flyout(template_text: str | None) -> str:
    """Build our flyout .res, reusing the look of an existing flyout when available."""
    btn_tpl: kv.KV | None = None
    bg_tpl: kv.KV | None = None
    y0, spacing = 0, 20
    if template_text:
        try:
            troot = kv.parse(template_text)
            ttop = troot.blocks()[0]
            tctrls = ttop.blocks()
            tbtns = sorted(
                (c for c in tctrls if "button" in _control_type(c) and _int(c.get("ypos")) is not None),
                key=lambda c: _int(c.get("ypos")),
            )
            if tbtns:
                btn_tpl = tbtns[0]
                y0 = _int(tbtns[0].get("ypos")) or 0
                ys = [_int(c.get("ypos")) for c in tbtns]
                diffs = [b - a for a, b in zip(ys, ys[1:]) if b > a]
                spacing = int(statistics.median(diffs)) if diffs else (_int(btn_tpl.get("tall")) or 20)
            bg_tpl = next((c for c in tctrls if _control_type(c) == "panel" or "background" in _name(c).lower()), None)
        except (IndexError, ValueError):
            btn_tpl = bg_tpl = None
    if btn_tpl is None:
        btn_tpl = kv.KV("Btn", None, [kv.KV(k, v) for k, v in DEFAULT_FLYOUT_BUTTON.items()])

    root = kv.KV("", None, [])
    top = kv.KV(FLYOUT_RES, None, [])
    root.children.append(top)

    n = len(FLYOUT_ITEMS)
    if bg_tpl is not None:
        bg = bg_tpl.clone()
        pad = 6
        bg.set("tall", str(y0 + spacing * n + pad))
        wide = _int(bg.get("wide"))
        btn_w = _int(btn_tpl.get("wide")) or 180
        if wide is None or wide < btn_w + 6:
            bg.set("wide", str(btn_w + 6))
        top.children.append(bg)
    else:
        bg = kv.KV("PnlBackground", None, [])
        for k, v in (
            ("ControlName", "Panel"), ("fieldName", "PnlBackground"), ("xpos", "0"), ("ypos", "0"), ("zpos", "-1"),
            ("wide", "186"), ("tall", str(spacing * n + 6)), ("visible", "1"), ("enabled", "1"),
            ("paintbackground", "1"), ("paintborder", "1"),
        ):
            bg.set(k, v)
        top.children.append(bg)

    for i, (name, label, cmd) in enumerate(FLYOUT_ITEMS):
        b = btn_tpl.clone()
        b.key = name
        b.set("fieldName", name)
        b.set("ypos", str(y0 + i * spacing))
        b.set("labelText", label)
        b.set("command", "engine " + cmd)
        b.set("navUp", FLYOUT_ITEMS[(i - 1) % n][0])
        b.set("navDown", FLYOUT_ITEMS[(i + 1) % n][0])
        for k in ("tooltiptext", "disabled_tooltiptext"):
            b.remove(k)
        b.set("tabPosition", "1" if i == 0 else "0")
        top.children.append(b)
    return kv.dump(root)


KB_SECTION = [
    ('"blank"', '"=========================="'),
    ('"blank"', '"VRoid Avatar Mod"'),
    ('"blank"', '"=========================="'),
    ('"vroid_tp_toggle"', '"Toggle third-person camera"'),
    ('"vroid_avatar_next"', '"Next VRoid avatar"'),
    ('"vroid_avatar_prev"', '"Previous VRoid avatar"'),
    ('"vroid_avatar_chell"', '"Switch back to Chell"'),
    ('"vroid_cam_dist_more"', '"Camera distance +"'),
    ('"vroid_cam_dist_less"', '"Camera distance -"'),
    ('"vroid_cam_up_more"', '"Camera height +"'),
    ('"vroid_cam_up_less"', '"Camera height -"'),
    ('"vroid_cam_right_more"', '"Camera right"'),
    ('"vroid_cam_right_less"', '"Camera left"'),
    ('"vroid_cam_reset"', '"Reset camera"'),
]


def patch_kb_act(text: str) -> str:
    """Append our actions to scripts/kb_act.lst so they show up in Options > Keyboard."""
    if "vroid_tp_toggle" in text:
        return text
    body = text.rstrip("\r\n\x00")
    lines = [f"{k}\t\t{v}" for k, v in KB_SECTION]
    nl = "\r\n" if "\r\n" in text else "\n"
    return body + nl + nl.join(lines) + nl
