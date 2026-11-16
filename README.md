# Portal 2 VRoid Avatar Mod

Play Portal 2 as your own **VRoid / VRM avatar** instead of Chell, with a
**third-person camera** you can toggle with a key and adjust from the pause menu.

* Pause menu entry **VRoid Avatar / Camera** for switching avatars, toggling third person and
  moving the camera (distance, height, left/right).
* Any `.vrm` file (VRoid Studio exports, VRM 0.x and 1.0) is converted to a Source model that
  plays Chell's own animations.
* The avatar **keeps its own body proportions** (limb lengths, head size, etc.) and is
  **scaled to Chell's height**.
* Key binding for third person (default **V**), also listed under *Options → Keyboard*.
* Hair, skirts and other VRM "spring bones" swing via Source jigglebones. A cel-shading
  light ramp gives the avatar an anime look.
* One `.exe`. It finds Portal 2 through Steam, installs into its own `portal2_vroid` folder, and never
  modifies Valve's files.

## Install

1. Download **`Portal2VRoid.exe`** from the [Releases](../../releases) page, or from the
   *Portal2VRoid-windows* artifact of the latest **Build** workflow run.
2. Run it. The Portal 2 folder is detected automatically.
3. Click **Install / Repair**.
4. Click **Add VRM avatar...** and pick your `.vrm` file. Conversion takes about 10–60 seconds.
   The first avatar you add becomes the startup avatar.
5. Start Portal 2 and load any single-player chapter or save.

To uninstall, click **Uninstall** in the app (or delete the `portal2_vroid` folder that holds
`vroid_mod.txt`).

## In game

| Action | Default key | Pause menu → VRoid Avatar / Camera | Console |
|---|---|---|---|
| Toggle third-person camera | `V` | Toggle Third-Person Camera | `vroid_tp_toggle` |
| Next / previous avatar | – | Next Avatar / Previous Avatar | `vroid_avatar_next`, `vroid_avatar_prev` |
| Back to Chell | – | Use Chell | `vroid_avatar_chell` |
| Camera distance | – | Camera Distance + / − | `vroid_cam_dist_more`, `vroid_cam_dist_less` |
| Camera height | – | Camera Height + / − | `vroid_cam_up_more`, `vroid_cam_up_less` |
| Camera horizontal position | – | Camera Right / Camera Left | `vroid_cam_right_more`, `vroid_cam_right_less` |
| Reset camera | – | Reset Camera | `vroid_cam_reset` |
| Status / installed avatars | – | – | `vroid_status` |

Changes made from the pause menu take effect when you resume the game. All actions can be bound
in *Options → Keyboard* (section "VRoid Avatar Mod"), in the app's **Keys** box, or in the
console (`bind f5 vroid_avatar_next`).

The app also sets **defaults**: whether the game starts in third person, camera distance /
height / horizontal offset, startup avatar, keys, texture size, toon shading and jigglebones.
Click **Save settings**, then restart Portal 2.

## Things to know

* **Single player only.** In co-op the mod switches itself off.
* **sv_cheats.** Source only lets the game switch to a model that is loaded ("precached") for
  the current map, and Portal 2 scripts can't load one directly. The mod therefore briefly
  turns on `sv_cheats` to spawn and delete a hidden prop of your avatar. Portal 2's third-person
  camera also requires `sv_cheats 1`. While cheats are on, achievements are paused for that map.
  With the startup avatar set to Chell and third person off, the mod never touches cheats.
* **Conversion needs `studiomdl.exe`**, Valve's model compiler. It comes with the free
  **Portal 2 Authoring Tools - Beta**: in Steam, open *Library*, switch the filter from
  *Games* to *Tools*, and install it. It installs into `Portal 2\bin\`. If your copy is
  elsewhere, point the app at it with the **studiomdl.exe...** button. Conversion only runs
  on Windows.
* **Source limits.** A model can have at most 128 bones and 3 bone weights per vertex. Extra
  VRM bones (usually hair tips) are merged into their parents to fit. Facial expressions and
  MToon outlines are not converted.
* If an avatar is the wrong size, remove it, set **Height x** (for example `1.1` = 10% taller
  than Chell), and add it again.

## Troubleshooting

1. In the app, click **Check installation**. Every line should say `OK`. If any says
   `PROBLEM`, click **Install / Repair** and check again. Steam's "Verify integrity of game
   files" removes the mod's line from `portal2/gameinfo.txt`, so run **Install / Repair**
   again after verifying.
2. Restart Portal 2 completely after installing or adding avatars.
3. In a single-player map, open the console (enable it under *Options → Keyboard →
   Advanced*) and type `vroid_status`. Lines starting with `[VRoid]` show which avatar is
   active, the player's current model, and whether the mod's update loop is running.
   If `vroid_status` is an unknown command, the mod folder isn't being loaded (see step 1).
   `exec vroid` starts the mod by hand on the current map.
4. The app keeps a full log at `%LOCALAPPDATA%\Portal2VRoid\Portal2VRoid.log`. If a
   conversion fails, studiomdl's output is in `portal2_vroid\_build\<avatar>\studiomdl.log`.

The installer changes two of Portal 2's own files, and **Uninstall** puts both back:
`portal2/gameinfo.txt` (one search-path line, backup kept as `gameinfo.txt.vroid_backup`) and
`portal2/cfg/autoexec.cfg` (one `exec` line, so keys work from game start).

## How it works

```
your.vrm ──► Portal2VRoid.exe ──► portal2_vroid/
               │  read Chell's skeleton from the game's VPK
               │  place + scale avatar, repose T-pose → Chell's rest pose
               │  build skeleton: Chell's bone names & orientations, avatar's joint positions
               │  write SMD/QC + VTF/VMT, compile with the game's studiomdl.exe
               ▼
   models/vroid/*.mdl   materials/models/vroid/...
   scripts/vscripts/mapspawn.nut  (game's own copy + loads vroid_main.nut)
   scripts/vscripts/vroid/*.nut   (avatar switching, camera, state)
   resource/ui/basemodui/*.res    (game's own pause menu + one extra entry)
   scripts/kb_act.lst             (game's own list + VRoid actions)
   cfg/vroid/*.cfg                (console aliases, key binds)
```

* **Animation retargeting.** The model `$includemodel`s Chell's player model, so it uses all of
  her animations, pose parameters and attachments (the portal gun attaches as usual). Each
  output bone has the same rest *orientation* as Chell's, so her rotations look right on the
  avatar. Source animations also store bone *positions*, which would force Chell's limb
  lengths. An autoplay `predelta` sequence (the "proportion trick") restores the avatar's
  own joint positions on every frame.
* **Height.** The avatar is scaled uniformly so its full height in the rest pose matches the
  height of Chell's mesh (read from `player.vvd`).
* **No Valve files are shipped.** Menus, `mapspawn.nut` and `kb_act.lst` are patched from your
  own copies when you install.

## Development

```bash
pip install -r requirements.txt pytest
python -m pytest tests            # synthetic VRM + Chell skeleton, no game needed
python -m portal2vroid            # GUI (needs tkinter)
python -m portal2vroid --help     # CLI: install / add / remove / list / default / convert
python -m portal2vroid convert avatar.vrm --chell-mdl player.mdl --chell-vvd player.vvd --out out/
```

`packaging/build_exe.bat` builds the `.exe` locally. GitHub Actions
(`.github/workflows/build.yml`) builds it on every push and attaches it to releases for `v*`
tags.

| Module | Purpose |
|---|---|
| `vrm.py` | GLB / VRM 0.x & 1.0 loader (humanoid, spring bones, skinned meshes, materials) |
| `mdl.py`, `vpk.py`, `gamefs.py` | Read Chell's skeleton and original files from the game |
| `bonemap.py`, `retarget.py` | Bone mapping, reposing, scaling, output skeleton & weights |
| `source_export.py` | SMD + QC (proportion trick, IK chains, jigglebones) |
| `vtf.py`, `materials.py` | DXT1/DXT5 VTF writer, VMT generation |
| `menu_patch.py`, `keyvalues.py` | Pause-menu / keyboard-list patching |
| `installer.py`, `compiler.py` | Mod folder management, studiomdl |
| `game_files/` | VScript runtime + console aliases copied into the game |
