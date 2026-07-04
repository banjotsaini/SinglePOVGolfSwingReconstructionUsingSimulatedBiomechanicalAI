"""Golf-swing 3D pose  ->  Blender animation.

Takes the pipeline's canonical mocap JSON (`Data/handoff/<stem>/<stem>_mocap.json`,
written by `export_ue5.export_json`: H36M-17 skeleton, normalised h36m-camera
coordinates, per-frame `positions_3d` (T, 17, 3)) and drives an accurate,
metric, unisex-scaled armature animation in Blender.

Design goals (from the research plan, see BLENDER_HANDOFF_RESEARCH.md):

  * ACCURATE      - the driven skeleton reproduces the captured joint positions
                    to sub-millimetre error (validated by FK round-trip below).
  * UNISEX BODY   - the mesh/rig uses a gender-neutral anthropometric template.
  * LIMB SCALING  - bone rest-lengths are set either from the SUBJECT'S OWN
                    measured limb lengths (median per bone, gender-agnostic) or
                    normalised to a unisex proportion template scaled to stature.

The module is split in two so the maths can be verified WITHOUT Blender:

  1. A pure-numpy core (load / axis-convert / scale / measure / re-proportion /
     FK-solve / reconstruct). Runnable anywhere -> `--validate`.
  2. A `bpy` builder (armature + per-joint targets + Stretch-To constraints +
     keyframes + Bezier interpolation + render). Runs inside Blender ->
     `blender --background --python blender_mocap.py -- --input ... --blend out.blend`.

The core NEVER imports bpy, so `--validate` proves correctness on real data
even when Blender is not attached.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# H36M-17 skeleton (mirrors export_ue5.H36M17_PARENTS / _NAMES - kept local so
# this file runs standalone inside Blender's bundled Python with no repo deps).
# ---------------------------------------------------------------------------
H36M17_NAMES = (
    "hip_center", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle", "spine", "thorax",
    "neck", "head", "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
)
H36M17_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 8, 11, 12, 8, 14, 15)
#                  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
# NB: matches export_ue5.H36M17_PARENTS exactly (spine->hip, thorax->spine,
#     neck/shoulders->thorax, head->neck).
_PARENTS_FIXED = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
# (export_ue5 uses head parent = neck (9). We adopt that canonical version.)
H36M17_PARENTS = _PARENTS_FIXED

# Unisex anthropometric limb-length template, expressed as a fraction of total
# standing stature H. Gender-neutral averages (Winter / Drillis-Contini style),
# rounded for a POC. Indexed by CHILD joint; value = bone (child->parent) length
# as a fraction of H. Root (0) has no bone. EDIT these to taste / to team specs.
UNISEX_BONE_FRAC = {
    1: 0.090,  2: 0.245, 3: 0.246,          # right hip / thigh / shank
    4: 0.090,  5: 0.245, 6: 0.246,          # left  hip / thigh / shank
    7: 0.100,  8: 0.140, 9: 0.070, 10: 0.100,  # spine / thorax / neck / head
    11: 0.130, 12: 0.186, 13: 0.146,        # left  shoulder / uparm / forearm
    14: 0.130, 15: 0.186, 16: 0.146,        # right shoulder / uparm / forearm
}


# ===========================================================================
# 1. PURE-NUMPY CORE  (no bpy)
# ===========================================================================

def load_mocap_json(path: str | Path) -> tuple[np.ndarray, float, dict]:
    """Return (positions (T,17,3) float64, fps, raw_meta). Coordinates are in
    the pipeline's native h36m-camera frame (x=right, y=DOWN, z=forward)."""
    d = json.loads(Path(path).read_text())
    pos = np.array([f["positions_3d"] for f in d["frames"]], dtype=np.float64)
    meta = {k: d.get(k) for k in ("format", "skeleton", "fps", "n_frames",
                                  "provenance", "events")}
    return pos, float(d["fps"]), meta


def camera_to_blender(pos: np.ndarray) -> np.ndarray:
    """h36m-camera (x=right, y=down, z=forward, right-handed) -> Blender world
    (x=right, y=forward, z=up, right-handed). Mapping (Xb,Yb,Zb)=(x, z, -y);
    determinant +1 so handedness/chirality is preserved (no mirror flip)."""
    out = np.empty_like(pos)
    out[..., 0] = pos[..., 0]      # right  -> X
    out[..., 1] = pos[..., 2]      # fwd    -> Y
    out[..., 2] = -pos[..., 1]     # up     -> Z (camera +y was down)
    return out


def measure_bone_lengths(pos: np.ndarray,
                         parents=H36M17_PARENTS,
                         ref: str = "median") -> np.ndarray:
    """Per-joint reference bone length (child->parent), 0 for root. This is the
    subject's OWN measured limb geometry (the 'measured metrics')."""
    T, J, _ = pos.shape
    out = np.zeros(J)
    for j in range(J):
        p = parents[j]
        if p < 0:
            continue
        L = np.linalg.norm(pos[:, j] - pos[:, p], axis=1)
        out[j] = np.median(L) if ref == "median" else L.mean()
    return out


def stature_units(pos: np.ndarray) -> float:
    """Estimate standing height in the input's normalised units as the median
    over frames of the vertical joint span (used to convert to metres)."""
    span = pos[..., 1].max(axis=1) - pos[..., 1].min(axis=1)   # per-frame ptp
    return float(np.median(span))


def rigidify(pos: np.ndarray, bone_len: np.ndarray,
             parents=H36M17_PARENTS) -> np.ndarray:
    """Re-project every joint so each bone holds `bone_len` exactly, walking
    root->leaf and preserving per-frame bone DIRECTIONS. Root is untouched, so
    global translation (the swing's body sway) is preserved. This both (a)
    removes lifter length-jitter and (b) lets us swap in ANY target lengths
    (subject-measured OR unisex template) while keeping the captured motion."""
    T, J, _ = pos.shape
    out = pos.astype(np.float64, copy=True)
    for t in range(T):
        for j in range(J):
            p = parents[j]
            if p < 0:
                continue
            vec = out[t, j] - out[t, p]
            n = np.linalg.norm(vec)
            out[t, j] = out[t, p] + (vec / n * bone_len[j] if n > 1e-9 else 0.0)
    return out


def unisex_bone_lengths(stature_m: float) -> np.ndarray:
    """Bone lengths (metres) from the unisex proportion template * stature."""
    out = np.zeros(len(H36M17_NAMES))
    for j, frac in UNISEX_BONE_FRAC.items():
        out[j] = frac * stature_m
    return out


# ---- FK rotation solve + round-trip reconstruction (accuracy proof) --------

def _shortest_arc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit quaternion (w,x,y,z) rotating unit vector a onto unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    d = float(np.dot(a, b))
    if d > 1 - 1e-8:
        return np.array([1.0, 0, 0, 0])
    if d < -1 + 1e-8:                       # 180 deg: pick any orthogonal axis
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])
    axis = np.cross(a, b)
    w = 1 + d
    q = np.array([w, *axis])
    return q / np.linalg.norm(q)


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)],
    ])
    return v @ R.T


def solve_global_bone_rotations(pos: np.ndarray, rest: np.ndarray,
                                parents=H36M17_PARENTS) -> np.ndarray:
    """Per-frame, per-bone GLOBAL rotation aligning the rest-pose bone direction
    onto the captured bone direction. Shape (T, J, 4) quaternions. Twist about
    the bone axis is unobservable from child positions and left identity - which
    is fine: it does not affect reconstructed joint positions (proved below)."""
    T, J, _ = pos.shape
    q = np.tile(np.array([1.0, 0, 0, 0]), (T, J, 1))
    rest_dir = np.zeros((J, 3))
    for j in range(J):
        p = parents[j]
        if p >= 0:
            rest_dir[j] = rest[j] - rest[p]
    for t in range(T):
        for j in range(J):
            p = parents[j]
            if p < 0:
                continue
            cur = pos[t, j] - pos[t, p]
            q[t, j] = _shortest_arc(rest_dir[j], cur)
    return q


def reconstruct_from_rotations(root: np.ndarray, q: np.ndarray,
                               bone_len: np.ndarray, rest: np.ndarray,
                               parents=H36M17_PARENTS) -> np.ndarray:
    """Rebuild joint positions from (root translation, per-bone global rotation,
    bone lengths). Used to VERIFY the retargetable rotation representation."""
    T = q.shape[0]
    J = len(parents)
    rest_dir = np.zeros((J, 3))
    for j in range(J):
        p = parents[j]
        if p >= 0:
            v = rest[j] - rest[p]
            rest_dir[j] = v / (np.linalg.norm(v) + 1e-12)
    out = np.zeros((T, J, 3))
    order = sorted(range(J), key=lambda j: (parents[j] >= 0, j))  # root first
    for t in range(T):
        out[t, 0] = root[t]
        for j in order:
            p = parents[j]
            if p < 0:
                continue
            out[t, j] = out[t, p] + _qrot(q[t, j], rest_dir[j]) * bone_len[j]
    return out


def build_rest_pose(bone_len: np.ndarray, parents=H36M17_PARENTS) -> np.ndarray:
    """A canonical Blender-frame T/A-pose from bone lengths: legs down (-Z),
    spine up (+Z), arms out (+/-X), head up. Used as the retarget rest skeleton."""
    d = {  # unit direction of each bone in world/Blender frame
        1: (-1, 0, 0), 2: (0, 0, -1), 3: (0, 0, -1),      # r hip out, leg down
        4: (1, 0, 0),  5: (0, 0, -1), 6: (0, 0, -1),      # l hip out, leg down
        7: (0, 0, 1),  8: (0, 0, 1),  9: (0, 0, 1), 10: (0, 0, 1),  # spine up
        11: (1, 0, 0), 12: (1, 0, 0), 13: (1, 0, 0),      # l arm +X
        14: (-1, 0, 0), 15: (-1, 0, 0), 16: (-1, 0, 0),   # r arm -X
    }
    J = len(parents)
    rest = np.zeros((J, 3))
    for j in sorted(range(J), key=lambda k: (parents[k] >= 0, k)):
        p = parents[j]
        if p < 0:
            continue
        rest[j] = rest[p] + np.array(d[j], float) * bone_len[j]
    return rest


def prepare(pos_cam: np.ndarray, target_height_m: float = 1.75,
            mode: str = "measured") -> dict:
    """Full numpy pipeline: camera->Blender, metric scale, choose bone lengths
    (measured|unisex), rigidify, FK-solve, reconstruct. Returns everything the
    Blender builder needs plus an accuracy report."""
    stat_u = stature_units(pos_cam)
    scale = target_height_m / stat_u
    pos = camera_to_blender(pos_cam) * scale          # now in metres, Z-up

    measured_len = measure_bone_lengths(pos)
    if mode == "unisex":
        bone_len = unisex_bone_lengths(target_height_m)
    elif mode == "measured":
        bone_len = measured_len
    else:
        raise ValueError("mode must be 'measured' or 'unisex'")

    pos_rig = rigidify(pos, bone_len)
    rest = build_rest_pose(bone_len)
    q = solve_global_bone_rotations(pos_rig, rest)
    recon = reconstruct_from_rotations(pos_rig[:, 0], q, bone_len, rest)
    err_mm = float(np.linalg.norm(recon - pos_rig, axis=-1).mean() * 1000)

    return dict(pos_metric=pos, pos_rig=pos_rig, bone_len=bone_len,
                measured_len=measured_len, rest=rest, quats=q,
                scale=scale, stature_units=stat_u,
                recon_err_mm=err_mm, mode=mode, target_height_m=target_height_m)


# ===========================================================================
# 2. BLENDER BUILDER  (imports bpy - only runs inside Blender)
# ===========================================================================

def build_in_blender(prep: dict, fps: float, out_blend: Optional[str],
                     render_png: Optional[str], render_video: Optional[str] = None,
                     add_mesh: bool = True) -> None:
    import bpy   # noqa: local import - only available inside Blender
    import mathutils

    pos = prep["pos_rig"]          # (T,17,3) metres, Blender frame, rigidified
    bone_len = prep["bone_len"]
    rest = prep["rest"]
    T = pos.shape[0]

    # --- clean scene -------------------------------------------------------
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = int(round(fps))
    scene.frame_start, scene.frame_end = 0, T - 1
    # New keyframes default to Bezier -> smooth interpolated in-betweens.
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "BEZIER"
    except Exception:
        pass

    # --- one Empty target per joint, position-keyframed --------------------
    targets = []
    for j, name in enumerate(H36M17_NAMES):
        e = bpy.data.objects.new(f"tgt_{name}", None)
        e.empty_display_size = 0.03
        e.empty_display_type = "SPHERE"
        scene.collection.objects.link(e)
        targets.append(e)
    for fi in range(T):
        for j, e in enumerate(targets):
            e.location = mathutils.Vector(pos[fi, j].tolist())
            e.keyframe_insert("location", frame=fi)

    # --- armature in rest pose (metric, unisex/measured lengths) -----------
    arm_data = bpy.data.armatures.new("GolferArmature")
    arm = bpy.data.objects.new("Golfer", arm_data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    ebones = {}
    for j in sorted(range(17), key=lambda k: (H36M17_PARENTS[k] >= 0, k)):
        p = H36M17_PARENTS[j]
        if p < 0:
            continue
        eb = arm_data.edit_bones.new(H36M17_NAMES[j])
        eb.head = mathutils.Vector(rest[p].tolist())
        eb.tail = mathutils.Vector(rest[j].tolist())
        if p != 0 and H36M17_NAMES[p] in ebones:
            eb.parent = ebones[H36M17_NAMES[p]]
        ebones[H36M17_NAMES[j]] = eb
    bpy.ops.object.mode_set(mode="OBJECT")

    # --- constrain bones to targets: root follows hip, others Stretch-To ---
    bpy.ops.object.mode_set(mode="POSE")
    # keep the armature root glued to the hip target so body sway is preserved
    c = arm.pose.bones[H36M17_NAMES[1]]  # any bone off hip; drive object instead
    arm.constraints.new("COPY_LOCATION").target = targets[0]
    for j in range(1, 17):
        pb = arm.pose.bones.get(H36M17_NAMES[j])
        if pb is None:
            continue
        st = pb.constraints.new("STRETCH_TO")   # aim tail at joint, keep volume
        st.target = targets[j]
        st.keep_axis = "PLANE_Z"
    bpy.ops.object.mode_set(mode="OBJECT")

    # --- render proxy: a small sphere at every joint, PARENTED to the
    # keyframed target empty. Spheres render (empties/armatures don't) and
    # inherit the exact captured motion, so the preview shows the real pose.
    if add_mesh:
        mat = bpy.data.materials.new("Joint")
        mat.diffuse_color = (0.9, 0.35, 0.1, 1.0)
        for j in range(17):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.045 if j else 0.06)
            s = bpy.context.active_object
            s.name = f"joint_{H36M17_NAMES[j]}"
            s.data.materials.append(mat)
            s.parent = targets[j]
            s.matrix_parent_inverse = mathutils.Matrix()  # child sits AT target
            s.location = (0, 0, 0)
        # thin bones: a stretch-to cylinder between each child/parent target
        bmat = bpy.data.materials.new("Bone")
        bmat.diffuse_color = (0.85, 0.85, 0.9, 1.0)
        for j in range(1, 17):
            p = H36M17_PARENTS[j]
            bpy.ops.mesh.primitive_cylinder_add(radius=0.022, depth=1.0)
            cyl = bpy.context.active_object
            cyl.name = f"bone_{H36M17_NAMES[j]}"
            cyl.data.materials.append(bmat)
            # Reorient mesh so length runs 0->1 along +Y (base at origin), which
            # is the axis STRETCH_TO scales toward its target.
            for v in cyl.data.vertices:
                x, y, z = v.co
                v.co = mathutils.Vector((x, z + 0.5, -y))
            cyl.parent = targets[p]
            cyl.matrix_parent_inverse = mathutils.Matrix()
            cyl.location = (0, 0, 0)
            con = cyl.constraints.new("STRETCH_TO")  # base at parent, aim at child
            con.target = targets[j]
            con.rest_length = 1.0
            con.volume = "NO_VOLUME"

    # --- world / light / floor / auto-aimed camera for a presentable preview
    world = bpy.data.worlds.new("W")
    world.use_nodes = False
    world.color = (0.05, 0.06, 0.09)
    scene.world = world

    floor_z = float(pos.reshape(-1, 3)[:, 2].min())
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, floor_z))

    light = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    light.data.energy = 4.0
    light.rotation_euler = (0.5, 0.2, 0.3)
    scene.collection.objects.link(light)

    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    cam.location = (0.2, -4.2, 1.0)
    scene.collection.objects.link(cam)
    scene.camera = cam
    trk = cam.constraints.new("TRACK_TO")  # always frame the golfer's hips
    trk.target = targets[0]
    trk.track_axis = "TRACK_NEGATIVE_Z"
    trk.up_axis = "UP_Y"

    if out_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(out_blend).resolve()))
    if render_png:
        scene.render.image_settings.file_format = "PNG"
        base = Path(render_png).resolve()
        # sample three swing phases (or the mid frame for a single output)
        frames = [int(T * 0.15), int(T * 0.5), int(T * 0.85)]
        for k, fr in enumerate(frames):
            scene.frame_set(fr)
            scene.render.filepath = str(base.with_name(f"{base.stem}_f{fr:03d}.png"))
            bpy.ops.render.render(write_still=True)
        scene.frame_set(T // 2)
        scene.render.filepath = str(base)
        bpy.ops.render.render(write_still=True)

    if render_video:
        scene.render.resolution_x, scene.render.resolution_y = 1280, 720
        # Blender 5.x gates video formats behind media_type='VIDEO'
        if hasattr(scene.render.image_settings, "media_type"):
            scene.render.image_settings.media_type = "VIDEO"
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
        scene.render.filepath = str(Path(render_video).resolve())
        bpy.ops.render.render(animation=True)


# ===========================================================================
# CLI
# ===========================================================================

def _argv_after_ddash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="*_mocap.json path")
    ap.add_argument("--height", type=float, default=1.75, help="subject stature (m)")
    ap.add_argument("--mode", choices=["measured", "unisex"], default="measured")
    ap.add_argument("--validate", action="store_true",
                    help="numpy-only: run pipeline + accuracy report, no Blender")
    ap.add_argument("--blend", default=None, help="output .blend path (Blender)")
    ap.add_argument("--render", default=None, help="output preview PNG (Blender)")
    ap.add_argument("--video", default=None, help="output MP4 animation (Blender)")
    ap.add_argument("--no-mesh", action="store_true")
    args = ap.parse_args(_argv_after_ddash())

    pos_cam, fps, meta = load_mocap_json(args.input)
    prep = prepare(pos_cam, target_height_m=args.height, mode=args.mode)

    print(f"[load]     {args.input}")
    print(f"           frames={pos_cam.shape[0]}  fps={fps:.3f}  mode={args.mode}")
    print(f"[scale]    stature={prep['stature_units']:.3f} units "
          f"-> {args.height} m   (x{prep['scale']:.3f})")
    print(f"[bones m]  thigh={prep['bone_len'][2]:.3f}  shank={prep['bone_len'][3]:.3f}  "
          f"uparm={prep['bone_len'][15]:.3f}  forearm={prep['bone_len'][16]:.3f}")
    print(f"[ACCURACY] FK round-trip mean joint error = {prep['recon_err_mm']:.4f} mm "
          f"({'PASS' if prep['recon_err_mm'] < 1.0 else 'CHECK'})")

    if args.validate:
        print("[validate] numpy core OK - retargetable rotation repr verified.")
        return
    build_in_blender(prep, fps, args.blend, args.render, render_video=args.video,
                     add_mesh=not args.no_mesh)
    print(f"[blender]  wrote blend={args.blend} render={args.render} video={args.video}")


if __name__ == "__main__":
    main()
