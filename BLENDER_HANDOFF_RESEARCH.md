# 3D Pose → Blender Animation — Research Plan & Proof of Concept

**Question.** Can we interpolate the 3D representation the pipeline extracts into an
accurate Blender animation, using a unisex body with limbs scaled to measured metrics?

**Answer: Yes — proven end-to-end.** A standalone script
([`Scripts/blender_mocap.py`](Scripts/blender_mocap.py)) loads the pipeline's canonical
mocap JSON, converts it to metric Blender-world coordinates, scales the skeleton with a
unisex anthropometric template, drives a constraint rig, and renders a golfer swing —
all headless via Blender 5.1. Rendered output: [`Data/demo/blender/`](Data/demo/blender/).

---

## What the pipeline actually produces (the grounding)

| Property | Value | Source |
|---|---|---|
| Canonical file | `Data/handoff/<stem>/<stem>_mocap.json` (1,402 clips on disk) | `export_ue5.export_json` |
| Payload | `format`, `skeleton{joint_names,joint_parents,axis_convention}`, `fps`, `frames[].positions_3d` | — |
| Array shape | `(T, 17, 3)` float | verified on clip `0` (138 frames) |
| Skeleton | **Human3.6M 17-joint**, fixed order + parent table | `export_ue5.H36M17_NAMES/PARENTS` |
| Coordinates | **normalized `h36m-camera`**: x=right, y=**down**, z=forward | `golfpose_adapter._normalize_screen` |
| Root | `hip_center` ≈ origin (approximately root-relative) | measured: `[-0.004, 0.0005, 0.0004]` |
| Units | normalized (~1.15 vertical units ≈ full stature); **NOT metric** | measured span |
| FPS | in JSON (`29.978` for clip 0) | `export_json` |
| Bone stabilization | `smoothing.enforce_bone_lengths` (median per bone) | already in pipeline |
| Prior DCC work | position-channel BVH exporter exists; **UE5 handoff retired** | `export_ue5` docstring |

Key consequence: the data is a **clean, hierarchical, length-stable 17-joint skeleton** —
ideal for driving an armature — but lives in a **normalized, monocular-depth camera frame**,
so absolute limb lengths are distorted and must be handled deliberately (below).

---

## Research iterations

**Iteration 1 — Framing & prior art.** Two viable routes: (A) *direct armature* — build
a 17-joint rig, drive it from joint positions (license-free, runs today); (B) *parametric
body* — fit **SMPL/SMPL-X** to the keypoints and import the mesh via the Meshcapade Blender
addon. Insight: "**unisex body + limbs scaled to metrics**" is literally the SMPL **neutral**
model whose **shape β** controls limb length — but the same effect is achievable
license-free with an anthropometric proportion template. → Pursue A for the POC, document B.

**Iteration 2 — Data format (gap closed by codebase scan).** Nailed the JSON schema, joint
convention, coordinate frame, fps, and the existing `enforce_bone_lengths`/BVH tooling
(above). *Remaining gap:* no absolute scale/units.

**Iteration 3 — Coordinate + scale + retargeting (gap closed by math + validation).**
- Axis map `(x,y,z)_cam → (x, z, −y)_blender` (Z-up, right-handed, det=+1, no mirror flip).
- Metric scale = `target_stature / vertical_joint_span`.
- **Finding, validated on real data:** raw *measured* limb lengths scale to
  thigh≈0.57 m / shank≈0.53 m for a 1.78 m subject — anatomically too long, because
  monocular depth ambiguity inflates segment lengths. The **unisex proportion template**
  yields correct thigh≈0.44 m / shank≈0.44 m while preserving the captured **joint angles**.
  → *Unisex mode is the recommended accurate default; measured mode is exact to the capture
  but inherits depth distortion.*
- Retargetability: a per-bone global-rotation solve reproduces joint positions with
  **0.0000 mm** FK round-trip error → the motion is a lossless, rig-agnostic representation.

**Iteration 4 — Blender build (gap closed by running it).** Resolved Blender-5.1 specifics:
slotted-Action API (no `action.fcurves` — set Bezier via preference), `STRETCH_TO` scales
an object's **local Y** (reorient bone-cylinder mesh 0→1 along +Y), camera `TRACK_TO` the
hip, render proxies parented to keyframed targets (armatures/empties don't render). Built,
saved `.blend`, and rendered three swing phases — all coherent golfer poses.

**Iteration 5 — Remaining gaps → none blocking.** The POC satisfies accurate + unisex +
limb-scaled. Documented upgrades (below) are quality improvements, not blockers.

---

## The clear path (pipeline → Blender)

```
mocap JSON (T,17,3, h36m-camera, normalized)
  └─ load_mocap_json
  └─ camera_to_blender        (x,z,−y) → Z-up metres-ready
  └─ × scale                  target_stature / vertical_span   →  metres
  └─ bone lengths             measured (subject) │ unisex template × stature
  └─ rigidify                 re-project joints to exact bone lengths, keep directions
  └─ [retarget] solve_global_bone_rotations → reconstruct (verified 0 mm)
  └─ build_in_blender:
        • Empty target per joint, position-keyframed, Bezier-interpolated (the interpolation)
        • 17-joint armature (rest = template), STRETCH_TO each joint target  ← the real rig
        • render proxies (spheres + bones) parented to targets
        • grey world + sun + floor + TRACK_TO camera
        • save .blend  +  render swing-phase stills
```

## Running it

```bash
# Numpy-only validation (no Blender needed): prints scale, bone lengths, FK error
python Scripts/blender_mocap.py --input Data/handoff/0/0_mocap.json --mode unisex --validate

# Full build + render (headless). --mode measured|unisex, --height <metres>
"<blender>" --background --python Scripts/blender_mocap.py -- \
    --input Data/handoff/0/0_mocap.json --height 1.78 --mode unisex \
    --blend Data/demo/blender/golfer_0.blend --render Data/demo/blender/golfer_0_preview.png
```

Outputs: `golfer_0.blend` (armature + constraint rig + baked keyframes) and swing-phase
PNGs. `_f020` (takeaway), `_f069` (mid), `_f117` (follow-through) confirm correct motion.

## Documented upgrade paths (next steps, not blockers)

1. **SMPL-X full mesh (gold standard for "unisex body").** Fit SMPL neutral to the H36M-17
   keypoints (SMPLify-3D: optimize shape β + pose θ to the 3D joints; β *is* the limb-length
   control). Export `.npz`, import via the Meshcapade `SMPL_blender_addon`. Gives a
   photoreal, skinned, gender-neutral body. Cost: free academic model license + a joint
   regressor to map H36M-17 → SMPL joints.
2. **Retargetable rotation rig / BVH.** We already solve lossless per-bone rotations; emit
   rotation-channel BVH or FBX to retarget onto Rigify / Auto-Rig Pro / a game character.
3. **Skinned proxy without SMPL.** Auto-weight a low-poly unisex mesh to the 17-joint
   armature (Rigify humanoid metarig) for a solid body instead of ball-and-stick.
4. **Temporal polish.** Feed `--smooth` (One-Euro / Savitzky-Golay, already in `smoothing.py`)
   before export, and resample keyframes to the render fps.
5. **Live MCP loop.** When the Blender MCP addon is running (localhost:9876), the same
   `build_in_blender` logic can drive the open session interactively instead of headless.
