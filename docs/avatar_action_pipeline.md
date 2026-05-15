# Avatar Action Pipeline

This pipeline turns one person/action sentence into a HA-VLN/ViCo-compatible
textured human asset:

1. Select the best matching ViCo/Mixamo avatar GLB from an avatar library.
2. Load an existing GEM/Kimodo motion output, or call `kimodo_gen`.
3. Retarget SMPL-22/Kimodo local rotations onto the selected avatar's own skin joints.
   When posed joints are available, legs use calibrated two-bone IK so the
   target avatar's own knee pole controls hip-knee-ankle bending.
4. Optionally stabilize stationary action yaw and lock support/landing feet after retargeting.
5. Export static HAPS frames while preserving the original avatar mesh, UVs, skin weights,
   base-color textures, and PBR materials.
6. Add a lightweight visible shell and Habitat-safe material flags so side/back
   cameras do not collapse thin clothing surfaces into paper-like cutouts.
7. Write `skeleton.json`, `persona.json`, `frameXXX.glb`, and `frameXXX.object_config.json`.

The key compatibility rule is: do not project a ViCo texture onto a different SMPL-X
topology unless you explicitly want that lossy transfer. The stable path is to keep
the original skinned avatar and drive its existing Mixamo/ViCo skeleton.

Avatar selection treats large transparent texture atlases as risky for Habitat.
A strong suit/formal match with a single alpha atlas is allowed only because the
exporter now solidifies the mesh and forces double-sided materials; other large
alpha avatars are heavily penalized to avoid hollow or paper-thin humans.

## Existing Kimodo Motion

```bash
PYTHONPATH=src python3 scripts/generate_avatar_action.py \
  "An office worker in a suit does a backflip and waves." \
  --output-root outputs/demo_action \
  --motion-npz local_experiments/kimodo_motion_backflip_wave/office_worker_backflip_then_wave_00.npz \
  --frames 120
```

## Generate With Kimodo

Run this in an environment where NVIDIA Kimodo is installed and `kimodo_gen` is on
`PATH`:

```bash
PYTHONPATH=src python3 scripts/generate_avatar_action.py \
  "An office worker in a suit walks forward, turns, and waves." \
  --output-root outputs/demo_action \
  --generator kimodo \
  --kimodo-model Kimodo-SMPLX-RP-v1 \
  --duration 5.0 \
  --frames 120
```

## Override Avatar Selection

```bash
PYTHONPATH=src python3 scripts/generate_avatar_action.py \
  "A police officer waves to the viewer." \
  --output-root outputs/police_wave \
  --motion-npz path/to/motion.npz \
  --avatar-glb vico_assets_probe/models/police.glb
```

## Retarget Tuning

The defaults enable calibrated two-bone leg IK when the source motion provides
`posed_joints` such as Kimodo `.npz` outputs. The solver measures the target
avatar bind-pose hip-knee-ankle chain, stores the target rig knee pole, then
rebuilds each leg from the source hip-to-ankle reach while keeping the knee on
the target rig's valid bend side.

For sources without `posed_joints`, such as the current GEM `smpl_params.pt`
loader path, the fallback deliberately damps lower-leg and foot rotations more
than the rest of the body. This avoids backward knee bending when SMPL/GEM leg
axes do not exactly match a ViCo/Mixamo avatar's local bone axes.

Use `--rotation-scale` for the whole body, then only raise
`--lower-leg-rotation-scale` or `--foot-rotation-scale` when the avatar rig has
been verified to bend cleanly.

Use `--no-calibrated-leg-ik` to disable the two-bone leg solver for debugging.

Keep retargeting low-intrusion by default. `--body-relative-leg-ik` and
`--airborne-leg-stabilization` are experimental opt-in tools for diagnosing a
bad motion source, not default production fixes. If an airborne acrobatic pose
folds the human into itself, prefer regenerating or selecting a cleaner
Kimodo/GEM motion source over forcing a canonical tuck in post-processing.

For stationary acrobatics such as standing backflips, add:

```bash
--stabilize-root-yaw --foot-contact-lock
```

`--stabilize-root-yaw` removes unwanted horizontal spin while preserving the
flip's pitch/roll. `--foot-contact-lock` now prefers grounded foot IK when
calibrated leg IK is available. It starts with strict geometric contacts
(low foot height plus low horizontal velocity), rejects degenerate Kimodo
`foot_contacts` labels, expands low-foot events into alternating gait support
windows, then pins each support foot with the target rig's two-bone leg IK
instead of translating the whole body.

Tune `--foot-contact-height`, `--foot-contact-velocity`,
`--foot-support-min-frames`, `--foot-support-max-frames`,
`--foot-support-max-air-frames`, and `--foot-lock-blend-frames` when a running
source touches down too early, stays planted too long, or spends too many frames
with both feet airborne. `--no-grounded-foot-ik` falls back to the older
whole-frame contact correction for debugging only.

Contact-foot orientation stabilization is enabled by default, aligning the toe
direction to the target rig's body-facing direction, compensating for root yaw
stabilization, aligning the foot-up axis to the floor normal, and neutralizing
toe-base curl. Use `--no-foot-orientation-lock` when isolating height/position
locking from ankle twist problems.

For running clips where generated arm motion looks robotic, damp the upper-arm,
forearm, and hand channels independently with `--arm-rotation-scale`,
`--forearm-rotation-scale`, and `--hand-rotation-scale`.

## Habitat Material And Thickness

By default each exported frame:

- converts non-hair materials to `OPAQUE`;
- converts hair/eyelash/alpha-card materials to `MASK`;
- sets every material `doubleSided=true`;
- adds a symmetric mesh shell (`--body-shell-thickness`, default `0.018m`) so
  side cameras see body volume instead of a single surface.

Use `--no-solidify-shell` only for debugging source mesh deformation. In HA-VLN
recordings, leaving the shell enabled is the safer default.
