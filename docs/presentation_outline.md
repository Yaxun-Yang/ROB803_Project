# Slides Outline: 比爱心 Gesture Retargeting Pipeline

**Total: ~12 slides | 5 minutes**

---

## Slide 0 — Title (15s)
- ROB803 Course Project
- Presented by Yaxun Yang, 25011222

---

## Slide 0.5 — Task Introduction (30s)
- Task: map a human gesture video onto a humanoid robot (Unitree G1) in MuJoCo simulation
- Chosen gesture: 比爱心 ("make a heart shape with hands")
  - A popular East Asian hand gesture where both hands are raised to chest level and fingers are interlocked to form a heart shape
  - Chosen because it is expressive, asymmetric in depth, and exercises shoulder/elbow DOFs fully

---

## Slide 1 — Pipeline Overview (30s)
- One-sentence hook: "Video in → robot animation out, 5 stages"
- Diagram: `Video → Pose Extraction → Smoothing → Retargeting → IK Refinement → Physics Render`

---

## Slide 2 — Pose Extraction: Why MediaPipe (30s)
- MediaPipe Pose: 33 landmarks, two coordinate spaces
  - `landmarks` (normalized image): used for pelvis screen-position tracking
  - `world_landmarks` (hip-centered metric 3D): used for all angle computation
- Key tradeoff: no depth sensor → z-axis ambiguity; world_landmarks partially mitigate this via body-proportional normalization

---

## Slide 3 — Smoothing: Two-Pass Design (30s)
- **Pass 1** — Gap fill: linear interpolation for gaps ≤10 frames; longer gaps warned and skipped
- **Pass 2** — Zero-phase Butterworth low-pass (6 Hz cutoff, 2nd order `filtfilt`)
  - Why `filtfilt`: no phase lag (no temporal shift in gesture timing)
  - Why 6 Hz: preserves expressive arm motion, removes >6 Hz noise from monocular jitter
- Show raw vs smoothed jitter numbers: wrist ~45 mm/frame before smoothing

---

## Slide 4 — Retargeting: Direct Angle Mapping (45s)
- Build torso-local frame from shoulder/hip midpoints: `x=forward, y=left, z=up`
- Project limb direction vectors into this frame, decompose into sequential Euler angles:
  - Shoulder: `pitch = atan2(-d_x, -d_z)`, `roll = arcsin(d_y)`, then yaw (forearm projection)
  - Hip, knee, waist: same pattern
- Why not full IK from scratch: direct mapping is deterministic, interpretable, and fast; avoids IK singularities in global solve
- Singularity handling: yaw faded to zero when arm near-straight (elbow flex < 30°)

---

## Slide 5 — IK Wrist Refinement (30s)
- Problem: direct angle mapping accumulates error along the 7-DOF chain → wrist can be off
- Solution: after angle mapping, run damped least-squares IK (`damping=0.05`, 80 iter) on arm DOFs only (joints 15–28), with legs/waist frozen
- Human→robot wrist target: scale relative shoulder-to-wrist vector by G1 arm length ratio
- Light Butterworth re-smooth on IK-refined arm joints (4 Hz) to remove solver jitter

---

## Slide 6 — Physics Simulation & Rendering (30s)
- PD controller drives joints to target angles (`kp=1000`, `kd=50`), high stiffness for tracking fidelity
- Pelvis trajectory: extracted from normalized hip screen-position, mapped to world coords via camera geometry; per-frame yaw from hip-vector orientation
- MuJoCo offscreen renderer + skeleton overlay projected via manual frustum math (no extra deps)

---

## Slide 7 — Results Visualization (45s)
- Show `output/side_by_side.mp4`: Original | MediaPipe skeleton | G1 robot
- Key metrics from `evaluation_report.json`:
  - Detection rate: **94.3%** (198/210 frames)
  - Avg landmark visibility: **0.78**
  - Left shoulder/hip: near-perfect (~0.999); wrists: **0.52–0.73** (gesture hand occlusion)

---

## Slide 8 — Evaluation Figures (30s)
- Show `confidence_over_time.png`: detection dip when hands cross body center
- Show `joint_smoothness.png` heatmap: shoulder_yaw is the most active/variable joint (mean Δ=0.10 rad/frame) — correctly reflects the heart-shape arm rotation
- Show `ik_error.png`: left wrist mean error **79 mm**, right **58 mm** — acceptable for expressive gesture at this scale

---

## Slide 9 — Failure Cases & Analysis (60s)

Show video clips from each `output_failure*/` directory alongside the description.

**Failure 1 — Mixed arm/leg joints** (`output_failure1_mixArmLegs`)
- Bug: G1 joint index array was wrong — computed arm angles were written into leg DOF slots and vice versa
- Symptom: robot flails legs when arms should be moving, arms hang limp
- Fix: carefully verified `G1_JOINT_NAMES` ordering against MJCF and mapped each DOF index explicitly

**Failure 2 — Position control instead of joint control** (`output_failure2_positionNotjoint`)
- Bug: early version set `data.xpos` (body Cartesian positions) directly instead of `data.qpos` (joint angles)
- Symptom: robot snaps to impossible poses each frame, physically invalid configurations
- Fix: switched to proper joint-angle pipeline; Cartesian targets only used as IK goals, not applied directly

**Failure 3 — Axis misalignment** (`output_failure3_notalighnaxis`)
- Bug: MediaPipe world frame (y=down) was fed directly to G1 (z=up) without the torso-local rotation `R`
- Symptom: robot bends sideways or inverts when human raises arms; gesture direction is wrong
- Fix: built explicit torso-local rotation matrix `R = [forward, -right, up]` to unify coordinate conventions

**Failure 4 — PD controller instability** (`output_failure4_PD`)
- Bug: initial gains too low (`kp=10`); robot couldn't track fast arm motions
- Symptom: robot arms lag severely, oscillate, and never reach target poses within a frame
- Fix: raised gains to `kp=1000, kd=50` (high stiffness appropriate for kinematic replay, not balance control)

**Failure 5 — Fixed pelvis** (`output_failure5_fixedPelvis`)
- Bug: pelvis was pinned at origin with identity quaternion; no trajectory tracking
- Symptom: robot always stands at scene center facing a fixed direction regardless of human position/rotation
- Fix: added `_compute_pelvis_trajectory()` — extracts hip screen position and hip-vector yaw, maps to world frame per frame

**Failure 6 — No torso orientation** (`output_failure6_noOrientation`)
- Bug: pelvis position was tracked but yaw was not; robot always faced the same direction
- Symptom: when human turns slightly, arm angles look correct in human frame but appear mirrored on robot
- Fix: computed pelvis yaw from `atan2(hip_vec_z, -hip_vec_x)` and applied as per-frame quaternion

**Failure 7 — Low wrist accuracy before IK** (`output_failure7_notAccuricy`)
- Bug: direct angle mapping alone; no IK refinement pass
- Symptom: wrists visibly off-target — heart shape is recognizable but hands don't meet at the correct position
- Fix: added IK wrist correction pass (Slide 5); reduced mean wrist error from ~150 mm to ~79/58 mm

---

## Slide 10 — Summary & What Would Fix It (30s)
- What worked: torso-local frame + direct Euler mapping gives stable, expressive results; iterative failure-driven debugging was essential
- What limited quality: monocular depth ambiguity for self-occluding gestures (left wrist worst at 0.52 visibility)
- Next step: replace MediaPipe `world_landmarks` z with a depth camera (e.g. RealSense) or a learned depth-from-video model to improve wrist IK target accuracy

---

**Timing guide:** Slides 0–1 (~1 min) → Slides 2–6 (~2 min) → Slides 7–8 (~1 min) → Slides 9–10 (~1 min)
