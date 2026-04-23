# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a gesture retargeting pipeline for ROB803 that maps human pose data from video onto the Unitree G1 humanoid robot in MuJoCo. The specific gesture is the 比爱心 (heart-with-hands) gesture.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the G1 robot model (required before running the pipeline):
```bash
python download_g1.py
```

Place your input video at `input/gesture_video.mp4` (see `docs/recording_protocol.md` for capture requirements — 30 FPS, landscape, full-body visible).

## Running the Pipeline

Full pipeline (video → robot animation + evaluation):
```bash
python run_pipeline.py input/gesture_video.mp4
```

Run individual stages standalone:
```bash
python extract_pose.py input/gesture_video.mp4   # Step 1: pose extraction
python retarget.py                                # Step 4: uses output/pose_data.json
python evaluate.py                               # Step 5: uses output/ JSON files
```

## Architecture

The pipeline has 5 sequential steps, all orchestrated by `run_pipeline.py`:

1. **`extract_pose.py`** — Uses MediaPipe Pose to extract 33 body landmarks per frame from the input video. Falls back to synthetic sinusoidal data if MediaPipe is unavailable. Outputs `output/pose_data.json`.

2. **`smooth.py`** — Two-pass filtering: first `interpolate_missing_frames()` fills gaps ≤10 frames via linear interpolation, then `butterworth_smooth()` applies a zero-phase 2nd-order Butterworth low-pass filter (default 6 Hz cutoff). Operates on `world_landmarks` (hip-centered metric 3D coords).

3. **`visualize_skeleton.py`** — Draws MediaPipe skeleton overlays on the original video and renders a 3D skeleton animation. Outputs `output/skeleton_overlay.mp4`.

4. **`retarget.py`** — Core retargeting logic using direct joint-angle mapping:
   - Builds a torso-local coordinate frame (x=forward, y=left, z=up) from MediaPipe shoulder/hip landmarks
   - Decomposes limb direction vectors into sequential Euler angles matching the G1 kinematic chain (pitch→roll→yaw): `pitch = atan2(-d_x, -d_z)`, `roll = arcsin(d_y)` where d is the unit limb direction in torso-local frame
   - Computes hip pitch/roll, knee flexion, shoulder pitch/roll, elbow flexion, and waist yaw/roll/pitch for all 29 actuated joints
   - Clamps computed angles to MJCF joint limits, applies them to `qpos[7:]`, and renders via MuJoCo's offscreen renderer
   - Outputs `output/robot_gesture.mp4` and `output/joint_angles.json`

5. **`evaluate.py`** — Computes detection rate, per-landmark visibility and jitter, IK residual errors (mm), and joint smoothness. Generates 5 matplotlib figures in `output/figures/` and assembles a 3-panel side-by-side comparison video (`output/side_by_side.mp4`).

## Key Configuration (`config.py`)

All shared constants live here — paths, MediaPipe landmark indices, G1 joint name ordering, DOF index groups, IK parameters, and smoothing parameters. Change IK behavior (damping, iterations, step size) here rather than in individual files.

The G1 model uses `qpos[0:7]` for the free joint (position + quaternion) and `qpos[7:]` for the 29 actuated joints. The `ACTIVE_DOF_INDICES = np.arange(12, 29)` restricts IK to waist + arms only.

## G1 Model

- Located in `g1_model/`: `g1_29dof.xml` (robot definition), `scene_ik.xml` (scene with floor and lighting)
- End-effector body names: `left_wrist_yaw_link` and `right_wrist_yaw_link`
