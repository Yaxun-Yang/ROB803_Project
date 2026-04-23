"""Retarget human pose data onto the Unitree G1 robot in MuJoCo.

Uses direct joint-angle mapping from MediaPipe skeleton landmarks.
Each human joint angle is computed geometrically from limb direction vectors
in a torso-local coordinate frame, then mapped to the corresponding
G1 actuated joint with proper offsets for the G1's default pose geometry.
"""

import json
import os
from datetime import datetime

import cv2
import imageio
import mujoco
import numpy as np

from config import (
    CAMERA_AZIMUTH,
    CAMERA_DISTANCE,
    CAMERA_ELEVATION,
    CAMERA_LOOKAT,
    FPS,
    G1_JOINT_NAMES,
    G1_MODEL_PATH,
    G1_PELVIS_HEIGHT,
    MP_LEFT_ANKLE,
    MP_LEFT_ELBOW,
    MP_LEFT_HIP,
    MP_LEFT_KNEE,
    MP_LEFT_SHOULDER,
    MP_LEFT_WRIST,
    MP_RIGHT_ANKLE,
    MP_RIGHT_ELBOW,
    MP_RIGHT_HIP,
    MP_RIGHT_KNEE,
    MP_RIGHT_SHOULDER,
    MP_RIGHT_WRIST,
    OUTPUT_DIR,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)


# G1 skeleton connections for overlay visualization
G1_SKELETON_CONNECTIONS = [
    ("pelvis", "torso_link"),
    ("pelvis", "left_hip_yaw_link"),
    ("left_hip_yaw_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_yaw_link"),
    ("right_hip_yaw_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("torso_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
]

G1_SKELETON_BODIES = list({b for pair in G1_SKELETON_CONNECTIONS for b in pair})


def _project_bodies(model, data, renderer, cam, width, height):
    """Project G1 body 3D positions to 2D pixel coordinates."""
    scene = renderer._scene
    gl_cam = scene.camera[0]
    pos = np.array(gl_cam.pos)
    fwd = np.array(gl_cam.forward)
    up = np.array(gl_cam.up)
    right = np.cross(fwd, up)

    near = gl_cam.frustum_near
    fw = gl_cam.frustum_width
    fb = gl_cam.frustum_bottom
    ft = gl_cam.frustum_top

    if abs(fw) < 1e-10 or abs(ft - fb) < 1e-10:
        return {}

    projections = {}
    for name in G1_SKELETON_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        world_pos = data.xpos[bid]
        delta = world_pos - pos
        x = np.dot(delta, right)
        y = np.dot(delta, up)
        z = np.dot(delta, fwd)
        if z <= 0.01:
            continue
        x_near = x * near / z
        y_near = y * near / z
        px = int((x_near + fw) / (2 * fw) * width)
        py = int((ft - y_near) / (ft - fb) * height)
        if 0 <= px < width and 0 <= py < height:
            projections[name] = (px, py)
    return projections


def _draw_skeleton(img, projections):
    """Draw G1 skeleton overlay on a rendered frame."""
    for (a, b) in G1_SKELETON_CONNECTIONS:
        if a in projections and b in projections:
            cv2.line(img, projections[a], projections[b], (0, 255, 0), 2)
    for name, pt in projections.items():
        cv2.circle(img, pt, 4, (0, 255, 255), -1)


def _compute_default_flexions(model):
    """Compute default elbow/knee flexion angles from G1 zero pose.

    Returns (elbow_flexion, knee_flexion) in radians.
    """
    data = mujoco.MjData(model)
    data.qpos[0:3] = [0.0, 0.0, G1_PELVIS_HEIGHT]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    def body_pos(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[bid].copy()

    def vec_angle(a, b):
        c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return np.arccos(np.clip(c, -1.0, 1.0))

    # Elbow: angle between upper arm and forearm at zero pose
    ls = body_pos("left_shoulder_pitch_link")
    le = body_pos("left_elbow_link")
    lw = body_pos("left_wrist_yaw_link")
    elbow_flex = vec_angle(le - ls, lw - le)

    # Knee: angle between thigh and shin at zero pose
    lh = body_pos("left_hip_yaw_link")
    lk = body_pos("left_knee_link")
    la = body_pos("left_ankle_roll_link")
    knee_flex = vec_angle(lk - lh, la - lk)

    return elbow_flex, knee_flex


def _compute_shoulder_yaw(d_ua, d_fa):
    """Compute shoulder yaw (twist about upper arm axis).

    d_ua: unit upper arm direction in torso-local
    d_fa: forearm direction in torso-local
    """
    d_fa_n = d_fa / (np.linalg.norm(d_fa) + 1e-8)
    # Project forearm onto plane perpendicular to upper arm
    proj = d_fa_n - np.dot(d_fa_n, d_ua) * d_ua
    proj_norm = np.linalg.norm(proj)
    if proj_norm < 1e-6:
        return 0.0
    proj = proj / proj_norm

    # Reference: "zero yaw" direction — perpendicular to upper arm, in the
    # plane containing the upper arm and the torso z-axis (up)
    z_up = np.array([0.0, 0.0, 1.0])
    ref = z_up - np.dot(z_up, d_ua) * d_ua
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-6:
        # Upper arm parallel to z; use x-axis as fallback
        ref = np.array([1.0, 0.0, 0.0])
        ref = ref - np.dot(ref, d_ua) * d_ua
        ref_norm = np.linalg.norm(ref)
        if ref_norm < 1e-6:
            return 0.0
    ref = ref / ref_norm

    cos_yaw = np.dot(ref, proj)
    sin_yaw = np.dot(np.cross(ref, proj), d_ua)
    return np.arctan2(sin_yaw, cos_yaw)


def compute_joint_angles(world_landmarks, default_elbow_flex, default_knee_flex):
    """Compute G1 joint angles from MediaPipe world landmarks.

    Builds a torso-local coordinate frame (x=forward, y=left, z=up),
    decomposes limb directions into pitch/roll/yaw, and applies offsets
    to match G1's default pose geometry.
    """
    wl = {lm["id"]: lm for lm in world_landmarks}

    def get_pos(idx):
        return np.array([wl[idx]["x"], wl[idx]["y"], wl[idx]["z"]])

    left_shoulder = get_pos(MP_LEFT_SHOULDER)
    right_shoulder = get_pos(MP_RIGHT_SHOULDER)
    left_elbow = get_pos(MP_LEFT_ELBOW)
    right_elbow = get_pos(MP_RIGHT_ELBOW)
    left_wrist = get_pos(MP_LEFT_WRIST)
    right_wrist = get_pos(MP_RIGHT_WRIST)
    left_hip = get_pos(MP_LEFT_HIP)
    right_hip = get_pos(MP_RIGHT_HIP)
    left_knee = get_pos(MP_LEFT_KNEE)
    right_knee = get_pos(MP_RIGHT_KNEE)
    left_ankle = get_pos(MP_LEFT_ANKLE)
    right_ankle = get_pos(MP_RIGHT_ANKLE)

    # --- Build torso coordinate frame ---
    hip_mid = (left_hip + right_hip) / 2
    shoulder_mid = (left_shoulder + right_shoulder) / 2

    up = shoulder_mid - hip_mid
    up_norm = np.linalg.norm(up)
    up = up / up_norm if up_norm > 1e-6 else np.array([0.0, 1.0, 0.0])

    right = right_shoulder - left_shoulder
    right = right - np.dot(right, up) * up
    right_norm = np.linalg.norm(right)
    right = right / right_norm if right_norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    forward = np.cross(right, up)

    # Rotation: MP world → torso-local (x=forward, y=left, z=up)
    R = np.array([forward, -right, up])

    def to_local(v):
        return R @ v

    def safe_arcsin(x):
        return np.arcsin(np.clip(x, -1.0, 1.0))

    def flexion_angle(v_parent, v_child):
        c = np.dot(v_parent, v_child) / (
            np.linalg.norm(v_parent) * np.linalg.norm(v_child) + 1e-8)
        return np.arccos(np.clip(c, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Arms  (pitch → roll → yaw → elbow)
    # ------------------------------------------------------------------
    # Left arm
    l_ua = to_local(left_elbow - left_shoulder)
    l_fa = to_local(left_wrist - left_elbow)
    l_d = l_ua / (np.linalg.norm(l_ua) + 1e-8)

    l_sh_pitch = np.arctan2(-l_d[0], -l_d[2])
    l_sh_roll = safe_arcsin(l_d[1])
    l_sh_yaw = _compute_shoulder_yaw(l_d, l_fa)
    l_human_flex = flexion_angle(l_ua, l_fa)
    l_elbow = default_elbow_flex - l_human_flex
    # Attenuate yaw near straight arm (singularity when flexion ≈ 0)
    yaw_fade = np.radians(30)
    if l_human_flex < yaw_fade:
        l_sh_yaw *= l_human_flex / yaw_fade

    # Right arm
    r_ua = to_local(right_elbow - right_shoulder)
    r_fa = to_local(right_wrist - right_elbow)
    r_d = r_ua / (np.linalg.norm(r_ua) + 1e-8)

    r_sh_pitch = np.arctan2(-r_d[0], -r_d[2])
    r_sh_roll = safe_arcsin(r_d[1])
    r_sh_yaw = _compute_shoulder_yaw(r_d, r_fa)
    r_human_flex = flexion_angle(r_ua, r_fa)
    r_elbow = default_elbow_flex - r_human_flex
    if r_human_flex < yaw_fade:
        r_sh_yaw *= r_human_flex / yaw_fade

    # ------------------------------------------------------------------
    # Legs  (hip_pitch → hip_roll → hip_yaw → knee)
    # ------------------------------------------------------------------
    # Left leg
    l_th = to_local(left_knee - left_hip)
    l_sh_leg = to_local(left_ankle - left_knee)
    l_th_d = l_th / (np.linalg.norm(l_th) + 1e-8)

    l_hip_pitch = np.arctan2(-l_th_d[0], -l_th_d[2])
    l_hip_roll = safe_arcsin(l_th_d[1])
    l_knee_human = flexion_angle(l_th, l_sh_leg)
    l_knee = l_knee_human - default_knee_flex

    # Right leg
    r_th = to_local(right_knee - right_hip)
    r_sh_leg = to_local(right_ankle - right_knee)
    r_th_d = r_th / (np.linalg.norm(r_th) + 1e-8)

    r_hip_pitch = np.arctan2(-r_th_d[0], -r_th_d[2])
    r_hip_roll = safe_arcsin(r_th_d[1])
    r_knee_human = flexion_angle(r_th, r_sh_leg)
    r_knee = r_knee_human - default_knee_flex

    # ------------------------------------------------------------------
    # Waist  (yaw → roll → pitch)
    # ------------------------------------------------------------------
    torso_vec = shoulder_mid - hip_mid
    # MediaPipe Tasks API: y points DOWN, so -y is the vertical (up) component
    tv = -torso_vec[1]
    tv = max(tv, 0.01)
    w_pitch = np.arctan2(-torso_vec[2], tv)
    w_roll = np.arctan2(-torso_vec[0], tv)

    pelvis_r = right_hip - left_hip
    pelvis_r = pelvis_r / (np.linalg.norm(pelvis_r) + 1e-8)
    torso_r = right_shoulder - left_shoulder
    torso_r = torso_r / (np.linalg.norm(torso_r) + 1e-8)
    w_yaw = np.arctan2(
        np.dot(np.cross(pelvis_r, torso_r), up),
        np.dot(pelvis_r, torso_r),
    )

    # ------------------------------------------------------------------
    # Pack into 29-element array
    # ------------------------------------------------------------------
    q = np.zeros(29)

    # Left leg [0-5]
    q[0] = l_hip_pitch
    q[1] = l_hip_roll
    q[3] = l_knee

    # Right leg [6-11]
    q[6] = r_hip_pitch
    q[7] = r_hip_roll
    q[9] = r_knee

    # Waist [12-14]
    q[12] = w_yaw
    q[13] = w_roll
    q[14] = w_pitch

    # Left arm [15-21]
    q[15] = l_sh_pitch
    q[16] = l_sh_roll
    q[17] = l_sh_yaw
    q[18] = l_elbow

    # Right arm [22-28]
    q[22] = r_sh_pitch
    q[23] = r_sh_roll
    q[24] = r_sh_yaw
    q[25] = r_elbow

    return q


def clamp_to_joint_limits(model, joint_angles):
    """Clamp joint angles to the MuJoCo model's joint limits."""
    clamped = joint_angles.copy()
    for i, name in enumerate(G1_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0 and model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            clamped[i] = np.clip(clamped[i], lo, hi)
    return clamped


def _compute_wrist_targets(pose_data, model):
    """Compute target wrist positions in G1 world frame from human pose data.

    Maps human wrist positions to the robot's coordinate frame by scaling
    limb-relative vectors from human shoulder to wrist.
    """
    data = mujoco.MjData(model)
    data.qpos[0:3] = [0.0, 0.0, G1_PELVIS_HEIGHT]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    # G1 shoulder positions and arm length at zero pose
    ls_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                "left_shoulder_pitch_link")
    rs_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                "right_shoulder_pitch_link")
    le_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_elbow_link")
    lw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                "left_wrist_yaw_link")
    g1_ls = data.xpos[ls_bid].copy()
    g1_rs = data.xpos[rs_bid].copy()
    g1_le = data.xpos[le_bid].copy()
    g1_lw = data.xpos[lw_bid].copy()
    g1_arm_len = np.linalg.norm(g1_le - g1_ls) + np.linalg.norm(g1_lw - g1_le)

    frames = pose_data["frames"]
    targets = []  # list of (left_target, right_target) or None per frame

    for frame in frames:
        if not frame["detected"] or not frame["world_landmarks"]:
            targets.append(None)
            continue

        wl = {lm["id"]: lm for lm in frame["world_landmarks"]}

        def gp(idx):
            return np.array([wl[idx]["x"], wl[idx]["y"], wl[idx]["z"]])

        h_ls = gp(MP_LEFT_SHOULDER)
        h_rs = gp(MP_RIGHT_SHOULDER)
        h_le = gp(MP_LEFT_ELBOW)
        h_re = gp(MP_RIGHT_ELBOW)
        h_lw = gp(MP_LEFT_WRIST)
        h_rw = gp(MP_RIGHT_WRIST)
        h_lhip = gp(MP_LEFT_HIP)
        h_rhip = gp(MP_RIGHT_HIP)

        # Human arm lengths
        h_left_arm = np.linalg.norm(h_le - h_ls) + np.linalg.norm(h_lw - h_le)
        h_right_arm = np.linalg.norm(h_re - h_rs) + np.linalg.norm(h_rw - h_re)

        # Build torso-local frame (same as in compute_joint_angles)
        hip_mid = (h_lhip + h_rhip) / 2
        sh_mid = (h_ls + h_rs) / 2
        up = sh_mid - hip_mid
        up_n = np.linalg.norm(up)
        up = up / up_n if up_n > 1e-6 else np.array([0.0, -1.0, 0.0])
        right = h_rs - h_ls
        right = right - np.dot(right, up) * up
        right_n = np.linalg.norm(right)
        right = right / right_n if right_n > 1e-6 else np.array([1.0, 0.0, 0.0])
        fwd = np.cross(right, up)
        R = np.array([fwd, -right, up])  # torso-local: x=fwd, y=left, z=up

        # Human wrist relative to shoulder in torso-local frame, scaled to G1
        l_rel = R @ (h_lw - h_ls)
        r_rel = R @ (h_rw - h_rs)
        l_scale = g1_arm_len / max(h_left_arm, 0.01)
        r_scale = g1_arm_len / max(h_right_arm, 0.01)

        # Target in G1 world: shoulder_pos + scaled relative vector
        # G1 world has same axes as torso-local (x=fwd, y=left, z=up)
        l_target = g1_ls + l_rel * l_scale
        r_target = g1_rs + r_rel * r_scale

        targets.append((l_target, r_target))

    return targets


def _compute_pelvis_trajectory(pose_data):
    """Compute pelvis world-space trajectory and per-frame orientation.

    Position: maps human's on-screen hip position to MuJoCo world coordinates.
    Orientation: detects pelvis yaw from hip landmark vector in world coords,
    so the robot rotates when the human rotates.

    Returns (positions[n,3], quats[n,4]).
    """
    frames = pose_data["frames"]
    n = len(frames)
    vid_fps = pose_data["metadata"]["fps"]

    # --- Camera geometry ---
    az = np.radians(CAMERA_AZIMUTH)  # -135°
    cam_fwd = np.array([-np.sin(az), np.cos(az)])
    cam_right = np.array([cam_fwd[1], -cam_fwd[0]])

    # Base yaw: robot faces toward camera when pelvis yaw_delta = 0
    # Camera is at direction (sin(az), -cos(az)) from lookat
    cam_dir = np.array([np.sin(az), -np.cos(az)])
    base_yaw = np.arctan2(cam_dir[1], cam_dir[0])

    # --- Extract data from landmarks ---
    hip_x_raw = np.full(n, np.nan)
    hip_y_raw = np.full(n, np.nan)
    pelvis_yaw_raw = np.full(n, np.nan)

    for i, frame in enumerate(frames):
        if not frame["detected"]:
            continue

        # Normalized landmarks for screen position
        if frame["landmarks"]:
            nl = {lm["id"]: lm for lm in frame["landmarks"]}
            if MP_LEFT_HIP in nl and MP_RIGHT_HIP in nl:
                hip_x_raw[i] = (nl[MP_LEFT_HIP]["x"] + nl[MP_RIGHT_HIP]["x"]) / 2
                hip_y_raw[i] = (nl[MP_LEFT_HIP]["y"] + nl[MP_RIGHT_HIP]["y"]) / 2

        # World landmarks for pelvis orientation
        if frame["world_landmarks"]:
            wl = {lm["id"]: lm for lm in frame["world_landmarks"]}
            if MP_LEFT_HIP in wl and MP_RIGHT_HIP in wl:
                lh = np.array([wl[MP_LEFT_HIP]["x"], wl[MP_LEFT_HIP]["y"],
                               wl[MP_LEFT_HIP]["z"]])
                rh = np.array([wl[MP_RIGHT_HIP]["x"], wl[MP_RIGHT_HIP]["y"],
                               wl[MP_RIGHT_HIP]["z"]])
                # hip_vec points from left hip to right hip
                hip_vec = rh - lh
                # MediaPipe world: x=subject's left, y=down, z=toward camera
                # Pelvis facing in horizontal xz plane: rotate hip_vec 90° CW
                # facing = (hip_z, 0, -hip_x)
                # yaw = atan2(facing_x, facing_z) = atan2(hip_z, -hip_x)
                # When facing camera: hip_vec≈(-hx,0,0) → yaw=atan2(0,hx)=0 ✓
                facing_x = hip_vec[2]
                facing_z = -hip_vec[0]
                pelvis_yaw_raw[i] = np.arctan2(facing_x, facing_z)

    # --- Fill NaN gaps ---
    def fill_nan(arr):
        valid = ~np.isnan(arr)
        if not valid.any():
            arr[:] = 0.0
            return
        first = arr[valid][0]
        last_val = first
        for i in range(n):
            if np.isnan(arr[i]):
                arr[i] = last_val
            else:
                last_val = arr[i]
        for i in range(n):
            if arr[i] == first and i < np.argmax(valid):
                arr[i] = first

    fill_nan(hip_x_raw)
    fill_nan(hip_y_raw)
    fill_nan(pelvis_yaw_raw)

    # Unwrap yaw to avoid atan2 discontinuities, then smooth
    pelvis_yaw_raw = np.unwrap(pelvis_yaw_raw)

    if n > 10:
        from scipy.signal import butter, filtfilt
        nyq = vid_fps / 2.0
        cutoff_pos = min(2.0, nyq * 0.9)
        b_pos, a_pos = butter(2, cutoff_pos / nyq, btype="low")
        hip_x_raw = filtfilt(b_pos, a_pos, hip_x_raw)
        hip_y_raw = filtfilt(b_pos, a_pos, hip_y_raw)
        cutoff_yaw = min(2.0, nyq * 0.9)
        b_yaw, a_yaw = butter(2, cutoff_yaw / nyq, btype="low")
        pelvis_yaw_raw = filtfilt(b_yaw, a_yaw, pelvis_yaw_raw)

    # --- Map image position to world ---
    aspect = VIDEO_WIDTH / VIDEO_HEIGHT
    fovy_rad = np.radians(45.0)
    fovx_rad = 2.0 * np.arctan(aspect * np.tan(fovy_rad / 2.0))
    visible_width = 2.0 * CAMERA_DISTANCE * np.tan(fovx_rad / 2.0)
    lateral_scale = visible_width

    # Center so gesture region (frames 60-130) is at origin
    gesture_mask = np.zeros(n, dtype=bool)
    gesture_mask[min(60, n):min(130, n)] = True
    center_x = np.mean(hip_x_raw[gesture_mask]) if gesture_mask.any() else 0.5

    positions = np.zeros((n, 3))
    for i in range(n):
        lateral_offset = (hip_x_raw[i] - center_x) * lateral_scale
        positions[i, 0] = lateral_offset * cam_right[0]
        positions[i, 1] = lateral_offset * cam_right[1]
        positions[i, 2] = G1_PELVIS_HEIGHT

    # --- Per-frame pelvis quaternion: base_yaw + detected pelvis rotation ---
    quats = np.zeros((n, 4))
    for i in range(n):
        yaw = base_yaw + pelvis_yaw_raw[i]
        quats[i] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

    yaw_range = np.degrees(pelvis_yaw_raw.max() - pelvis_yaw_raw.min())
    print(f"  Pelvis trajectory: lateral range "
          f"{positions[:, 0].min():.2f}..{positions[:, 0].max():.2f}m (x), "
          f"{positions[:, 1].min():.2f}..{positions[:, 1].max():.2f}m (y), "
          f"yaw range {yaw_range:.1f}°")

    return positions, quats


def retarget_gesture(pose_data, output_video_path=None, output_joints_path=None,
                     fps=None):
    """Main retargeting pipeline: joint-angle mapping + IK wrist correction
    + physics simulation with pelvis tracking.
    """
    if output_video_path is None:
        output_video_path = os.path.join(OUTPUT_DIR, "robot_gesture.mp4")
    if output_joints_path is None:
        output_joints_path = os.path.join(OUTPUT_DIR, "joint_angles.json")
    if fps is None:
        fps = pose_data["metadata"]["fps"]

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Load MuJoCo model
    print("  Loading G1 model...")
    model = mujoco.MjModel.from_xml_path(G1_MODEL_PATH)
    data = mujoco.MjData(model)

    # Compute default pose offsets
    default_elbow_flex, default_knee_flex = _compute_default_flexions(model)
    print(f"  Default flexions: elbow={np.degrees(default_elbow_flex):.1f}° "
          f"knee={np.degrees(default_knee_flex):.1f}°")

    frames = pose_data["frames"]
    total_frames = len(frames)

    # ---- Pass 1: Compute raw joint angles ----
    print(f"  Computing joint angles for {total_frames} frames...")
    all_angles = np.zeros((total_frames, 29))
    detected_mask = np.zeros(total_frames, dtype=bool)

    for i, frame in enumerate(frames):
        if frame["detected"] and frame["world_landmarks"]:
            all_angles[i] = compute_joint_angles(
                frame["world_landmarks"], default_elbow_flex, default_knee_flex)
            detected_mask[i] = True

    # Fill undetected frames with nearest detected frame's angles
    last_detected = None
    for i in range(total_frames):
        if detected_mask[i]:
            last_detected = i
        elif last_detected is not None:
            all_angles[i] = all_angles[last_detected]

    # ---- Pass 2: Unwrap angular discontinuities ----
    unwrap_joints = [0, 6, 12, 15, 17, 22, 24]
    for j in unwrap_joints:
        all_angles[:, j] = np.unwrap(all_angles[:, j])

    # ---- Pass 3: Clamp → Smooth → Clamp ----
    for i in range(total_frames):
        all_angles[i] = clamp_to_joint_limits(model, all_angles[i])

    if total_frames > 10:
        from scipy.signal import butter, filtfilt
        nyquist = fps / 2.0
        cutoff = min(6.0, nyquist * 0.9)
        b, a = butter(2, cutoff / nyquist, btype="low")
        cutoff_yaw = min(3.0, nyquist * 0.9)
        b_yaw, a_yaw = butter(2, cutoff_yaw / nyquist, btype="low")
        yaw_joints = {17, 24}
        for j in range(29):
            if j in yaw_joints:
                all_angles[:, j] = filtfilt(b_yaw, a_yaw, all_angles[:, j])
            else:
                all_angles[:, j] = filtfilt(b, a, all_angles[:, j])

    for i in range(total_frames):
        all_angles[i] = clamp_to_joint_limits(model, all_angles[i])

    # ---- Pass 4: IK wrist correction ----
    print("  Computing wrist IK targets...")
    from ik_solver import solve_ik_multi
    wrist_targets = _compute_wrist_targets(pose_data, model)
    # Only modify arm joints (15-28), keep legs and waist frozen
    arm_only_dof = np.arange(15, 29)

    lw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                "left_wrist_yaw_link")
    rw_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                "right_wrist_yaw_link")

    print("  Refining wrist positions with IK...")
    for i in range(total_frames):
        if wrist_targets[i] is None:
            continue
        l_target, r_target = wrist_targets[i]

        # Set initial pose from direct mapping
        data.qpos[0:3] = [0.0, 0.0, G1_PELVIS_HEIGHT]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:36] = all_angles[i]
        mujoco.mj_forward(model, data)

        # Run IK to refine arm-only joints toward wrist targets
        targets = [
            ("left_wrist_yaw_link", l_target),
            ("right_wrist_yaw_link", r_target),
        ]
        solve_ik_multi(model, data, targets,
                       tol=0.005, damping=0.05, max_iter=80, step_size=0.4,
                       active_dof_indices=arm_only_dof)

        # Extract refined angles (only arm joints change, legs/waist preserved)
        all_angles[i] = data.qpos[7:36].copy()

    # Light smoothing on IK-refined arm joints only, then clamp
    if total_frames > 10:
        cutoff_ik = min(4.0, nyquist * 0.9)
        b_ik, a_ik = butter(2, cutoff_ik / nyquist, btype="low")
        for j in range(15, 29):
            all_angles[:, j] = filtfilt(b_ik, a_ik, all_angles[:, j])
    for i in range(total_frames):
        all_angles[i] = clamp_to_joint_limits(model, all_angles[i])

    # ---- Pass 5: Compute pelvis trajectory + Physics simulation + Render ----
    print("  Computing pelvis trajectory...")
    pelvis_positions, pelvis_quats = _compute_pelvis_trajectory(pose_data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = CAMERA_LOOKAT
    cam.distance = CAMERA_DISTANCE
    cam.azimuth = CAMERA_AZIMUTH
    cam.elevation = CAMERA_ELEVATION

    renderer = mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)

    joint_data = {
        "metadata": {
            "source_pose_data": pose_data["metadata"].get("source_video", ""),
            "robot_model": "g1_29dof",
            "retarget_method": "joint_angle_mapping_ik",
            "retarget_timestamp": datetime.now().isoformat(),
        },
        "joint_names": G1_JOINT_NAMES,
        "frames": [],
    }

    # PD controller gains — high stiffness for accurate tracking
    kp = np.full(29, 1000.0)
    kd = np.full(29, 50.0)

    # Initialize simulation
    dt = model.opt.timestep
    steps_per_frame = max(1, int(round(1.0 / (fps * dt))))

    data.qpos[0:3] = pelvis_positions[0]
    data.qpos[3:7] = pelvis_quats[0]
    data.qpos[7:36] = all_angles[0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    print(f"  Rendering {total_frames} frames (physics, {steps_per_frame} "
          f"substeps/frame, dt={dt})...")
    writer = imageio.get_writer(
        output_video_path, fps=fps, codec="libx264", quality=None,
        output_params=["-pix_fmt", "yuv420p", "-profile:v", "baseline",
                       "-level", "3.1", "-movflags", "+faststart"],
    )

    # Separate data for IK error measurement at origin pose
    ik_data = mujoco.MjData(model)

    for i in range(total_frames):
        target_q = all_angles[i]
        target_pos = pelvis_positions[i]
        target_quat = pelvis_quats[i]

        # PD control with pelvis following trajectory
        for _ in range(steps_per_frame):
            q_err = target_q - data.qpos[7:36]
            q_vel = data.qvel[6:35]
            data.ctrl[:] = np.clip(kp * q_err - kd * q_vel,
                                   model.actuator_ctrlrange[:, 0],
                                   model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(model, data)
            # Constrain pelvis to follow trajectory (position + orientation)
            # but let joint physics evolve naturally
            data.qpos[0:3] = target_pos
            data.qpos[3:7] = target_quat
            data.qvel[0:6] = 0

        # Record actual joint angles (from physics)
        actual_q = data.qpos[7:36].copy()

        # Compute IK error using FK at origin (where IK targets were computed)
        ik_err_l = ik_err_r = None
        if wrist_targets[i] is not None:
            l_tgt, r_tgt = wrist_targets[i]
            ik_data.qpos[0:3] = [0.0, 0.0, G1_PELVIS_HEIGHT]
            ik_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            ik_data.qpos[7:36] = actual_q
            mujoco.mj_forward(model, ik_data)
            ik_err_l = float(np.linalg.norm(l_tgt - ik_data.xpos[lw_bid]))
            ik_err_r = float(np.linalg.norm(r_tgt - ik_data.xpos[rw_bid]))

        joint_data["frames"].append({
            "frame_index": i,
            "timestamp_sec": frames[i]["timestamp_sec"],
            "joint_angles": [round(float(a), 6) for a in actual_q],
            "ik_error_left": round(ik_err_l, 6) if ik_err_l is not None else None,
            "ik_error_right": round(ik_err_r, 6) if ik_err_r is not None else None,
        })

        renderer.update_scene(data, camera=cam)
        frame_img = renderer.render().copy()
        projections = _project_bodies(model, data, renderer, cam,
                                      VIDEO_WIDTH, VIDEO_HEIGHT)
        _draw_skeleton(frame_img, projections)
        writer.append_data(frame_img)

        if (i + 1) % 30 == 0:
            print(f"  Frame {i + 1}/{total_frames}")

    writer.close()
    renderer.close()
    print(f"  Robot video saved to {output_video_path}")

    with open(output_joints_path, "w") as f:
        json.dump(joint_data, f, indent=2)
    print(f"  Joint angles saved to {output_joints_path}")

    return joint_data


if __name__ == "__main__":
    from extract_pose import load_pose_data
    from smooth import butterworth_smooth, interpolate_missing_frames

    pose_path = os.path.join(OUTPUT_DIR, "pose_data.json")
    pose_data = load_pose_data(pose_path)
    pose_data = interpolate_missing_frames(pose_data)
    pose_data = butterworth_smooth(pose_data)
    retarget_gesture(pose_data)
