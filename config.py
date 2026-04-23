"""Shared configuration for the gesture retargeting pipeline."""

import os
import numpy as np

# ---- Paths ----
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
G1_MODEL_PATH = os.path.join(PROJECT_DIR, "g1_model", "scene_ik.xml")
INPUT_DIR = os.path.join(PROJECT_DIR, "input")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# ---- Video ----
FPS = 30
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

# ---- MediaPipe Landmark Indices (33 total) ----
MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_PINKY = 17
MP_RIGHT_PINKY = 18
MP_LEFT_INDEX = 19
MP_RIGHT_INDEX = 20
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28

MP_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# ---- G1 Robot Body Names ----
G1_LEFT_HAND = "left_wrist_yaw_link"
G1_RIGHT_HAND = "right_wrist_yaw_link"
G1_LEFT_FOOT = "left_ankle_roll_link"
G1_RIGHT_FOOT = "right_ankle_roll_link"
G1_TORSO = "torso_link"
G1_PELVIS = "pelvis"

# ---- G1 Actuated Joint Names (qpos[7:] ordering) ----
G1_JOINT_NAMES = [
    # Left leg (indices 0-5)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right leg (indices 6-11)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (indices 12-14)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left arm (indices 15-21)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right arm (indices 22-28)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# ---- DOF Index Groups (indices into actuated joints, i.e. qpos[7+i]) ----
LEG_DOF_INDICES = np.arange(0, 12)       # both legs
WAIST_DOF_INDICES = np.arange(12, 15)    # waist yaw/roll/pitch
LEFT_ARM_DOF_INDICES = np.arange(15, 22) # left arm 7 DOF
RIGHT_ARM_DOF_INDICES = np.arange(22, 29) # right arm 7 DOF
ACTIVE_DOF_INDICES = np.arange(0, 29)    # all joints (legs + waist + arms)

# ---- G1 Reference Dimensions (meters, from MJCF) ----
G1_PELVIS_HEIGHT = 0.793
G1_LEFT_SHOULDER_OFFSET = np.array([0.0039563, 0.10022, 0.23778])
G1_RIGHT_SHOULDER_OFFSET = np.array([0.0039563, -0.10021, 0.23778])
G1_SHOULDER_WIDTH = 0.10022 + 0.10021  # ~0.200m total

# ---- IK Parameters ----
IK_DAMPING = 0.05
IK_MAX_ITER = 50
IK_STEP_SIZE = 0.5
IK_TOLERANCE = 1e-3

# ---- Smoothing ----
SMOOTHING_WINDOW = 5
BUTTERWORTH_CUTOFF = 6.0  # Hz
BUTTERWORTH_ORDER = 2

# ---- Confidence Threshold ----
MIN_VISIBILITY = 0.5

# ---- MuJoCo Camera ----
CAMERA_LOOKAT = [0.0, 0.0, 0.8]
CAMERA_DISTANCE = 2.5
CAMERA_AZIMUTH = -135
CAMERA_ELEVATION = -15
