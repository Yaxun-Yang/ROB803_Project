"""Download Unitree G1 29-DOF MuJoCo model from the official repository."""

import os
import time
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/unitreerobotics/unitree_mujoco/main/unitree_robots/g1"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "g1_model")
MESH_DIR = os.path.join(MODEL_DIR, "meshes")

XML_FILES = [
    "g1_29dof.xml",
    "scene_29dof.xml",
]

MESH_FILES = [
    "head_link.STL",
    "left_ankle_pitch_link.STL",
    "left_ankle_roll_link.STL",
    "left_elbow_link.STL",
    "left_hand_index_0_link.STL",
    "left_hand_index_1_link.STL",
    "left_hand_middle_0_link.STL",
    "left_hand_middle_1_link.STL",
    "left_hand_palm_link.STL",
    "left_hand_thumb_0_link.STL",
    "left_hand_thumb_1_link.STL",
    "left_hand_thumb_2_link.STL",
    "left_hip_pitch_link.STL",
    "left_hip_roll_link.STL",
    "left_hip_yaw_link.STL",
    "left_knee_link.STL",
    "left_rubber_hand.STL",
    "left_shoulder_pitch_link.STL",
    "left_shoulder_roll_link.STL",
    "left_shoulder_yaw_link.STL",
    "left_wrist_pitch_link.STL",
    "left_wrist_roll_link.STL",
    "left_wrist_roll_rubber_hand.STL",
    "left_wrist_yaw_link.STL",
    "logo_link.STL",
    "pelvis.STL",
    "pelvis_contour_link.STL",
    "right_ankle_pitch_link.STL",
    "right_ankle_roll_link.STL",
    "right_elbow_link.STL",
    "right_hand_index_0_link.STL",
    "right_hand_index_1_link.STL",
    "right_hand_middle_0_link.STL",
    "right_hand_middle_1_link.STL",
    "right_hand_palm_link.STL",
    "right_hand_thumb_0_link.STL",
    "right_hand_thumb_1_link.STL",
    "right_hand_thumb_2_link.STL",
    "right_hip_pitch_link.STL",
    "right_hip_roll_link.STL",
    "right_hip_yaw_link.STL",
    "right_knee_link.STL",
    "right_rubber_hand.STL",
    "right_shoulder_pitch_link.STL",
    "right_shoulder_roll_link.STL",
    "right_shoulder_yaw_link.STL",
    "right_wrist_pitch_link.STL",
    "right_wrist_roll_link.STL",
    "right_wrist_roll_rubber_hand.STL",
    "right_wrist_yaw_link.STL",
    "torso_constraint_L_link.STL",
    "torso_constraint_L_rod_link.STL",
    "torso_constraint_R_link.STL",
    "torso_constraint_R_rod_link.STL",
    "torso_link.STL",
    "waist_constraint_L.STL",
    "waist_constraint_R.STL",
    "waist_roll_link.STL",
    "waist_support_link.STL",
    "waist_yaw_link.STL",
]

SCENE_IK_XML = """\
<mujoco model="g1_29dof_ik">
  <include file="g1_29dof.xml"/>

  <statistic center="0 0 0.7" extent="2.0"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
      width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""


def download_file(url, dest, retries=3):
    """Download a file with retry logic. Skips if already exists."""
    if os.path.exists(dest):
        return
    for attempt in range(retries):
        try:
            print(f"  Downloading {os.path.basename(dest)}...")
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:
            if attempt < retries - 1:
                print(f"    Retry {attempt + 1}/{retries}...")
                time.sleep(1)
            else:
                raise RuntimeError(
                    f"Failed to download {os.path.basename(dest)}: {e}"
                ) from e


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(MESH_DIR, exist_ok=True)

    print("Downloading G1 XML model files...")
    for f in XML_FILES:
        download_file(f"{BASE_URL}/{f}", os.path.join(MODEL_DIR, f))

    # Write scene_ik.xml (custom scene with IK target markers)
    scene_ik_path = os.path.join(MODEL_DIR, "scene_ik.xml")
    if not os.path.exists(scene_ik_path):
        print("  Writing scene_ik.xml...")
        with open(scene_ik_path, "w") as fp:
            fp.write(SCENE_IK_XML)

    print(f"Downloading {len(MESH_FILES)} mesh files...")
    for f in MESH_FILES:
        download_file(f"{BASE_URL}/meshes/{f}", os.path.join(MESH_DIR, f))

    print(f"Done! G1 model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
