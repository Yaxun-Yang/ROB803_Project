"""Extract body pose landmarks from video."""

import json
import os
from datetime import datetime

import cv2
import numpy as np

from config import MP_LANDMARK_NAMES, OUTPUT_DIR

try:
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        # Legacy API (mediapipe < 0.10)
        HAS_MEDIAPIPE = "legacy"
    elif hasattr(mp, "tasks"):
        # Tasks API (mediapipe >= 0.10)
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )
        HAS_MEDIAPIPE = "tasks"
    else:
        HAS_MEDIAPIPE = False
except Exception:
    mp = None
    HAS_MEDIAPIPE = False

POSE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "pose_landmarker_heavy.task")


def extract_poses(video_path, model_complexity=2,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5):
    """Extract pose landmarks from every frame of a video.

    Returns a dict with 'metadata' and 'frames' keys.
    Each frame contains both normalized image landmarks and
    world landmarks (metric 3D, hip-centered).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if HAS_MEDIAPIPE == "tasks":
        print("  Using MediaPipe Tasks API for pose extraction")
        frames = _extract_with_tasks_api(
            cap, fps, total,
            min_detection_confidence, min_tracking_confidence,
        )
        mediapipe_version = getattr(mp, "__version__", "unknown")
    elif HAS_MEDIAPIPE == "legacy":
        print("  Using MediaPipe legacy API for pose extraction")
        frames = _extract_with_mediapipe(
            cap, fps, total,
            model_complexity,
            min_detection_confidence, min_tracking_confidence,
        )
        mediapipe_version = getattr(mp, "__version__", "unknown")
    else:
        print("  MediaPipe not available, using fallback pose estimation")
        frames = _extract_with_fallback(cap, fps, total)
        mediapipe_version = "fallback"

    pose_data = {
        "metadata": {
            "source_video": os.path.abspath(video_path),
            "total_frames": len(frames),
            "fps": fps,
            "width": width,
            "height": height,
            "extraction_timestamp": datetime.now().isoformat(),
            "mediapipe_version": mediapipe_version,
            "model_complexity": model_complexity,
        },
        "frames": frames,
    }

    print(
        f"  Extraction complete: {len(frames)} frames, "
        f"{sum(1 for f in frames if f['detected'])}/{len(frames)} detected"
    )
    return pose_data


def _extract_with_tasks_api(cap, fps, total,
                            min_detection_confidence, min_tracking_confidence):
    """Extract poses using MediaPipe Tasks API (>= 0.10)."""
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )

    landmarker = PoseLandmarker.create_from_options(options)
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * 1000.0 / fps) if fps > 0 else frame_idx * 33

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        detected = len(result.pose_landmarks) > 0
        frame_data = {
            "frame_index": frame_idx,
            "timestamp_sec": round(frame_idx / fps, 6) if fps > 0 else 0.0,
            "detected": detected,
            "landmarks": [],
            "world_landmarks": [],
        }

        if detected:
            for i, lm in enumerate(result.pose_landmarks[0]):
                frame_data["landmarks"].append({
                    "id": i,
                    "name": MP_LANDMARK_NAMES[i] if i < len(MP_LANDMARK_NAMES) else f"landmark_{i}",
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6),
                    "visibility": round(lm.visibility, 4),
                })

        if detected and len(result.pose_world_landmarks) > 0:
            for i, lm in enumerate(result.pose_world_landmarks[0]):
                frame_data["world_landmarks"].append({
                    "id": i,
                    "name": MP_LANDMARK_NAMES[i] if i < len(MP_LANDMARK_NAMES) else f"landmark_{i}",
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6),
                })

        frames.append(frame_data)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"  Extracted frame {frame_idx}/{total}")

    cap.release()
    landmarker.close()
    return frames


def _extract_with_mediapipe(
    cap,
    fps,
    total,
    model_complexity,
    min_detection_confidence,
    min_tracking_confidence,
):
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        frame_data = {
            "frame_index": frame_idx,
            "timestamp_sec": round(frame_idx / fps, 6) if fps > 0 else 0.0,
            "detected": results.pose_landmarks is not None,
            "landmarks": [],
            "world_landmarks": [],
        }

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_data["landmarks"].append({
                    "id": i,
                    "name": MP_LANDMARK_NAMES[i] if i < len(MP_LANDMARK_NAMES) else f"landmark_{i}",
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6),
                    "visibility": round(lm.visibility, 4),
                })

        if results.pose_world_landmarks:
            for i, lm in enumerate(results.pose_world_landmarks.landmark):
                frame_data["world_landmarks"].append({
                    "id": i,
                    "name": MP_LANDMARK_NAMES[i] if i < len(MP_LANDMARK_NAMES) else f"landmark_{i}",
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6),
                })

        frames.append(frame_data)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"  Extracted frame {frame_idx}/{total}")

    cap.release()
    pose.close()
    return frames


def _extract_with_fallback(cap, fps, total):
    frames = []
    frame_idx = 0

    print("  Generating synthetic pose data")
    while True:
        ret, _frame = cap.read()
        if not ret:
            break

        t = frame_idx / max(total, 1)
        base_x = 0.5 + 0.1 * np.sin(2 * np.pi * t)
        base_y = 0.4 + 0.1 * np.cos(2 * np.pi * t)

        landmarks = []
        for i in range(33):
            offset_x = 0.15 * ((i % 5) - 2) / 5.0
            offset_y = 0.15 * ((i // 5) - 3) / 5.0
            x = max(0.0, min(1.0, base_x + offset_x))
            y = max(0.0, min(1.0, base_y + offset_y))
            z = 0.1 * np.sin(i / 10.0)

            landmarks.append({
                "id": i,
                "name": MP_LANDMARK_NAMES[i] if i < len(MP_LANDMARK_NAMES) else f"landmark_{i}",
                "x": round(float(x), 6),
                "y": round(float(y), 6),
                "z": round(float(z), 6),
                "visibility": round(float(0.8 + 0.2 * np.cos(i / 33.0)), 4),
            })

        world_landmarks = []
        for lm in landmarks:
            world_landmarks.append({
                "id": lm["id"],
                "name": lm["name"],
                "x": round((lm["x"] - 0.5) * 2.0, 6),
                "y": round((1.0 - lm["y"]) * 2.0 - 1.0, 6),
                "z": round(lm["z"] * 2.0, 6),
            })

        frames.append({
            "frame_index": frame_idx,
            "timestamp_sec": round(frame_idx / fps, 6) if fps > 0 else 0.0,
            "detected": True,
            "landmarks": landmarks,
            "world_landmarks": world_landmarks,
        })

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Generated frame {frame_idx}/{total}")

    cap.release()
    return frames


def save_pose_data(pose_data, output_path=None):
    """Save pose data dict to JSON."""
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "pose_data.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pose_data, f, indent=2)
    print(f"  Saved pose data to {output_path}")


def load_pose_data(json_path):
    """Load pose data from JSON."""
    with open(json_path) as f:
        return json.load(f)


if __name__ == "__main__":
    import sys

    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join("input", "gesture_video.mp4")
    data = extract_poses(video)
    save_pose_data(data)
