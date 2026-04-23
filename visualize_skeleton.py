"""Skeleton visualization: 2D overlay on video and 3D stick figure animation."""

import os

import cv2
import imageio
import numpy as np

from config import FPS, OUTPUT_DIR, VIDEO_HEIGHT, VIDEO_WIDTH

# MediaPipe pose connection pairs for drawing skeleton
POSE_CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso sides
    (23, 24),  # hips
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
    (15, 17), (15, 19), (17, 19),  # left hand
    (16, 18), (16, 20), (18, 20),  # right hand
    (27, 29), (29, 31),  # left foot
    (28, 30), (30, 32),  # right foot
    (0, 11), (0, 12),    # nose to shoulders (approximate neck)
]

# Landmark indices used for skeleton drawing (upper body focus)
UPPER_BODY_LANDMARKS = {0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24}


def _visibility_color(visibility):
    """Return BGR color based on visibility score."""
    if visibility > 0.8:
        return (0, 255, 0)    # green
    elif visibility > 0.5:
        return (0, 255, 255)  # yellow
    else:
        return (0, 0, 255)    # red


def draw_skeleton_overlay(video_path, pose_data, output_path=None, fps=None):
    """Draw 2D skeleton on each frame of the original video.

    Uses normalized image coordinates from pose_data landmarks.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "skeleton_overlay.mp4")
    if fps is None:
        fps = pose_data["metadata"]["fps"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_data = pose_data["frames"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", quality=None,
        output_params=["-pix_fmt", "yuv420p", "-profile:v", "baseline",
                       "-level", "3.1", "-movflags", "+faststart"],
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(frames_data) and frames_data[frame_idx]["detected"]:
            landmarks = frames_data[frame_idx]["landmarks"]
            lm_dict = {lm["id"]: lm for lm in landmarks}

            # Draw connections
            for (a, b) in POSE_CONNECTIONS:
                if a in lm_dict and b in lm_dict:
                    la, lb = lm_dict[a], lm_dict[b]
                    if la["visibility"] > 0.3 and lb["visibility"] > 0.3:
                        pt1 = (int(la["x"] * img_w), int(la["y"] * img_h))
                        pt2 = (int(lb["x"] * img_w), int(lb["y"] * img_h))
                        avg_vis = (la["visibility"] + lb["visibility"]) / 2
                        color = _visibility_color(avg_vis)
                        cv2.line(frame, pt1, pt2, color, 2)

            # Draw landmarks
            for lm in landmarks:
                if lm["visibility"] > 0.3:
                    pt = (int(lm["x"] * img_w), int(lm["y"] * img_h))
                    color = _visibility_color(lm["visibility"])
                    radius = 5 if lm["id"] in UPPER_BODY_LANDMARKS else 3
                    cv2.circle(frame, pt, radius, color, -1)

        # Convert BGR to RGB for imageio
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"  Overlay frame {frame_idx}")

    cap.release()
    writer.close()
    print(f"  Skeleton overlay saved to {output_path}")


def render_3d_skeleton(pose_data, output_path=None, fps=None):
    """Render standalone 3D skeleton animation using matplotlib.

    Uses world_landmarks (metric coordinates, hip-centered).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "skeleton_3d.mp4")
    if fps is None:
        fps = pose_data["metadata"]["fps"]

    frames_data = pose_data["frames"]

    # Compute axis limits from all frames
    all_coords = []
    for frame in frames_data:
        if frame["detected"] and frame["world_landmarks"]:
            for lm in frame["world_landmarks"]:
                all_coords.append([lm["x"], lm["y"], lm["z"]])

    if not all_coords:
        print("  No detected frames, skipping 3D skeleton render")
        return

    all_coords = np.array(all_coords)
    center = all_coords.mean(axis=0)
    max_range = max(all_coords.max(axis=0) - all_coords.min(axis=0)) * 0.6

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", quality=None,
        output_params=["-pix_fmt", "yuv420p", "-profile:v", "baseline",
                       "-level", "3.1", "-movflags", "+faststart"],
    )

    fig = plt.figure(figsize=(8, 6), dpi=120)

    for i, frame in enumerate(frames_data):
        fig.clear()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[2] - max_range, center[2] + max_range)  # z->vertical
        ax.set_zlim(center[1] - max_range, center[1] + max_range)
        ax.set_xlabel("X (right)")
        ax.set_ylabel("Z (depth)")
        ax.set_zlabel("Y (up)")
        ax.set_title(f"Frame {i} / {len(frames_data)}  t={frame['timestamp_sec']:.2f}s")

        # Slow camera orbit
        ax.view_init(elev=15, azim=-60 + i * 0.3)

        if frame["detected"] and frame["world_landmarks"]:
            lm_dict = {lm["id"]: lm for lm in frame["world_landmarks"]}

            # Draw connections
            for (a, b) in POSE_CONNECTIONS:
                if a in lm_dict and b in lm_dict:
                    la, lb = lm_dict[a], lm_dict[b]
                    ax.plot(
                        [la["x"], lb["x"]],
                        [la["z"], lb["z"]],  # depth as y-axis
                        [la["y"], lb["y"]],  # height as z-axis
                        "b-", linewidth=1.5,
                    )

            # Draw landmarks
            for lm in frame["world_landmarks"]:
                color = "green" if lm["id"] in UPPER_BODY_LANDMARKS else "gray"
                size = 20 if lm["id"] in UPPER_BODY_LANDMARKS else 8
                ax.scatter(lm["x"], lm["z"], lm["y"], c=color, s=size)

        # Render to image
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        writer.append_data(img)

        if (i + 1) % 30 == 0:
            print(f"  3D skeleton frame {i + 1}/{len(frames_data)}")

    plt.close(fig)
    writer.close()
    print(f"  3D skeleton saved to {output_path}")


if __name__ == "__main__":
    import sys
    from extract_pose import load_pose_data

    pose_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTPUT_DIR, "pose_data.json")
    data = load_pose_data(pose_path)

    if len(sys.argv) > 2:
        video = sys.argv[2]
        draw_skeleton_overlay(video, data)

    render_3d_skeleton(data)
