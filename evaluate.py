"""Evaluation metrics and visualization for the retargeting pipeline."""

import json
import os

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    FIGURES_DIR,
    FPS,
    G1_JOINT_NAMES,
    MP_LANDMARK_NAMES,
    OUTPUT_DIR,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)


def evaluate_extraction(pose_data):
    """Compute extraction quality metrics.

    Returns dict with detection_rate, avg_visibility, per_landmark_visibility,
    jitter_mm (per-landmark frame-to-frame displacement).
    """
    frames = pose_data["frames"]
    n_frames = len(frames)

    detected_count = sum(1 for f in frames if f["detected"])
    detection_rate = detected_count / n_frames if n_frames > 0 else 0.0

    # Visibility stats
    all_vis = []
    per_lm_vis = {i: [] for i in range(33)}
    for frame in frames:
        if frame["landmarks"]:
            for lm in frame["landmarks"]:
                all_vis.append(lm["visibility"])
                per_lm_vis[lm["id"]].append(lm["visibility"])

    avg_visibility = float(np.mean(all_vis)) if all_vis else 0.0
    per_landmark_visibility = {
        MP_LANDMARK_NAMES[i]: round(float(np.mean(v)), 4) if v else 0.0
        for i, v in per_lm_vis.items()
    }

    # Jitter: frame-to-frame displacement in world_landmarks (mm)
    coords = np.zeros((n_frames, 33, 3))
    detected = np.zeros(n_frames, dtype=bool)
    for i, frame in enumerate(frames):
        if frame["detected"] and frame["world_landmarks"]:
            detected[i] = True
            for lm in frame["world_landmarks"]:
                coords[i, lm["id"]] = [lm["x"], lm["y"], lm["z"]]

    jitter_per_landmark = {}
    for lid in range(33):
        displacements = []
        for t in range(1, n_frames):
            if detected[t] and detected[t - 1]:
                d = np.linalg.norm(coords[t, lid] - coords[t - 1, lid]) * 1000
                displacements.append(d)
        name = MP_LANDMARK_NAMES[lid] if lid < len(MP_LANDMARK_NAMES) else f"lm_{lid}"
        jitter_per_landmark[name] = {
            "mean_mm": round(float(np.mean(displacements)), 3) if displacements else 0.0,
            "max_mm": round(float(np.max(displacements)), 3) if displacements else 0.0,
        }

    return {
        "detection_rate": round(detection_rate, 4),
        "detected_frames": detected_count,
        "total_frames": n_frames,
        "avg_visibility": round(avg_visibility, 4),
        "per_landmark_visibility": per_landmark_visibility,
        "jitter_per_landmark": jitter_per_landmark,
    }


def evaluate_retargeting(joint_data):
    """Compute retargeting quality metrics.

    Returns dict with ik_error stats, joint smoothness, joint utilization.
    """
    frames = joint_data["frames"]

    # IK errors
    left_errors = [f["ik_error_left"] for f in frames if f["ik_error_left"] is not None]
    right_errors = [f["ik_error_right"] for f in frames if f["ik_error_right"] is not None]

    def error_stats(errors, label):
        if not errors:
            return {}
        arr = np.array(errors) * 1000  # to mm
        return {
            f"{label}_mean_mm": round(float(arr.mean()), 3),
            f"{label}_max_mm": round(float(arr.max()), 3),
            f"{label}_std_mm": round(float(arr.std()), 3),
        }

    ik_stats = {}
    ik_stats.update(error_stats(left_errors, "left_wrist"))
    ik_stats.update(error_stats(right_errors, "right_wrist"))

    # Joint angle smoothness: frame-to-frame change
    n_frames = len(frames)
    n_joints = 29
    angles = np.zeros((n_frames, n_joints))
    for i, f in enumerate(frames):
        angles[i] = f["joint_angles"]

    joint_smoothness = {}
    for j in range(n_joints):
        deltas = np.abs(np.diff(angles[:, j]))
        name = G1_JOINT_NAMES[j] if j < len(G1_JOINT_NAMES) else f"joint_{j}"
        joint_smoothness[name] = {
            "mean_delta_rad": round(float(deltas.mean()), 6),
            "max_delta_rad": round(float(deltas.max()), 6),
        }

    return {
        "ik_error": ik_stats,
        "joint_smoothness": joint_smoothness,
        "total_frames": n_frames,
    }


def generate_figures(extraction_metrics, retargeting_metrics, pose_data,
                     joint_data, output_dir=None):
    """Generate evaluation plots."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    # 1. Confidence over time
    frames = pose_data["frames"]
    n_frames = len(frames)
    mean_vis = []
    for frame in frames:
        if frame["landmarks"]:
            vis = [lm["visibility"] for lm in frame["landmarks"]]
            mean_vis.append(np.mean(vis))
        else:
            mean_vis.append(0.0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mean_vis, linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean Visibility")
    ax.set_title("Landmark Detection Confidence Over Time")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confidence_over_time.png"), dpi=150)
    plt.close(fig)

    # 2. Jitter analysis (top 10 landmarks by mean jitter)
    jitter = extraction_metrics["jitter_per_landmark"]
    sorted_lm = sorted(jitter.items(), key=lambda x: x[1]["mean_mm"], reverse=True)[:10]
    names = [x[0] for x in sorted_lm]
    means = [x[1]["mean_mm"] for x in sorted_lm]
    maxes = [x[1]["max_mm"] for x in sorted_lm]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, means, 0.4, label="Mean", color="steelblue")
    ax.bar(x + 0.2, maxes, 0.4, label="Max", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Jitter (mm/frame)")
    ax.set_title("Top 10 Landmarks by Temporal Jitter")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "jitter_analysis.png"), dpi=150)
    plt.close(fig)

    # 3. IK error over time
    jf = joint_data["frames"]
    left_err = [f["ik_error_left"] for f in jf]
    right_err = [f["ik_error_right"] for f in jf]
    # Replace None with NaN for plotting
    left_err = [e * 1000 if e is not None else np.nan for e in left_err]
    right_err = [e * 1000 if e is not None else np.nan for e in right_err]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(left_err, label="Left wrist", linewidth=0.8)
    ax.plot(right_err, label="Right wrist", linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("IK Error (mm)")
    ax.set_title("IK Residual Error Per Frame")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ik_error.png"), dpi=150)
    plt.close(fig)

    # 4. Joint smoothness heatmap (active joints only: waist + arms, indices 12-28)
    n_jf = len(jf)
    active_indices = list(range(12, 29))
    active_names = [G1_JOINT_NAMES[j] for j in active_indices]
    angles = np.zeros((n_jf, len(active_indices)))
    for i, f in enumerate(jf):
        for k, j in enumerate(active_indices):
            angles[i, k] = f["joint_angles"][j]

    deltas = np.abs(np.diff(angles, axis=0))

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(deltas.T, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Joint")
    ax.set_yticks(range(len(active_names)))
    ax.set_yticklabels([n.replace("_joint", "") for n in active_names], fontsize=7)
    ax.set_title("Joint Angle Change Per Frame (rad)")
    fig.colorbar(im, ax=ax, label="|\u0394q| (rad)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "joint_smoothness.png"), dpi=150)
    plt.close(fig)

    # 5. Trajectory comparison (3D target vs actual)
    targets_left = []
    actual_left = []
    targets_right = []
    actual_right = []
    for f in jf:
        if "left_wrist_target" in f and f["left_wrist_target"] is not None:
            targets_left.append(f["left_wrist_target"])
            actual_left.append(f["left_wrist_actual"])
            targets_right.append(f["right_wrist_target"])
            actual_right.append(f["right_wrist_actual"])

    if targets_left:
        tl = np.array(targets_left)
        al = np.array(actual_left)
        tr = np.array(targets_right)
        ar = np.array(actual_right)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(tl[:, 0], tl[:, 1], tl[:, 2], "r-", label="Left target", alpha=0.7)
        ax.plot(al[:, 0], al[:, 1], al[:, 2], "r--", label="Left actual", alpha=0.7)
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], "b-", label="Right target", alpha=0.7)
        ax.plot(ar[:, 0], ar[:, 1], ar[:, 2], "b--", label="Right actual", alpha=0.7)
        ax.set_xlabel("X (forward)")
        ax.set_ylabel("Y (left)")
        ax.set_zlabel("Z (up)")
        ax.set_title("Wrist Trajectory: Target vs Actual")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "trajectory_comparison.png"), dpi=150)
        plt.close(fig)

    print(f"  Figures saved to {output_dir}")


def generate_side_by_side_video(original_video, skeleton_video, robot_video,
                                output_path=None, fps=None):
    """Create 3-panel side-by-side comparison video."""
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "side_by_side.mp4")

    cap_orig = cv2.VideoCapture(original_video)
    cap_skel = cv2.VideoCapture(skeleton_video)
    cap_robot = cv2.VideoCapture(robot_video)

    if fps is None:
        fps = cap_orig.get(cv2.CAP_PROP_FPS) or FPS

    panel_w = VIDEO_WIDTH // 3
    panel_h = VIDEO_HEIGHT

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", quality=None,
        output_params=["-pix_fmt", "yuv420p", "-profile:v", "baseline",
                       "-level", "3.1", "-movflags", "+faststart"],
    )

    frame_idx = 0
    while True:
        ret1, f1 = cap_orig.read()
        ret2, f2 = cap_skel.read()
        ret3, f3 = cap_robot.read()

        if not (ret1 and ret2 and ret3):
            break

        # Resize each panel
        p1 = cv2.resize(f1, (panel_w, panel_h))
        p2 = cv2.resize(f2, (panel_w, panel_h))
        # Robot video is already RGB from imageio, but skeleton overlay is BGR from OpenCV
        p3_bgr = cv2.resize(f3, (panel_w, panel_h))

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(p1, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(p2, "Skeleton", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(p3_bgr, "Robot", (10, 30), font, 0.8, (255, 255, 255), 2)

        combined = np.hstack([p1, p2, p3_bgr])
        # Convert BGR to RGB for imageio
        writer.append_data(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        frame_idx += 1

    cap_orig.release()
    cap_skel.release()
    cap_robot.release()
    writer.close()
    print(f"  Side-by-side video saved to {output_path} ({frame_idx} frames)")


def run_evaluation(pose_data, joint_data, video_path=None):
    """Run full evaluation and save results."""
    print("Evaluating extraction quality...")
    ext_metrics = evaluate_extraction(pose_data)

    print("Evaluating retargeting quality...")
    ret_metrics = evaluate_retargeting(joint_data)

    report = {
        "extraction": ext_metrics,
        "retargeting": ret_metrics,
    }

    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {report_path}")

    print("Generating figures...")
    generate_figures(ext_metrics, ret_metrics, pose_data, joint_data)

    # Side-by-side video
    skeleton_path = os.path.join(OUTPUT_DIR, "skeleton_overlay.mp4")
    robot_path = os.path.join(OUTPUT_DIR, "robot_gesture.mp4")
    if video_path and os.path.exists(skeleton_path) and os.path.exists(robot_path):
        print("Generating side-by-side video...")
        generate_side_by_side_video(video_path, skeleton_path, robot_path)

    return report


if __name__ == "__main__":
    from extract_pose import load_pose_data

    pose_data = load_pose_data(os.path.join(OUTPUT_DIR, "pose_data.json"))
    with open(os.path.join(OUTPUT_DIR, "joint_angles.json")) as f:
        joint_data = json.load(f)
    run_evaluation(pose_data, joint_data)
