"""Run the complete gesture retargeting pipeline."""

import os
import sys

from config import OUTPUT_DIR


def main(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Extract poses
    print("=" * 60)
    print("STEP 1: Extracting poses with MediaPipe...")
    print("=" * 60)
    from extract_pose import extract_poses, save_pose_data
    pose_data = extract_poses(video_path)
    save_pose_data(pose_data)

    # Step 2: Smooth poses
    print("\n" + "=" * 60)
    print("STEP 2: Smoothing pose data...")
    print("=" * 60)
    from smooth import butterworth_smooth, interpolate_missing_frames
    pose_data = interpolate_missing_frames(pose_data)
    pose_data_smooth = butterworth_smooth(pose_data)
    # Save smoothed data
    from extract_pose import save_pose_data as save
    save(pose_data_smooth, os.path.join(OUTPUT_DIR, "pose_data_smoothed.json"))
    print("  Smoothing complete")

    # Step 3: Visualize skeleton
    print("\n" + "=" * 60)
    print("STEP 3: Generating skeleton visualizations...")
    print("=" * 60)
    from visualize_skeleton import draw_skeleton_overlay, render_3d_skeleton
    draw_skeleton_overlay(video_path, pose_data_smooth)
    render_3d_skeleton(pose_data_smooth)

    # Step 4: Retarget to G1 robot
    print("\n" + "=" * 60)
    print("STEP 4: Retargeting to G1 robot...")
    print("=" * 60)
    from retarget import retarget_gesture
    joint_data = retarget_gesture(pose_data_smooth)

    # Step 5: Evaluate
    print("\n" + "=" * 60)
    print("STEP 5: Evaluating results...")
    print("=" * 60)
    from evaluate import run_evaluation
    report = run_evaluation(pose_data_smooth, joint_data, video_path)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Detection rate: {report['extraction']['detection_rate']:.1%}")
    ik = report["retargeting"]["ik_error"]
    if "left_wrist_mean_mm" in ik:
        print(f"IK error (mean): L={ik['left_wrist_mean_mm']:.1f}mm "
              f"R={ik['right_wrist_mean_mm']:.1f}mm")
    print(f"\nOutputs in {OUTPUT_DIR}/:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            print(f"  {f:40s} {size/1024:.0f} KB")
    figures_dir = os.path.join(OUTPUT_DIR, "figures")
    if os.path.isdir(figures_dir):
        for f in sorted(os.listdir(figures_dir)):
            size = os.path.getsize(os.path.join(figures_dir, f))
            print(f"  figures/{f:33s} {size/1024:.0f} KB")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join("input", "gesture_video.mp4")
    main(video)
