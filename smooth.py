"""Temporal smoothing filters for MediaPipe pose landmarks."""

import copy
import warnings

import numpy as np


def _get_world_coords_array(pose_data):
    """Extract world landmark coordinates as (N_frames, 33, 3) array.

    Returns the array and a boolean mask of detected frames.
    """
    frames = pose_data["frames"]
    n_frames = len(frames)
    n_landmarks = 33
    coords = np.zeros((n_frames, n_landmarks, 3))
    detected = np.zeros(n_frames, dtype=bool)

    for i, frame in enumerate(frames):
        if frame["detected"] and frame["world_landmarks"]:
            detected[i] = True
            for lm in frame["world_landmarks"]:
                coords[i, lm["id"]] = [lm["x"], lm["y"], lm["z"]]

    return coords, detected


def _set_world_coords(pose_data, coords):
    """Write smoothed coordinates back into pose_data world_landmarks."""
    frames = pose_data["frames"]
    for i, frame in enumerate(frames):
        if frame["world_landmarks"]:
            for lm in frame["world_landmarks"]:
                lid = lm["id"]
                lm["x"] = round(float(coords[i, lid, 0]), 6)
                lm["y"] = round(float(coords[i, lid, 1]), 6)
                lm["z"] = round(float(coords[i, lid, 2]), 6)


def interpolate_missing_frames(pose_data):
    """Fill missing frames via linear interpolation.

    Gaps longer than 10 frames are left untouched with a warning.
    Modifies pose_data in-place and returns it.
    """
    coords, detected = _get_world_coords_array(pose_data)
    n_frames = len(detected)

    if detected.all():
        return pose_data

    # Find gaps
    i = 0
    while i < n_frames:
        if not detected[i]:
            # Find gap boundaries
            gap_start = i
            while i < n_frames and not detected[i]:
                i += 1
            gap_end = i  # first detected frame after gap

            gap_len = gap_end - gap_start
            if gap_len > 10:
                warnings.warn(
                    f"Long gap ({gap_len} frames) at frames {gap_start}-{gap_end-1}, "
                    "skipping interpolation for this segment"
                )
                continue

            # Find boundary frames for interpolation
            before = gap_start - 1 if gap_start > 0 and detected[gap_start - 1] else None
            after = gap_end if gap_end < n_frames and detected[gap_end] else None

            if before is not None and after is not None:
                for j in range(gap_start, gap_end):
                    t = (j - before) / (after - before)
                    coords[j] = (1 - t) * coords[before] + t * coords[after]
                    detected[j] = True
            elif before is not None:
                for j in range(gap_start, gap_end):
                    coords[j] = coords[before]
                    detected[j] = True
            elif after is not None:
                for j in range(gap_start, gap_end):
                    coords[j] = coords[after]
                    detected[j] = True
        else:
            i += 1

    # Update pose_data with interpolated values and mark as detected
    _set_world_coords(pose_data, coords)
    for i, frame in enumerate(pose_data["frames"]):
        if detected[i] and not frame["detected"]:
            frame["detected"] = True
            # Rebuild world_landmarks from coords
            frame["world_landmarks"] = [
                {"id": lid, "name": frame.get("world_landmarks", [{}])[0].get("name", f"landmark_{lid}") if frame.get("world_landmarks") else f"landmark_{lid}",
                 "x": round(float(coords[i, lid, 0]), 6),
                 "y": round(float(coords[i, lid, 1]), 6),
                 "z": round(float(coords[i, lid, 2]), 6)}
                for lid in range(33)
            ]

    return pose_data


def moving_average_smooth(pose_data, window=5):
    """Apply causal moving average to world_landmarks.

    Returns a deep copy with smoothed data.
    """
    smoothed = copy.deepcopy(pose_data)
    coords, detected = _get_world_coords_array(smoothed)
    n_frames = coords.shape[0]

    smoothed_coords = np.copy(coords)
    for i in range(n_frames):
        start = max(0, i - window + 1)
        mask = detected[start:i + 1]
        if mask.any():
            smoothed_coords[i] = coords[start:i + 1][mask].mean(axis=0)

    _set_world_coords(smoothed, smoothed_coords)
    return smoothed


def butterworth_smooth(pose_data, cutoff_hz=6.0, order=2):
    """Apply zero-phase Butterworth low-pass filter to world_landmarks.

    Returns a deep copy with smoothed data. Requires scipy.
    """
    from scipy.signal import butter, filtfilt

    smoothed = copy.deepcopy(pose_data)
    coords, detected = _get_world_coords_array(smoothed)
    fps = smoothed["metadata"]["fps"]

    if fps <= 0:
        warnings.warn("Invalid FPS, skipping Butterworth smoothing")
        return smoothed

    nyquist = fps / 2.0
    if cutoff_hz >= nyquist:
        cutoff_hz = nyquist * 0.9
        warnings.warn(f"Cutoff frequency adjusted to {cutoff_hz:.1f} Hz (below Nyquist)")

    b, a = butter(order, cutoff_hz / nyquist, btype="low")

    # Only filter detected segments
    n_frames = coords.shape[0]
    if detected.sum() < 2 * max(len(b), len(a)):
        warnings.warn("Not enough detected frames for Butterworth filter")
        return smoothed

    # Filter each landmark coordinate independently
    smoothed_coords = np.copy(coords)
    for lm_idx in range(33):
        for axis in range(3):
            signal = coords[:, lm_idx, axis]
            if detected.all():
                smoothed_coords[:, lm_idx, axis] = filtfilt(b, a, signal)
            else:
                # Filter only contiguous detected segments
                mask = detected.copy()
                indices = np.where(mask)[0]
                if len(indices) >= 2 * max(len(b), len(a)):
                    # Interpolate to fill gaps temporarily, then filter
                    interp_signal = np.interp(
                        np.arange(n_frames), indices, signal[indices]
                    )
                    filtered = filtfilt(b, a, interp_signal)
                    smoothed_coords[mask, lm_idx, axis] = filtered[mask]

    _set_world_coords(smoothed, smoothed_coords)
    return smoothed
