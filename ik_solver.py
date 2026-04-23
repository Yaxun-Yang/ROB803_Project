"""Damped Least-Squares Inverse Kinematics solver for MuJoCo.

Algorithm:
    Given target positions for one or more end-effectors:
    1. Compute current positions via forward kinematics
    2. Error: e = x_target - x_current
    3. Compute Jacobian J = d(x)/d(q) via MuJoCo
    4. Damped least-squares: dq = J^T (J J^T + lambda^2 I)^{-1} e
    5. Update actuated joints: q += step_size * dq
    6. Clip to joint limits
    Repeat until convergence or max iterations.
"""

import mujoco
import numpy as np


def solve_ik(model, data, target_pos, body_name,
             tol=1e-3, damping=0.01, max_iter=100, step_size=1.0):
    """Solve position-only IK for a single body.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (modified in-place).
        target_pos: Desired 3D position [x, y, z].
        body_name: Name of the end-effector body.
        tol: Convergence tolerance (meters).
        damping: DLS damping factor lambda.
        max_iter: Maximum iterations.
        step_size: Update step multiplier.

    Returns:
        Final position error norm (meters).
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    assert body_id >= 0, f"Body '{body_name}' not found"

    target_pos = np.asarray(target_pos, dtype=np.float64)
    jacp = np.zeros((3, model.nv))
    I3 = np.eye(3)

    for _ in range(max_iter):
        mujoco.mj_forward(model, data)

        err = target_pos - data.xpos[body_id]
        err_norm = np.linalg.norm(err)
        if err_norm < tol:
            break

        jacp[:] = 0
        mujoco.mj_jacBody(model, data, jacp, None, body_id)
        J = jacp[:, 6:]  # skip free joint DOFs

        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * I3, err)
        data.qpos[7:] += step_size * dq

        # Clip to joint limits
        for i in range(model.njnt):
            if model.jnt_limited[i]:
                qpos_adr = model.jnt_qposadr[i]
                lo, hi = model.jnt_range[i]
                data.qpos[qpos_adr] = np.clip(data.qpos[qpos_adr], lo, hi)

    mujoco.mj_forward(model, data)
    return np.linalg.norm(target_pos - data.xpos[body_id])


def solve_ik_multi(model, data, targets,
                   tol=1e-3, damping=0.05, max_iter=200, step_size=0.5,
                   active_dof_indices=None):
    """Solve position-only IK for multiple bodies simultaneously.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (modified in-place).
        targets: List of (body_name, target_pos) tuples.
        tol: Convergence tolerance.
        damping: DLS damping factor.
        max_iter: Maximum iterations.
        step_size: Update step multiplier.
        active_dof_indices: Optional array of actuated joint indices to modify.
            When provided, only these DOFs are updated (e.g., [12..28] for
            waist + arms). When None, all actuated joints are used.

    Returns:
        Final maximum error norm across all targets (meters).
    """
    body_ids = []
    target_positions = []
    for name, pos in targets:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, f"Body '{name}' not found"
        body_ids.append(bid)
        target_positions.append(np.asarray(pos, dtype=np.float64))

    n_targets = len(targets)
    jacp_full = np.zeros((3, model.nv))

    if active_dof_indices is not None:
        active_dof_indices = np.asarray(active_dof_indices)
        n_dof = len(active_dof_indices)
    else:
        n_dof = model.nv - 6

    I = np.eye(3 * n_targets)

    for _ in range(max_iter):
        mujoco.mj_forward(model, data)

        err = np.zeros(3 * n_targets)
        J = np.zeros((3 * n_targets, n_dof))

        max_err = 0.0
        for k, (bid, tgt) in enumerate(zip(body_ids, target_positions)):
            e = tgt - data.xpos[bid]
            err[3*k:3*k+3] = e
            max_err = max(max_err, np.linalg.norm(e))

            jacp_full[:] = 0
            mujoco.mj_jacBody(model, data, jacp_full, None, bid)

            if active_dof_indices is not None:
                J[3*k:3*k+3, :] = jacp_full[:, 6 + active_dof_indices]
            else:
                J[3*k:3*k+3, :] = jacp_full[:, 6:]

        if max_err < tol:
            break

        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * I, err)

        if active_dof_indices is not None:
            data.qpos[7 + active_dof_indices] += step_size * dq
        else:
            data.qpos[7:] += step_size * dq

        # Clip to joint limits
        for i in range(model.njnt):
            if model.jnt_limited[i]:
                qpos_adr = model.jnt_qposadr[i]
                lo, hi = model.jnt_range[i]
                data.qpos[qpos_adr] = np.clip(data.qpos[qpos_adr], lo, hi)

    mujoco.mj_forward(model, data)
    return max_err
