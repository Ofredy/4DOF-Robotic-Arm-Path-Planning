import numpy as np
from scipy.spatial.transform import Rotation as R

from system_constants import *


z_hat = np.array([0.0, 0.0, 1.0])
y_hat = np.array([0.0, 1.0, 0.0])
x_hat = np.array([1.0, 0.0, 0.0])


def yaw_pitch_quat(yaw, pitch):
    """
    Build Rz(yaw) * Ry(pitch) as a SciPy Rotation (unit quaternion under the hood).
    Active rotations: (Rz * Ry).apply(v) == Rz(Ry(v))
    """
    Rz = R.from_rotvec(yaw * z_hat)
    Ry = R.from_rotvec(pitch * y_hat)
    return Rz * Ry

def fk_ABCD(psi_A, theta_A, theta_B, theta_C):
    """
    Forward kinematics. Returns (origins in A), (rotations in A) as SciPy Rotations.
    """
    # Rotations between frames
    R_A_B = yaw_pitch_quat(psi_A, theta_A)     # ^A R_B
    R_B_C = R.from_rotvec(theta_B * y_hat)     # ^B R_C
    R_C_D = R.from_rotvec(theta_C * y_hat)     # ^C R_D

    R_A_C = R_A_B * R_B_C                      # ^A R_C
    R_A_D = R_A_C * R_C_D                      # ^A R_D

    # Origin positions expressed in A
    o_A = np.zeros(3)
    o_B = r_AB_inA.copy()
    r_BC_inA = R_A_B.apply(r_BC_inB)
    o_C = o_B + r_BC_inA
    r_CD_inA = R_A_C.apply(r_CD_inC)
    o_D = o_C + r_CD_inA

    return (o_A, o_B, o_C, o_D), (R_A_B, R_A_C, R_A_D)

def yaw_from_target(target_pos, psi_min=np.deg2rad(-90.0), psi_max=np.deg2rad(90.0)):
    """
    Base yaw from target direction, clipped to joint limits.
    """
    psi = np.arctan2(target_pos[1], target_pos[0])  # atan2(y, x)
    return float(np.clip(psi, psi_min, psi_max))

def cost_J(arm_angle, target_pos, 
           q_min=np.deg2rad([-90, -135, -135, -45]),
           q_max=np.deg2rad([ 90,  135,  135,  45]),
           lam=0.25):
    """
    J(q) = || p(q) - p_target ||^2
           + lam * sum_i( [q_i - q_max_i]_+^2 + [q_min_i - q_i]_+^2 )

    Args:
        q        : array-like (4,) -> [psi_A, theta_A, theta_B, theta_C] in radians
        p_target : array-like (3,) -> desired tip position in frame A
        q_min    : array-like (4,) -> per-joint lower bounds (radians)
        q_max    : array-like (4,) -> per-joint upper bounds (radians)
        lam      : float           -> penalty weight λ

    Requires:
        fk_ABCD(psi_A, theta_A, theta_B, theta_C) returning ((o_A,o_B,o_C,o_D), rotations)
        from the earlier code. We use o_D as p(q).
    """
    arm_angle      = np.asarray(arm_angle, dtype=float).reshape(4)
    q_min  = np.asarray(q_min, dtype=float).reshape(4)
    q_max  = np.asarray(q_max, dtype=float).reshape(4)
    p_tgt  = np.asarray(target_pos, dtype=float).reshape(3)

    # Forward kinematics to get tip position p(q) = o_D in frame A
    (o_A, o_B, o_C, o_D), _ = fk_ABCD(arm_angle[0], arm_angle[1], arm_angle[2], arm_angle[3])
    pos_err = o_D - p_tgt
    track_term = float(pos_err @ pos_err)

    # Hinge penalties for joint limits: [x]_+ = max(0, x)
    over  = np.maximum(arm_angle - q_max, 0.0)
    under = np.maximum(q_min - arm_angle, 0.0)
    penalty = float(over @ over + under @ under)

    return track_term + lam * penalty

def fd_position_jacobian_reduced(arm_angles, target_pos, h=None,
                                 psi_min=np.deg2rad(-90.0), psi_max=np.deg2rad(90.0)):
    """
    Finite-difference Jacobian J_p(q) = ∂p/∂[θ_A, θ_B, θ_C] (3x3),
    with base yaw ψ fixed from atan2(p_target.y, p_target.x), clipped to [psi_min, psi_max].

    Args:
        q_pitch : array-like (3,) -> [theta_A, theta_B, theta_C] in radians
        p_target : array-like (3,) target position, only used to set ψ
        h : None or (3,) finite-diff steps (rad); auto if None
    Returns:
        Jp_red : (3,3) ndarray
        p0     : (3,)   tip position at the nominal (ψ from target, q_pitch)
        psi    : float  yaw used (rad)
    """
    arm_angles = np.asarray(arm_angles, dtype=float).reshape(3)
    target_pos = np.asarray(target_pos, dtype=float).reshape(3)

    # fix yaw from target (NOT optimized)
    psi = yaw_from_target(target_pos, psi_min, psi_max)

    # auto step sizes
    if h is None:
        h = np.maximum(1e-6, 1e-3 * (np.abs(arm_angles) + 1.0))
    else:
        h = np.asarray(h, dtype=float).reshape(3)

    # base FK
    (_, _, _, p0), _ = fk_ABCD(psi, arm_angles[0], arm_angles[1], arm_angles[2])

    # central differences on θ_A, θ_B, θ_C only
    Jp = np.zeros((3, 3), dtype=float)
    for i in range(3):
        dq = np.zeros(3); dq[i] = h[i]
        # p(q + h e_i) with same ψ
        (_, _, _, p_plus), _  = fk_ABCD(psi, *(arm_angles + dq))
        # p(q - h e_i) with same ψ
        (_, _, _, p_minus), _ = fk_ABCD(psi, *(arm_angles - dq))
        Jp[:, i] = (p_plus - p_minus) / (2.0 * h[i])

    return Jp, p0, psi

def gn_delta_q_reduced(arm_angles, target_pos, mu=1e-3, h=None,
                       psi_min=np.deg2rad(-90.0), psi_max=np.deg2rad(90.0)):
    """
    Damped Gauss-Newton (Levenberg-Marquardt) step Δq for [θ_A, θ_B, θ_C] ONLY.
    Solves: (J^T J + mu I) Δq = - J^T (p(q) - p_target), with ψ fixed from target.

    Args:
        q_pitch : (3,) current [theta_A, theta_B, theta_C] (rad)
        p_target: (3,) target position in A
        mu      : damping (>=0), e.g. 1e-3 .. 1e-1
        h       : None or (3,) FD steps for Jacobian
    Returns:
        delta_q : (3,) GN/LM step for the pitch joints
        p0      : (3,) current tip position p(q)
        psi     : float yaw used (rad)
    """
    Jp, p0, psi = fd_position_jacobian_reduced(arm_angles, target_pos, h, psi_min, psi_max)
    e = p0 - np.asarray(target_pos, dtype=float).reshape(3)   # residual

    # damped normal equations (3x3 since we're solving only for three joints)
    H = Jp.T @ Jp + float(mu) * np.eye(3)
    g = Jp.T @ e
    delta_q = -np.linalg.solve(H, g)

    return delta_q, p0, psi

def numerical_grad_J(arm_angle, target_pos, 
                     q_min=np.deg2rad([-90, -135, -135, -45]),
                     q_max=np.deg2rad([ 90,  135,  135,  45]),
                     lam=0.25,
                     h=None, use_central=True):
    """
    Numerical gradient of the scalar cost J(q) via finite differences.
    Works directly with your cost_J() function.

    Args:
        q         : (4,) joint angles (rad)
        p_target  : (3,) desired tip position in frame A
        q_min     : (4,) lower bounds (rad)
        q_max     : (4,) upper bounds (rad)
        lam       : float, penalty weight
        h         : None or (4,) finite-diff steps per joint (rad). If None, auto-set.
        use_central : bool, True = central difference, False = forward difference

    Returns:
        g : (4,) ndarray, ∇J(q)
    """
    arm_angle = np.asarray(arm_angle, dtype=float).reshape(4)

    if h is None:
        h = np.maximum(1e-6, 1e-3 * (np.abs(arm_angle) + 1.0))
    else:
        h = np.asarray(h, dtype=float).reshape(4)

    g = np.zeros(4, dtype=float)

    if use_central:
        # central difference: (J(q+h) - J(q-h)) / (2h)
        for i in range(4):
            dq = np.zeros(4, dtype=float); dq[i] = h[i]
            Jp = cost_J(arm_angle + dq, target_pos, q_min, q_max, lam)
            Jm = cost_J(arm_angle - dq, target_pos, q_min, q_max, lam)
            g[i] = (Jp - Jm) / (2.0 * h[i])
    else:
        # forward difference: (J(q+h) - J(q)) / h
        J0 = cost_J(arm_angle, target_pos, q_min, q_max, lam)
        for i in range(4):
            dq = np.zeros(4, dtype=float); dq[i] = h[i]
            Jp = cost_J(arm_angle + dq, target_pos, q_min, q_max, lam)
            g[i] = (Jp - J0) / h[i]

    return g

def plan_robot_path(arm_angles, target_pos, 
                    update_factor=0.001, 
                    max_iterations=1000, 
                    state_error_theshold=0.5,
                    mu=1e-3):

    MAX_ITERATIONS = max_iterations
    STATE_ERROR_THRESHOLD = state_error_theshold # half an inch accuracy

    # initializing optimazation problem
    (_, _, _, current_pos), _ = fk_ABCD(arm_angles[0], arm_angles[1], arm_angles[2], arm_angles[3])
    state_error = np.linalg.norm(current_pos - target_pos)
    curr_iteration = 0

    # storing optimzation history
    # set yaw from target once (fixed) and use only pitch joints for optimization
    psi = yaw_from_target(target_pos)
    curr_arm_angle = np.array([psi, arm_angles[1], arm_angles[2], arm_angles[3]], dtype=float)
    arm_angle_history = [ curr_arm_angle ]
    arm_position_history = [ current_pos ]
    cost_history = [ cost_J(arm_angles, target_pos) ]
    state_error_history = [ state_error ]

    while state_error > STATE_ERROR_THRESHOLD and curr_iteration < MAX_ITERATIONS:

        # GN / LM step for [theta_A, theta_B, theta_C] with yaw fixed from target
        delta_q, current_pos, psi = gn_delta_q_reduced(curr_arm_angle[1:], target_pos, mu=mu)

        # take a step (optionally: add backtracking here)
        curr_arm_angle[1:] = curr_arm_angle[1:] + update_factor * delta_q

        state_error = np.linalg.norm(current_pos - target_pos)

        # book-keeping
        curr_cost_error = cost_J(curr_arm_angle, target_pos)
        curr_iteration += 1

        arm_angle_history.append(curr_arm_angle.copy())
        arm_position_history.append(current_pos.copy())
        cost_history.append(curr_cost_error)
        state_error_history.append(state_error)
    
    target_arm_angles = curr_arm_angle

    return (
            { 'arm_angles': arm_angle_history,
              'arm_positions': arm_position_history,
              'cost': cost_history,
              'state_error': state_error_history },
            target_arm_angles,
            curr_iteration )

def sample_targets_in_region(n, radius=10.5, seed=0):

    rng = np.random.default_rng(seed)
    pts = []
    # rejection-sample uniformly inside a sphere, then filter half-space constraints
    # expected acceptance ~ 1/2 (x>0) * ~ (1 - frac below z=-1) ~ OK
    while len(pts) < n:
        # sample inside sphere using Gaussian direction + radius^(1/3) trick
        v = rng.normal(size=(1000, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        u = rng.random(1000) ** (1.0/3.0)  # radius distribution for uniform ball
        cand = v * (u[:, None] * radius)
        # filter region: x>0, z>-1
        mask = (cand[:, 0] > 0.0) & (cand[:, 2] > -1.0)
        sel = cand[mask]
        need = n - len(pts)
        if need > 0 and len(sel) > 0:
            pts.extend(sel[:need])
    return np.array(pts[:n])

def path_planning_algorithm_analysis(
        n=200,
        start_angles=np.deg2rad(np.array([0.0, -90.0, 0.0, 0.0])),
        radius=10.5,
        seed=0,
        update_factor=0.01,
        state_error_threshold=0.1,
        max_iterations=5000):

    targets = sample_targets_in_region(n, radius=radius, seed=seed)
    iters = np.zeros(n, dtype=int)
    final_err = np.zeros(n, dtype=float)
    success = np.zeros(n, dtype=bool)
    final_q = np.zeros((n, 4), dtype=float)

    for i, tgt in enumerate(targets):
        print("iteration %d/%d" % (i, n))
        # just call your plan_robot_path here
        psi0 = np.clip(np.arctan2(tgt[1], tgt[0]), -np.pi/2, np.pi/2)
        start_angles[0] = psi0
        hist, q_sol, k = plan_robot_path(
            start_angles.copy(),
            tgt,
            update_factor=update_factor,
            max_iterations=max_iterations,
            state_error_theshold=state_error_threshold
        )

        iters[i] = k
        final_q[i] = q_sol
        final_err[i] = hist['state_error'][-1]
        success[i] = final_err[i] <= state_error_threshold

    summary = {
        "n": n,
        "success_rate": float(np.mean(success)),
        "median_iters_success": float(np.median(iters[success])) if np.any(success) else None,
        "mean_iters_success": float(np.mean(iters[success])) if np.any(success) else None,
        "median_final_err": float(np.median(final_err)),
        "mean_final_err": float(np.mean(final_err)),
        "max_final_err": float(np.max(final_err)),
    }

    results = {
        "targets": targets,
        "iters": iters,
        "final_err": final_err,
        "success": success,
        "final_q": final_q,
        "summary": summary
    }

    return results


if __name__ == '__main__':

    analysis = path_planning_algorithm_analysis(
        n=50,
        start_angles=np.deg2rad([0.0, 0.0, 0.0, 0.0]),
        radius=5.0,
        seed=42,
        update_factor=0.01,
        state_error_threshold=0.1,
        max_iterations=5000
    )

    for idx, targets in enumerate(analysis["targets"]):
        print(analysis["targets"][idx])
        import pdb; pdb.set_trace()
