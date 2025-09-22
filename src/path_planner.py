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

def fd_position_jacobian(q, h=None):
    """
    Finite-difference Jacobian of the position error r_p(q) = p(q) - p_target,
    but WITHOUT the target term so you can reuse it: J_p(q) = ∂p/∂q (3x4).

    Args:
        q : array-like (4,) [psi_A, theta_A, theta_B, theta_C] in radians
        h : None or array-like (4,) per-joint step sizes (radians). If None, auto-set.

    Returns:
        Jp : (3,4) ndarray, where column i ≈ (p(q+hi e_i) - p(q-hi e_i)) / (2*hi)
    """
    q = np.asarray(q, dtype=float).reshape(4)
    if h is None:
        # per-joint central-diff steps (robust across magnitudes)
        h = np.maximum(1e-6, 1e-3 * (np.abs(q) + 1.0))
    else:
        h = np.asarray(h, dtype=float).reshape(4)

    # base tip position p(q)
    (_, _, _, p0), _ = fk_ABCD(q[0], q[1], q[2], q[3])

    Jp = np.zeros((3, 4), dtype=float)
    for i in range(4):
        dq = np.zeros(4, dtype=float); dq[i] = h[i]
        # p(q + h e_i)
        (_, _, _, p_plus), _  = fk_ABCD(*(q + dq))
        # p(q - h e_i)
        (_, _, _, p_minus), _ = fk_ABCD(*(q - dq))
        Jp[:, i] = (p_plus - p_minus) / (2.0 * h[i])
    return Jp

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

def plan_robot_path(arm_angles, target_pos, update_factor=0.001, max_iterations=1000, state_error_theshold=0.5):

    MAX_ITERATIONS = max_iterations
    STATE_ERROR_THRESHOLD = state_error_theshold # half an inch accuracy

    # initializing optimazation problem
    (_, _, _, current_pos), _ = fk_ABCD(arm_angles[0], arm_angles[1], arm_angles[2], arm_angles[3])
    state_error = np.linalg.norm(current_pos - target_pos)
    curr_iteration = 0

    # storing optimzation history
    curr_arm_angle = arm_angles
    arm_angle_history = [ curr_arm_angle ]
    arm_position_history = [ current_pos ]
    cost_history = [ cost_J(arm_angles, target_pos) ]
    state_error_history = [ state_error ]

    while state_error > STATE_ERROR_THRESHOLD and curr_iteration < MAX_ITERATIONS:
    
        arm_angle_cost_jacobian = numerical_grad_J(curr_arm_angle, target_pos)
        curr_arm_angle -= arm_angle_cost_jacobian * update_factor

        curr_cost_error = cost_J(curr_arm_angle, target_pos)
        (_, _, _, current_pos), _ = fk_ABCD(curr_arm_angle[0], curr_arm_angle[1], curr_arm_angle[2], curr_arm_angle[3])
        state_error = np.linalg.norm( current_pos - target_pos )
        curr_iteration += 1

        arm_angle_history.append(curr_arm_angle)
        arm_position_history.append(current_pos)
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


if __name__ == '__main__':

    starting_arm_angles = np.deg2rad(np.array([ 0.0, -90.0, 0.0, 0.0 ]))
    target_pos = np.array([ 5.0, 3.0, 2.5 ])

    optimazation_history, target_arm_angles, num_iterations = plan_robot_path(starting_arm_angles, 
                                                                              target_pos,
                                                                              update_factor=0.01,
                                                                              state_error_theshold=0.1)
    print(optimazation_history['arm_positions'][-1])
    print(optimazation_history['state_error'][-1])
    print(np.rad2deg(target_arm_angles))
    print("num_iterations: %d" % (num_iterations))
