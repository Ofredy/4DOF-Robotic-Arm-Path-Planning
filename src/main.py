import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from system_model import *
from path_planner import *


def plot_arm(psi_A, theta_A, theta_B, theta_C, show_triad=True, title='4-DOF Robotic Arm'):
    """
    Plot the arm in 3D for the given joint angles (radians).
    """
    (o_A, o_B, o_C, o_D), (R_A_B, R_A_C, R_A_D) = fk_ABCD(psi_A, theta_A, theta_B, theta_C)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot links as a polyline A->B->C->D
    X = [o_A[0], o_B[0], o_C[0], o_D[0]]
    Y = [o_A[1], o_B[1], o_C[1], o_D[1]]
    Z = [o_A[2], o_B[2], o_C[2], o_D[2]]
    ax.plot(X, Y, Z, marker='o')

    # --- add frame labels at the origins ---
    offset = 0.15  # small nudge so the text doesn't overlap the markers
    ax.text(o_A[0] + offset, o_A[1] + offset, o_A[2] + offset, "A")
    ax.text(o_B[0] + offset, o_B[1] + offset, o_B[2] + offset, "B")
    ax.text(o_C[0] + offset, o_C[1] + offset, o_C[2] + offset, "C")
    ax.text(o_D[0] + offset, o_D[1] + offset, o_D[2] + offset, "D")

    # Equal-ish aspect box
    pts = np.array([o_A, o_B, o_C, o_D])
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    span = float(max(maxs - mins))
    if span < 1e-6:
        span = 1.0
    lims = np.array([center - span*0.8, center + span*0.8]).T
    ax.set_xlim(lims[0,0], lims[0,1])
    ax.set_ylim(lims[1,0], lims[1,1])
    ax.set_zlim(lims[2,0], lims[2,1])

    ax.set_xlabel('X (A)')
    ax.set_ylabel('Y (A)')
    ax.set_zlabel('Z (A)')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# --- quick demo: tweak these if you want degrees ---
def deg(d): return np.deg2rad(d)


if __name__ == "__main__":

    arm_angle = deg([ 30.96375653, 1.63247562, 42.82374577, 0.0 ])

    plot_arm(arm_angle[0], arm_angle[1], arm_angle[2], arm_angle[3])
