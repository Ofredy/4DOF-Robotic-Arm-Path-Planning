import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


# ---- Fixed link vectors (as given) ----
r_AB_inA = np.array([0.0, 0.0, 4.0])   # B wrt A expressed in A
r_BC_inB = np.array([4.5, 0.0, 0.0])   # C wrt B expressed in B
r_CD_inC = np.array([2.0, 0.0, 0.0])   # D wrt C expressed in C

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

def draw_triad(ax, origin, R_world_frame, scale=1.0):
    """
    Draw a small coordinate triad at 'origin' using rotation R_world_frame (SciPy Rotation).
    """
    Rx = R_world_frame.apply(x_hat)
    Ry = R_world_frame.apply(y_hat)
    Rz = R_world_frame.apply(z_hat)
    # Let matplotlib choose default colors (no explicit colors)
    ax.quiver(origin[0], origin[1], origin[2], Rx[0], Ry[0], Rz[0], length=0, arrow_length_ratio=0)
    ax.quiver(origin[0], origin[1], origin[2], Rx[0], Rx[1], Rx[2], length=scale, normalize=True)
    ax.quiver(origin[0], origin[1], origin[2], Ry[0], Ry[1], Ry[2], length=scale, normalize=True)
    ax.quiver(origin[0], origin[1], origin[2], Rz[0], Rz[1], Rz[2], length=scale, normalize=True)

def plot_arm(psi_A, theta_A, theta_B, theta_C, show_triad=True, title='OWI-535 A–B–C–D arm (quaternion rotations)'):
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

    # Triads
    if show_triad:
        draw_triad(ax, o_A, R.from_quat([0,0,0,1]), scale=1.0)     # frame A
        draw_triad(ax, o_B, R_A_B, scale=1.0)                      # frame B in A
        draw_triad(ax, o_C, R_A_C, scale=1.0)                      # frame C in A
        draw_triad(ax, o_D, R_A_D, scale=0.7)                      # frame D in A

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

    psi_A   = deg(0.0)   # yaw at A->B
    th_A    = deg(0.0)   # pitch at A->B
    th_B    = deg(0.0)  # pitch at B->C
    th_C    = deg(0.0)   # pitch at C->D

    plot_arm(psi_A, th_A, th_B, th_C)