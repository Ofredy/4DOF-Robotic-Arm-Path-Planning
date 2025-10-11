from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


def _as_np3(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(3)
    return a


def _as_np33(M: Sequence[Sequence[float]]) -> np.ndarray:
    A = np.asarray(M, dtype=float).reshape(3, 3)
    return A


def _normalize_quat(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Quaternion has zero norm.")
    return q / n


def _quat_to_R_wxyz(q: Sequence[float]) -> np.ndarray:
    """
    Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix.
    """
    w, x, y, z = _normalize_quat(q)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = np.array([
        [ww + xx - yy - zz,     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),           ww - xx + yy - zz,   2*(yz - wx)],
        [2*(xz - wy),           2*(yz + wx),         ww - xx - yy + zz]
    ], dtype=float)
    return R


def rotate_inertia(I_child: Sequence[Sequence[float]], q_parent_from_child: Sequence[float]) -> np.ndarray:
    """
    Rotate inertia tensor from child frame into parent frame:
        I_parent = R * I_child * R^T
    q is expected as [w, x, y, z].
    """
    I = _as_np33(I_child)
    R = _quat_to_R_wxyz(q_parent_from_child)
    return R @ I @ R.T


@dataclass
class Inertia_Node:
    """
    Python port of SpaceStationModuleInertia (C++) with NumPy.
    - Quaternions expected as [w, x, y, z].
    - Positions in meters; masses in kg.
    - Inertia tensors in kg*m^2.
    """
    name_: str = ""
    parent_: "Inertia_Node | None" = None

    # public (to mirror C++):
    inertia_wrt_parent: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=float))

    # private-like:
    _children: List["Inertia_Node"] = field(default_factory=list)
    _parent_distance_parent: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    _q_parent_from_child: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))  # identity
    _points_mxyz: List[Tuple[float, float, float, float]] = field(default_factory=list)  # [(m,x,y,z), ...]

    _total_mass: float = 0.0
    _com: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    _inertia_com: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=float))

    # ---------- API parity with C++ ----------

    def load_points_(self, m: Sequence[float], p: Sequence[Sequence[float]]) -> None:
        if len(m) == 0 or len(p) == 0:
            raise ValueError("Inputs must be non-empty.")
        if len(m) != len(p):
            raise ValueError("masses and positions must have the same length.")

        pts: List[Tuple[float, float, float, float]] = []
        for mi, pi in zip(m, p):
            if mi <= 0.0:
                raise ValueError("Each mass must be positive.")
            x, y, z = _as_np3(pi)
            pts.append((float(mi), float(x), float(y), float(z)))

        self._points_mxyz = pts

    def calculate_module_inertia(self) -> None:
        self._compute_total_mass_and_com_()
        self._compute_inertia_about_com_()

    def calculate_module_inertia_wrt_parent(self) -> None:
        if self._total_mass <= 0.0:
            raise RuntimeError("total mass not set. Call calculate_module_inertia first.")

        # rotate inertia about COM into parent frame
        I_com_parent = rotate_inertia(self._inertia_com, self._q_parent_from_child)

        # subtree mass (self + all descendants)
        M = self._calculate_subtree_mass_()

        dx, dy, dz = self._parent_distance_parent
        d2 = float(dx*dx + dy*dy + dz*dz)

        # Parallel-axis term: M * (||d||^2 I - d d^T)
        ddT = np.array([[dx*dx, dx*dy, dx*dz],
                        [dy*dx, dy*dy, dy*dz],
                        [dz*dx, dz*dy, dz*dz]], dtype=float)
        PA = M * (d2 * np.eye(3) - ddT)

        # Parent-aligned inertia at parent origin
        self.inertia_wrt_parent = I_com_parent + PA

    def add_module_child(self, child_module: "Inertia_Node") -> None:
        child_module.parent_ = self
        self._children.append(child_module)

    def set_parent_offset(self, r: Sequence[float]) -> None:
        self._parent_distance_parent = _as_np3(r)

    def set_q_parent_from_child(self, q_wxyz: Sequence[float]) -> None:
        self._q_parent_from_child = _normalize_quat(q_wxyz)

    def has_children(self) -> bool:
        return len(self._children) > 0

    def children(self) -> List["Inertia_Node"]:
        return list(self._children)

    # ---------- Internals (ported 1:1 in spirit) ----------

    def _calculate_subtree_mass_(self) -> float:
        m = self._total_mass
        for ch in self._children:
            m += ch._calculate_subtree_mass_()
        return float(m)

    def _compute_total_mass_and_com_(self) -> None:
        M = 0.0
        cx = cy = cz = 0.0
        for (mi, x, y, z) in self._points_mxyz:
            M += mi
            cx += mi * x
            cy += mi * y
            cz += mi * z

        if M == 0.0:
            raise RuntimeError("Total mass is zero.")

        self._total_mass = float(M)
        self._com = np.array([cx / M, cy / M, cz / M], dtype=float)

    def _compute_inertia_about_com_(self) -> None:
        I = np.zeros((3, 3), dtype=float)
        c = self._com

        # point-mass inertia about COM
        for (mi, x, y, z) in self._points_mxyz:
            rx, ry, rz = x - c[0], y - c[1], z - c[2]
            r2 = rx*rx + ry*ry + rz*rz
            # Add p.m * (r^2 I - r r^T)
            I[0, 0] += mi * (r2 - rx*rx)
            I[0, 1] -= mi * (rx*ry)
            I[0, 2] -= mi * (rx*rz)

            I[1, 0] -= mi * (ry*rx)
            I[1, 1] += mi * (r2 - ry*ry)
            I[1, 2] -= mi * (ry*rz)

            I[2, 0] -= mi * (rz*rx)
            I[2, 1] -= mi * (rz*ry)
            I[2, 2] += mi * (r2 - rz*rz)

        # NOTE: Mirrors your C++: add children tensors that are already
        # "in parent frame about parent origin" directly here.
        for child in self._children:
            I += _as_np33(child.inertia_wrt_parent)

        self._inertia_com = I

    # ---------- Convenience getters ----------

    @property
    def total_mass(self) -> float:
        return float(self._total_mass)

    @property
    def com(self) -> np.ndarray:
        return self._com.copy()

    @property
    def inertia_about_com(self) -> np.ndarray:
        return self._inertia_com.copy()
    

if __name__ == '__main__':

    pass

