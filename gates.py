import numpy as np

# -------------------------
# Single Qubit Gates
# -------------------------
X = np.array([[0, 1],
              [1, 0]], dtype=complex)

Y = np.array([[0, -1j],
              [1j, 0]], dtype=complex)

Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

H = (1/np.sqrt(2)) * np.array([[1, 1],
                               [1, -1]], dtype=complex)

# -------------------------
# Apply a gate
# -------------------------
def apply_gate(psi, gate):
    """
    psi: 2x1 numpy array (complex)
    gate: 2x2 numpy array
    """
    return gate @ psi

# -------------------------
# Convert qubit vector to Bloch coords
# -------------------------
def bloch_coords(psi):
    """
    Convert a 2-component qubit vector to Bloch sphere coordinates (x,y,z)
    """
    alpha, beta = psi
    x = 2 * (alpha.conjugate()*beta).real
    y = 2 * (alpha.conjugate()*beta).imag
    z = abs(alpha)**2 - abs(beta)**2
    return x, y, z

# -------------------------
# SU(2) gate -> SO(3) Bloch rotation
# -------------------------

# Pauli matrices (for mapping to Bloch sphere rotations)
_SX = np.array([[0, 1],
                [1, 0]], dtype=complex)

_SY = np.array([[0, -1j],
                [1j, 0]], dtype=complex)

_SZ = np.array([[1, 0],
                [0, -1]], dtype=complex)

_PAULIS = [_SX, _SY, _SZ]

def bloch_rotation_matrix(U: np.ndarray) -> np.ndarray:
    """
    Convert a 2x2 unitary U (SU(2) up to a global phase)
    into the corresponding 3x3 real rotation matrix R on Bloch vectors.

    For Bloch vector r, the transformed vector is r' = R r.
    """
    U = np.asarray(U, dtype=complex)
    if U.shape != (2, 2):
        raise ValueError(f"U must be 2x2, got shape {U.shape}")

    R = np.zeros((3, 3), dtype=float)
    Udag = U.conjugate().T

    # R_ij = (1/2) Tr( sigma_i U sigma_j U† )
    for i in range(3):
        for j in range(3):
            R[i, j] = 0.5 * np.trace(_PAULIS[i] @ U @ _PAULIS[j] @ Udag).real

    return R

# if __name__ == "__main__":
#     R = bloch_rotation_matrix(X)
#     print("R(X)=\n", R)
#     print("det =", np.linalg.det(R))
#     print("orthogonality check (R^T R):\n", R.T @ R)
