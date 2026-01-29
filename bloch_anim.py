import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, PillowWriter

from gates import H, apply_gate, bloch_coords, bloch_rotation_matrix


def bloch_vector(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def plot_bloch_sphere(ax):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

    # axes
    ax.plot([-1, 1], [0, 0], [0, 0], color="black")
    ax.plot([0, 0], [-1, 1], [0, 0], color="black")
    ax.plot([0, 0], [0, 0], [-1, 1], color="black")

    # labels
    ax.text(0, 0, 1.1, r"$|0\rangle$", ha="center")
    ax.text(0, 0, -1.2, r"$|1\rangle$", ha="center")


def rotate_about_axis(axis, angle):
    """
    Rodrigues' rotation formula: rotate a 3D vector around 'axis' by 'angle' radians.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=float)

    I = np.eye(3)
    return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# -------------------------
# Setup figure
# -------------------------
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

# Initial state
theta = np.pi / 3
phi = np.pi / 4
psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])

# Initial Bloch vector (red)
v0 = np.array(bloch_vector(theta, phi), dtype=float)

# Final state after Hadamard (for reference)
psi_new = apply_gate(psi, H)
v1 = np.array(bloch_coords(psi_new), dtype=float)

# -------------------------
# Physics-correct rotation from the gate
# -------------------------
R = bloch_rotation_matrix(H)

trail = []   # stores past tip positions of the evolving vector

# angle from trace: tr(R) = 1 + 2 cos(angle)
angle_total = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))

# axis from antisymmetric part
axis = np.array([
    R[2, 1] - R[1, 2],
    R[0, 2] - R[2, 0],
    R[1, 0] - R[0, 1]
], dtype=float)

# handle rare case: angle ~ 0 (no rotation)
if np.linalg.norm(axis) < 1e-9:
    axis = np.array([1.0, 0.0, 0.0])
else:
    axis = axis / np.linalg.norm(axis)


def update(frame):
    ax.clear()
    if frame == 0:
        trail.clear()
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.axis("off")

    plot_bloch_sphere(ax)

    # Draw red arrow (initial state)
    ax.quiver(0, 0, 0, v0[0], v0[1], v0[2],
              color="red", linewidth=2, arrow_length_ratio=0.1)

    # Rotate v0 toward its final position under the gate
    t = frame / 50.0
    Rt = rotate_about_axis(axis, t * angle_total)
    
    vec = Rt @ v0

    # --- trail: store the tip position ---
    trail.append(vec.copy())

    # --- trail: plot the path so far ---
    trail_arr = np.array(trail)
    ax.plot(trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2],
            color="blue", linewidth=1, alpha=0.6)

    # Draw blue arrow (evolving state)
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2],
            color="blue", linewidth=2, arrow_length_ratio=0.1)


    # Optional: label
    ax.text(1.05, 0, 0, "Initial", color="red")
    ax.text(1.05, 0, 0.1, "Evolving", color="blue")


ani = FuncAnimation(fig, update, frames=51, interval=50, blit=False)

# Save GIF
ani.save("bloch_rotation.gif", writer=PillowWriter(fps=20))

plt.show()
