import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, PillowWriter
from gates import X, H, apply_gate, bloch_coords

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
    ax.plot([-1, 1], [0, 0], [0, 0], color="black")
    ax.plot([0, 0], [-1, 1], [0, 0], color="black")
    ax.plot([0, 0], [0, 0], [-1, 1], color="black")
    ax.text(0, 0, 1.1, r"$|0\rangle$", ha="center")
    ax.text(0, 0, -1.2, r"$|1\rangle$", ha="center")

# -------------------------
# Main animation
# -------------------------
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")

# Initial state
theta = np.pi / 3
phi = np.pi / 4
psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
v0 = np.array(bloch_vector(theta, phi))   # red arrow

# Final state after Hadamard
psi_new = apply_gate(psi, H)
v1 = np.array(bloch_coords(psi_new))      # blue arrow

def update(frame):
    ax.clear()
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-1.2,1.2])
    ax.set_ylim([-1.2,1.2])
    ax.set_zlim([-1.2,1.2])
    ax.axis("off")

    plot_bloch_sphere(ax)

    # Draw red arrow (static)
    ax.quiver(0,0,0, v0[0], v0[1], v0[2], color="red", linewidth=2, arrow_length_ratio=0.1)

    # Draw blue arrow (interpolating)
    t = frame / 50
    vec = (1 - t) * v0 + t * v1
    ax.quiver(0,0,0, vec[0], vec[1], vec[2], color="blue", linewidth=2, arrow_length_ratio=0.1)

ani = FuncAnimation(fig, update, frames=51, interval=50, blit=False)
ani.save("bloch_rotation.gif", writer=PillowWriter(fps=20))
plt.show()
