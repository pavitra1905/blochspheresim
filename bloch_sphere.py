import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from gates import X, H, apply_gate, bloch_coords


# -------------------------
# Physics: qubit definition
# -------------------------
def bloch_vector(theta, phi):
    """
    Convert qubit parameters (theta, phi)
    to Bloch sphere coordinates.
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


# -------------------------
# Visualization functions
# -------------------------
def plot_bloch_sphere(ax):
    """
    Draws the Bloch sphere wireframe, axes, and |0>, |1> labels.
    """
    # Sphere coordinates
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Draw wireframe
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

    # Draw axes
    ax.plot([-1, 1], [0, 0], [0, 0], color="black")
    ax.plot([0, 0], [-1, 1], [0, 0], color="black")
    ax.plot([0, 0], [0, 0], [-1, 1], color="black")

    # Labels
    ax.text(0, 0, 1.1, r"$|0\rangle$", ha="center")
    ax.text(0, 0, -1.2, r"$|1\rangle$", ha="center")


def plot_state_vector(ax, x, y, z, color="red"):
    """
    Draw a single vector (arrow) on the Bloch sphere.
    """
    ax.quiver(
        0, 0, 0,
        x, y, z,
        color=color,
        linewidth=2,
        arrow_length_ratio=0.1
    )


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Create figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.axis("off")

    # Initial state |ψ⟩ with theta, phi
    theta = np.pi / 3
    phi = np.pi / 4
    psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])

    # Apply a Hadamard gate
    psi_new = apply_gate(psi, H)

    # Plot Bloch sphere
    plot_bloch_sphere(ax)

    # Original state (red)
    x0, y0, z0 = bloch_vector(theta, phi)
    plot_state_vector(ax, x0, y0, z0, color="red")

    # New state after gate (blue)
    x1, y1, z1 = bloch_coords(psi_new)
    plot_state_vector(ax, x1, y1, z1, color="blue")

    # Optional legend text
    ax.text(1.1, 0, 0, "Original", color="red")
    ax.text(1.1, 0, 0.1, "After H", color="blue")

    plt.show()
