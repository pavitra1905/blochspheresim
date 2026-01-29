import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from gates import bloch_coords


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


def psi_from_angles(theta, phi):
    # |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


# -------------------------
# Figure + 3D axis
# -------------------------
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])
ax.axis("off")

plot_bloch_sphere(ax)

# Initial values
theta0 = np.pi / 3
phi0 = np.pi / 4

psi0 = psi_from_angles(theta0, phi0)
x0, y0, z0 = bloch_coords(psi0)

# Draw initial arrow (we will redraw it on slider updates)
arrow = ax.quiver(0, 0, 0, x0, y0, z0, color="blue", linewidth=2, arrow_length_ratio=0.1)

# -------------------------
# Sliders
# -------------------------
# Make room at bottom
plt.subplots_adjust(bottom=0.22)

ax_theta = plt.axes([0.15, 0.10, 0.7, 0.03])
ax_phi   = plt.axes([0.15, 0.05, 0.7, 0.03])

theta_slider = Slider(ax_theta, r"$\theta$", 0.0, np.pi, valinit=theta0)
phi_slider   = Slider(ax_phi, r"$\phi$", 0.0, 2*np.pi, valinit=phi0)


def update(_):
    global arrow

    theta = theta_slider.val
    phi = phi_slider.val

    psi = psi_from_angles(theta, phi)
    x, y, z = bloch_coords(psi)

    # Remove old arrow and draw a new one (simplest reliable way)
    arrow.remove()
    arrow = ax.quiver(0, 0, 0, x, y, z, color="blue", linewidth=2, arrow_length_ratio=0.1)

    fig.canvas.draw_idle()


theta_slider.on_changed(update)
phi_slider.on_changed(update)

plt.show()
