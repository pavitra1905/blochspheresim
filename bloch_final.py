import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from gates import X, Y, Z, H, apply_gate, bloch_coords, bloch_rotation_matrix


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


def psi_from_angles(theta, phi):
    return np.array([np.cos(theta / 2),
                     np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def rotate_about_axis(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=float)
    I = np.eye(3)
    return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def axis_angle_from_R(R):
    # angle from trace
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ], dtype=float)

    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / np.linalg.norm(axis)

    return axis, angle


# -------------------------
# State (mutable globals for simplicity)
# -------------------------
psi = None              # current qubit state
target_psi = None       # after gate
animating = False

gate_queue = []          # list of (name, matrix)
queued_text = None       # text showing queued gates
play_requested = False

frames = 51
trail = []

history = []   # stack of previous psi states for Revert

# animation parameters
anim_axis = np.array([1.0, 0.0, 0.0])
anim_angle_total = 0.0


# -------------------------
# Figure setup
# -------------------------
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

plt.subplots_adjust(bottom=0.35)  # room for UI

def setup_axes():
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.axis("off")


setup_axes()
plot_bloch_sphere(ax)

# initial from sliders defaults
theta0 = np.pi / 3
phi0 = np.pi / 4
psi = psi_from_angles(theta0, phi0)

# artists we redraw
arrow_initial = None
arrow_current = None
trail_line = None

def fmt_complex(z, nd=3):
    a = np.round(z.real, nd)
    b = np.round(z.imag, nd)
    sign = "+" if b >= 0 else "-"
    return f"{a}{sign}{abs(b)}i"

def redraw_scene(v_initial=None, v_current=None, show_trail=True):
    global arrow_initial, arrow_current, trail_line, info_text
    """Clear and redraw everything."""
    ax.clear()
    setup_axes()
    plot_bloch_sphere(ax)

        # Re-create info panel each redraw (ax.clear() removes it)
    info_text = ax.text2D(
        0.02, 0.98, "", transform=ax.transAxes,
        va="top", ha="left"
    )

        # Queue display (top-right)
    global queued_text
    queued_text = ax.text2D(
        0.98, 0.98, "", transform=ax.transAxes,
        va="top", ha="right"
    )


    if v_initial is not None:
        arrow_initial = ax.quiver(0, 0, 0, v_initial[0], v_initial[1], v_initial[2],
                                  color="red", linewidth=2, arrow_length_ratio=0.1)

    if show_trail and len(trail) >= 2:
        tarr = np.array(trail)
        trail_line, = ax.plot(tarr[:, 0], tarr[:, 1], tarr[:, 2],
                              color="blue", linewidth=1, alpha=0.6)

    if v_current is not None:
        arrow_current = ax.quiver(0, 0, 0, v_current[0], v_current[1], v_current[2],
                                  color="blue", linewidth=2, arrow_length_ratio=0.1)

        # Update info panel (current psi + bloch coords)
    if psi is not None:
        alpha, beta = psi
        x, y, z = bloch_coords(psi)
        p0 = abs(alpha)**2
        p1 = abs(beta)**2

        info_text.set_text(
            "State |ψ⟩\n"
            f"α = {fmt_complex(alpha)}\n"
            f"β = {fmt_complex(beta)}\n\n"
            "Probabilities\n"
            f"P(0) = {p0:.3f}\n"
            f"P(1) = {p1:.3f}\n\n"
            "Bloch (x, y, z)\n"
            f"x = {x:.3f}\n"
            f"y = {y:.3f}\n"
            f"z = {z:.3f}\n"
        )


    # Update queue display
    qnames = [name for (name, _) in gate_queue]
    queued_text.set_text("Queue: " + (" → ".join(qnames) if qnames else "(empty)"))

    # small labels
    ax.text(1.05, 0, 0, "Initial", color="red")
    ax.text(1.05, 0, 0.1, "Current", color="blue")

info_text = ax.text2D(
    0.02, 0.98, "", transform=ax.transAxes,
    va="top", ha="left"
)

# initial draw (no trail, just one arrow as "current")
v_now = np.array(bloch_coords(psi), dtype=float)
redraw_scene(v_initial=v_now, v_current=v_now, show_trail=False)


# -------------------------
# Sliders
# -------------------------
ax_theta = plt.axes([0.15, 0.26, 0.7, 0.03])
ax_phi   = plt.axes([0.15, 0.21, 0.7, 0.03])
ax_speed = plt.axes([0.15, 0.16, 0.7, 0.03])


theta_slider = Slider(ax_theta, r"$\theta$", 0.0, np.pi, valinit=theta0)
phi_slider   = Slider(ax_phi, r"$\phi$", 0.0, 2*np.pi, valinit=phi0)
# Speed slider: interval in milliseconds per frame (smaller = faster)
speed_slider = Slider(ax_speed, "Speed", 5, 200, valinit=30, valstep=1)



def on_slider(_):
    global psi, animating, trail

    # if animating, ignore slider changes
    if animating:
        return

    theta = theta_slider.val
    phi = phi_slider.val
    psi = psi_from_angles(theta, phi)

    v = np.array(bloch_coords(psi), dtype=float)
    trail = []
    redraw_scene(v_initial=v, v_current=v, show_trail=False)
    fig.canvas.draw_idle()


theta_slider.on_changed(on_slider)
phi_slider.on_changed(on_slider)



# -------------------------
# Buttons
# -------------------------
btn_w, btn_h = 0.10, 0.045
ax_btnX = plt.axes([0.15, 0.05, btn_w, btn_h])
ax_btnY = plt.axes([0.27, 0.05, btn_w, btn_h])
ax_btnZ = plt.axes([0.39, 0.05, btn_w, btn_h])
ax_btnH = plt.axes([0.51, 0.05, btn_w, btn_h])

ax_btnPlay  = plt.axes([0.63, 0.05, btn_w, btn_h])
ax_btnClear = plt.axes([0.75, 0.05, btn_w, btn_h])

ax_btnRevert = plt.axes([0.87, 0.05, btn_w, btn_h])
bRevert = Button(ax_btnRevert, "Revert")


bX = Button(ax_btnX, "X")
bY = Button(ax_btnY, "Y")
bZ = Button(ax_btnZ, "Z")
bH = Button(ax_btnH, "H")

bPlay = Button(ax_btnPlay, "Play")
bClear = Button(ax_btnClear, "Clear")

def start_gate_animation(gate):
    global psi, target_psi, animating, trail, anim_axis, anim_angle_total

    if animating:
        return

    history.append(psi.copy())

    # current and target states (statevector)
    target_psi = apply_gate(psi, gate)

    # compute Bloch rotation parameters from the gate
    R = bloch_rotation_matrix(gate)
    anim_axis, anim_angle_total = axis_angle_from_R(R)

    # reset trail and mark animating
    trail = []
    animating = True


def enqueue_gate(name, gate):
    gate_queue.append((name, gate))
    fig.canvas.draw_idle()

def on_click_X(_): enqueue_gate("X", X)
def on_click_Y(_): enqueue_gate("Y", Y)
def on_click_Z(_): enqueue_gate("Z", Z)
def on_click_H(_): enqueue_gate("H", H)


bX.on_clicked(on_click_X)
bY.on_clicked(on_click_Y)
bZ.on_clicked(on_click_Z)
bH.on_clicked(on_click_H)

def on_click_Play(_):
    global play_requested
    play_requested = True

def on_click_Clear(_):
    global gate_queue, play_requested
    gate_queue = []
    play_requested = False
    fig.canvas.draw_idle()

def on_click_Revert(_):
    global psi, history, animating, trail, gate_queue, play_requested
    if animating:
        return  # don’t undo mid-animation

    if not history:
        return

    psi = history.pop()
    trail = []
    gate_queue = []
    play_requested = False
    fig.canvas.draw_idle()

bRevert.on_clicked(on_click_Revert)


bPlay.on_clicked(on_click_Play)
bClear.on_clicked(on_click_Clear)


# -------------------------
# Animation loop
# -------------------------
def update(frame):
    global psi, animating, trail

    v0 = np.array(bloch_coords(psi), dtype=float)

    if not animating:
        # If Play was pressed, and there are gates queued, start next one
        global play_requested
        if play_requested and gate_queue:
            name, gate = gate_queue.pop(0)
            start_gate_animation(gate)
            # redraw will happen next frames as animation runs
        redraw_scene(v_initial=v0, v_current=v0, show_trail=False)
        return


    # during animation: rotate current Bloch vector by fraction of the gate rotation
    t = frame / (frames - 1)
    Rt = rotate_about_axis(anim_axis, t * anim_angle_total)
    v = Rt @ v0

    trail.append(v.copy())
    redraw_scene(v_initial=v0, v_current=v, show_trail=True)

    # finish on last frame
    if frame == frames - 1:
        psi = target_psi
        animating = False
        if not gate_queue:
            play_requested = False

        


ani = FuncAnimation(fig, update, frames=frames, interval=int(speed_slider.val), blit=False)

def on_speed_change(val):
    # Matplotlib stores the timer in ani.event_source
    ani.event_source.interval = int(speed_slider.val)

speed_slider.on_changed(on_speed_change)

plt.show()

