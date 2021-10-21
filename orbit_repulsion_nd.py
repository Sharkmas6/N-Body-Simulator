import numpy as np
import matplotlib.pyplot as plt
from orbit.bodies_creator import *
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation


"""
Get trajectory of n-body system attracted by gravitational force in 2D (x,y)
through solving system of 2n 2nd-order ODEs.
"""


def grav_2body(g, m_other, *coords, sep=0, axis=0, ndim=2):
    """ calculate acceleration of 'self' caused by attraction of 'other' on k axis"""
    # eg. x1, y1, z1, x2, y2, z2
    mid = len(coords) / 2
    assert mid == ndim, "This function should only be used between 2 bodies"

    r2 = 0
    for i in np.arange(ndim):
        new = (coords[i + ndim] - coords[i]) ** 2

        if new > 10 ** - 16:
            r2 += new

    if r2 < 10 ** - 16:
        return 0

    r = np.sqrt(r2)  # r squared

    sep_func = np.exp(-r)

    return m_other * (g / r2 - sep * sep_func) * (coords[axis + ndim] - coords[axis])


def motion_nbody(coords, t, masses, g, sep=0, ndim=2, organise=True):
    """ find trajectory and speeds of each body """
    # coords eg. x1, y1, z1, x2, y2, z2, dx1, dy1, dz1, dx2, dy2, dz2
    f = []
    v_start = int(len(coords) / 2)
    nbodies = int(v_start / ndim)

    # for each body
    for nbody in range(nbodies):
        body = nbody * ndim

        # for each coordinate
        for axis in range(ndim):
            f.append(coords[v_start + body + axis])  # axis_nbody, eg. x1, y2, z4

            total_grav = 0
            for other in ndim * np.arange(nbodies)[np.arange(nbodies) != nbody]:
                coord_pairs = np.hstack((coords[body: body + ndim], coords[other: other + ndim]))
                total_grav += grav_2body(g, masses[int(other / ndim)], *coord_pairs,
                                         sep=sep, axis=axis, ndim=ndim)

            f.append(total_grav)  # eg. dx1, dy2, dz4

    # have form x1, dx1, y1, dy1, z1, dz1, x2, dx2, y2, dy2, z2, dz2

    if organise:
        f = f[::2] + f[1::2]  # organise order to be same as input

    return np.array(f)


def bodies_to_coords(*bodies):
    """ function to transform data from bodies objects to organised list """
    m, pos, vel = [], [], []
    pos0, vel0 = [], []

    for body in bodies:
        m.append(body.mass)
        for i, j, k, l in zip(body.pos, body.vel, body.pos0, body.vel0):
            pos.append(i)
            vel.append(j)
            pos0.append(k)
            vel0.append(l)

    return np.array(m), np.array(pos0 + vel0), np.array(pos + vel)


print("INITIATING SIMULATION")

# create bodies, timescale and constants
bodies = []
#       (m, x, y, vx, vy)
new_body(1, 3, 0, 0, 1, storage=bodies)
new_body(10, 0, 0, 0, 0, storage=bodies)
new_body(1, *(np.random.uniform(-1, 1, 4) * 3), storage=bodies, name="Random")  # random body
print(bodies[-1])

print("1 - Bodies created")

# timeframe
step = 0.1
last_t = 100
time = np.arange(0, last_t, step)

# set up constants
G = 0.5
R = 0  # 5 * G
dimensions = 2

print(f"2 - {dimensions}D simulation with G={G} and R={R}")

# extract data from bodies objects in usable form
masses, init, coords = bodies_to_coords(*bodies)


# solve equations of motion
ans = odeint(motion_nbody, init, time, args=(masses, G, R, dimensions, True))

# centre of mass
mid = 2 * len(bodies)
M = sum(masses)  # sum of all masses
masses_reshape = np.repeat(masses.reshape((1, -1)), len(time), axis=0)
"""
com_x = np.sum(ans[:, :mid:2] * masses_reshape, axis=1) / M  # (x * m) / M
com_y = np.sum(ans[:, 1:mid:2] * masses_reshape, axis=1) / M  # (y * m) / M

find all centre of mass positions, more intensive calculations needed
mean error to method bellow in magnitude of e-14
"""

com_x0 = np.sum(init[:mid:2] * masses) / M  # (x0 * m) / M
com_y0 = np.sum(init[1:mid:2] * masses) / M  # (y0 * m) / M

com_vx0 = np.sum(init[mid::2] * masses) / M  # (vx * m) / M
com_vy0 = np.sum(init[mid+1::2] * masses) / M  # (vy * m) / M

com_x = np.zeros_like(ans[:, 0])
com_x[0], com_x[1:] = com_x0, com_vx0 * step
com_x = np.cumsum(com_x)
com_y = np.zeros_like(ans[:, 1])
com_y[0], com_y[1:] = com_y0, com_vy0 * step
com_y = np.cumsum(com_y)
print("3 - Trajectory found")

# energies
ke = 0.5 * masses_reshape * np.sqrt(ans[:, mid::2] ** 2 + ans[:, mid+1::2] ** 2)  # kinetic

pe = np.ones_like(ke)  # potential energy, compute pairs and sum up for each particle
ix = np.arange(len(bodies))
for i in ix:
    #print("I ##", i)
    i_coords = ans[:, 2*i:2*i + 2]  # [xi, yi]
    #print(i_coords[:5])

    shape = (int(last_t / step), len(bodies) - 1)
    energies = np.ones(shape)

    for counter, j in enumerate(ix[ix != i]):
        #print("J", j)
        j_coords = ans[:, 2*j: 2*j + 2]  # [xj, yj]
        #print(j_coords[:5])

        r = np.sqrt((j_coords[:, 1] - i_coords[:, 1]) ** 2 + (j_coords[:, 0] - i_coords[:, 0]) ** 2)

        eg = G * masses[i] * masses[j] / r

        energies[:, counter] = eg

        counter += 1

        #print("R", r[:5])

    pe[:, i] = np.sum(energies, axis=1)


ans[:, :mid:2] -= np.repeat(com_x.reshape((-1, 1)), len(bodies), axis=1)
ans[:, 1:mid:2] -= np.repeat(com_y.reshape((-1, 1)), len(bodies), axis=1)

# PLOT 2D
fig, (ax, bar_ax) = plt.subplots(1, 2, figsize=(12, 4))

mid = int(len(ans) / 2)
ax.set_xlim(ans[:, :mid:2].min() - 0.2, ans[:, :mid:2].max() + 0.2)
ax.set_ylim(ans[:, 1:mid:2].min() - 0.2, ans[:, 1:mid:2].max() + 0.2)
time_template = "Time = %.1fs"
time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes, bbox=dict(edgecolor="k", fill="w"))

bar_labels = [f"Body {i+1}" for i in range(len(bodies))]
bar_ax.set_ylim(0, ke.max() + 0.2)
bar_ax.set_xticks(np.arange(len(bodies)))
bar_ax.set_xticklabels(bar_labels)


lines, trackers, colors = [], [], []
for n in range(len(bodies)):
    lines.append(ax.plot([], [], alpha=0.5)[0])
    colors.append(lines[n].get_color())
    ax.scatter(ans[0, 2 * n], ans[0, 2 * n + 1], c=colors[n], linewidths=masses[n])
    trackers.append(ax.plot([], [], "o", label=f"Body {n+1}", c=colors[n], linewidth=masses[n])[0])

com = ax.plot(0, 0, "*", c="k", label="Center of Mass")[0]  # initial CoM

bar_ax.bar(np.arange(len(bodies)), ke[0], color=colors)


def animate(i):
    for l, line, track in zip(range(len(lines)), lines, trackers):
        line.set_data(ans[:i, dimensions * l], ans[:i, dimensions * l + 1])
        track.set_data(ans[i, dimensions * l], ans[i, dimensions * l + 1])

    for y, rect in zip(ke[i], bar_ax.containers[0]):
        rect.set_height(y)  # update height

    #com.set_data(com_x[i], com_y[i])  # update center of mass

    time_text.set_text(time_template % ((i + 1) * step))  # update text

    return [*lines, *trackers, *bar_ax.containers[0], com, time_text]


ani = FuncAnimation(fig, animate, ans.shape[0], interval=step * 500, blit=True, repeat_delay=5000)

ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_title(f"{len(bodies)}-Body Orbital Trajectory with repulsion")

ax.legend()

bar_ax.set_ylabel("Energy (J)")
bar_ax.set_title("Energy Variation")

plt.show()
