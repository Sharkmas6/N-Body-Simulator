import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation


"""
Get trajectory of n-body system attracted by gravitational force in 3D (x,y,z)
through solving system of 2n 2nd-order ODEs.
"""


class Body:
    """ class to hold details of a certain body """

    def __init__(self, m, *coords0, ndim=2, name=None):
        assert len(coords0) // 2 != 0, "Must give even number of initial conditions (position, speed)" \
                                       "in vector form"

        self.mass = m
        self.pos, self.vel = [[] for i in range(ndim)], [[] for i in range(ndim)]
        mid = int(len(coords0) / 2)
        self.pos0, self.vel0 = [*coords0[:mid]], [*coords0[mid:]]
        self.ndim, self.name = ndim, name

    def __str__(self):
        return f"Name: {self.name}\n" \
               f"Mass: {self.mass}\n" \
               f"Initial Position: {self.pos0}\n" \
               f"Initial Velocity: {self.vel0}"

    def reset_trajectory(self):
        self.pos, self.vel = [[] for i in range(self.ndim)], [[] for i in range(self.ndim)]
        return self


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
                                         sep=sep,axis=axis, ndim=ndim)

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

    return m, pos0 + vel0, pos + vel


def new_body(m, *args, name=None):
    ndim = int(len(args) / 2)
    body = Body(m, *args, ndim=ndim, name=name)
    bodies.append(body)
    return body


# create bodies, timescale and constants
bodies = []
#       (m, x, y, z, vx, vy, vz)
new_body(1, 0, 2, 0, 0, 1, 0)
new_body(1, 3, 0, 1, 1, 0, 1)

step = 0.1
last_t = 50
time = np.arange(0, last_t, step)

G = 1  # attraction constant
I = 3 * G  # repulsion constant
dimensions = 3

# extract data from bodies objects in usable form
masses, init, coords = bodies_to_coords(*bodies)

for i in bodies:
    print(i)

# solve equations of motion
ans = odeint(motion_nbody, init, time, args=(masses, G, I, dimensions, True))


# PLOT 2D
fig = plt.figure()
ax = fig.gca(projection="3d")

mid = int(len(ans) / 2)
ax.set_xlim(ans[:, 0:mid:2].min() - 0.2, ans[:, 0:mid:2].max() + 0.2)
ax.set_ylim(ans[:, 1:mid:2].min() - 0.2, ans[:, 1:mid:2].max() + 0.2)
ax.set_zlim(ans[:, 2:mid:2].min() - 0.2, ans[:, 3:mid:2].max() + 0.2)
time_template = "Time = %.1fs"
time_text = ax.text(0, 0, 1, "", transform=ax.transAxes, bbox=dict(edgecolor="k", fill="w"))

lines, trackers = [], []
for n in range(len(bodies)):
    dummmies = [[] for j in range(dimensions)]
    initial = [ans[0, dimensions * n + j] for j in range(dimensions)]

    lines.append(ax.plot(*dummmies, label=f"Trajectory {n}", alpha=0.5)[0])
    ax.scatter(*initial, label=f"Start {n}", linewidths=masses[n])
    trackers.append(ax.plot(*dummmies, "o", label=f"Body {n}", c=lines[n].get_color(), linewidth=masses[n])[0])


def animate(i, ndim):
    for l, line, track in zip(range(len(lines)), lines, trackers):

        segs = [ans[:i, dimensions * l + n] for n in range(ndim)]
        tr_segs = [ans[i, dimensions * l + n] for n in range(ndim)]

        line.set_data_3d(*segs)
        track.set_data_3d(*tr_segs)

    time_text.set_text(time_template % (i * step))

    return lines, trackers, #time_text


ani = FuncAnimation(fig, animate, ans.shape[0], fargs=(dimensions,), interval=step * 500, blit=False)

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title(f"{len(bodies)}-Body Orbital Trajectory with repulsion")

plt.show()
