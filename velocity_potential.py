# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from typing import Tuple

# Globals - To be set in main (we abuse the dont-touch-globals rule in this code, can wrap using extra module)
OUTPUT_PATH = "/Users/home/Documents/Studies/Semester_B/Mechanics of Continua/Numeric/output/"

X_RANGE = (-0.5, 0.5)
Y_RANGE = (-0.5, 0.5)
N = 100
M = 100

SOURCE = True
WING_RATIO = 0.1
DELTA = 1.0
INCLINE_ANGLE = 0.01 * (0.5 * pi)
U = 1.0

MAX_ITERATIONS = 20000
MAX_ERROR = 1e-6
BLOWN_ERROR = 100.0


def get_xy_mesh(cartesian=False):
    """Returns a tuple of matrices (X, Y) where X[i,j] = x coordinate at the index (i,j) and same for Y"""
    # Note - since we want both start and end points we need N + 1 or M + 1 to hold boundary conditions
    # (we divide space to N pieces so we have N + 1 values of x)
    X = np.linspace(X_RANGE[0], X_RANGE[1], N + 1, endpoint=True)
    Y = np.linspace(Y_RANGE[0], Y_RANGE[1], M + 1, endpoint=True)
    if cartesian: return np.meshgrid(X, Y, indexing='xy')
    return np.meshgrid(X, Y, indexing='ij')


def infinity_boundary(phi):
    Ux = U * np.cos(INCLINE_ANGLE)
    Uy = U * np.sin(INCLINE_ANGLE)
    dx = abs(X_RANGE[1] - X_RANGE[0]) / N
    dy = abs(Y_RANGE[1] - Y_RANGE[0]) / M

    # Left
    phi[0, :] = phi[1, :] - Ux * dx
    phi[0, 1:] = phi[0, :-1] + Uy * dy
    # Right
    phi[-1, :] = phi[-2, :] + Ux * dx
    phi[-1, 1:] = phi[-1, :-1] + Uy * dy
    # Up
    phi[1:, -1] = phi[:-1, -1] + Ux * dx
    phi[:, -1] = phi[:, -2] + Uy * dy
    # Down
    phi[1:, 0] = phi[:-1, 0] + Ux * dx
    phi[:, 0] = phi[:, 1] - Uy * dy

    return phi


def wing_boundary_old(phi):
    i0 = int(N / 2)
    j0 = int(M / 2)
    i_max = int(WING_RATIO * N)
    x_inds = np.arange(-i_max + i0, i0 + i_max)
    pos_x = np.arange(i0 + 1, i0 + i_max)

    delta = DELTA / i_max

    # We give finite width of 2 dys

    phi[x_inds, j0:j0 + 3] = np.array([phi[x_inds, j0 - 1]] * 3).transpose()
    phi[pos_x, j0 + 2] += delta

    phi[x_inds[0] - 1, j0:j0 + 2] = np.copy(phi[x_inds[0], j0:j0 + 2])
    phi[x_inds[-1] + 1, j0:j0 + 2] = np.copy(phi[x_inds[-1], j0:j0 + 2])

    return phi


def wing_boundary2(phi):
    i0 = int(N / 2)
    j0 = int(M / 2)
    i_max = int(WING_RATIO * N)
    x_inds = np.arange(-i_max + i0, i0 + i_max + 1)
    pos_x = np.arange(i0 + 1, i0 + i_max + 1)

    delta = DELTA / i_max

    phi[x_inds, j0 + 1] = np.copy(phi[x_inds, j0 + 2])
    phi[x_inds, j0] = np.copy(phi[x_inds, j0 - 1])

    phi[x_inds[0], j0:j0 + 2] = np.copy(phi[x_inds[0] - 1, j0:j0 + 2])
    phi[x_inds[-1], j0:j0 + 2] = np.copy(phi[x_inds[-1] + 1, j0:j0 + 2])

    # jump at wind
    phi[x_inds, j0 + 1] = np.copy(phi[x_inds, j0]) + delta
    return phi


def wing_boundary(phi):
    i0 = int(N / 2)
    j0 = int(M / 2)
    i_max = int(WING_RATIO * N)
    x_inds = np.arange(-i_max + i0, i0 + i_max + 1)
    pos_x = np.arange(i0 + 1, i0 + i_max + 1)

    delta = DELTA / i_max

    phi[x_inds, j0:j0 + 2] = 0
    phi[pos_x, j0 + 1] = delta
    return phi


def impose_boundary(phi):
    # Infinity:
    phi = infinity_boundary(phi)

    # Around the Wing:
    if SOURCE:
        phi = wing_boundary2(phi)
    return phi


def new_grid(phi_old):
    # Set new Internal Points based on phi_old
    #    --- Note ---
    #   We override the boundary conditions at y=0 (around the wing). At the end we impose again B.C.

    phi_new = np.copy(phi_old)
    phi_new[1:-1, 1:-1] = 0.25 * (phi_old[2:, 1:-1] + phi_old[:-2, 1:-1] + phi_old[1:-1, 2:] + phi_old[1:-1, :-2])

    phi_new = impose_boundary(phi_new)

    return phi_new


def one_step(phi, alpha):
    phi_new = new_grid(phi)
    phi_new = phi + alpha * (phi_new - phi)
    return phi_new


def init_phi():
    phi = np.ones(shape=(N + 1, M + 1))
    phi = impose_boundary(phi)  # Check that it doesnt copy back and forth. Otherwise, access by reference
    phi -= np.average(phi)
    return phi


def solve_phi(circulation=1.0, alpha=1.0):
    # Initiation
    phi = init_phi()
    iterations = 0
    error = 10.0

    while error > MAX_ERROR:
        phi_new = one_step(phi, alpha)
        phi_new -= np.average(phi_new)  # Shift back around 0 (avoid big numbers)

        error = np.max(np.abs(phi_new - phi))

        phi = phi_new
        iterations += 1

        # breakpoints
        if iterations >= MAX_ITERATIONS:
            print("Reached maximum iterations")
            break
        elif error >= BLOWN_ERROR:
            print("Blown error")
            break
    return phi, iterations, error


def init_globals(incline=INCLINE_ANGLE, max_iter=MAX_ITERATIONS, wing_length=WING_RATIO, n=N, m=M,
                 circulation=DELTA, xrange=X_RANGE, yrange=Y_RANGE, u=U):
    global N, M, X_RANGE, Y_RANGE, DELTA, WING_RATIO, MAX_ITERATIONS, U, INCLINE_ANGLE

    if wing_length > 1.0 or wing_length < 0.0:
        raise ValueError("Invalid wing length. Should be in [0,1]")

    INCLINE_ANGLE = incline
    N, M = n, m
    X_RANGE, Y_RANGE = xrange, yrange
    DELTA = circulation
    WING_RATIO = wing_length
    MAX_ITERATIONS = max_iter
    U = u
    return


def get_velocity_map(phi):
    dx = abs(X_RANGE[1] - X_RANGE[0]) / N
    dy = abs(Y_RANGE[1] - Y_RANGE[0]) / M
    Vx = (phi[1:, :] - phi[:-1, :]) * dx
    Vy = (phi[:, 1:] - phi[:, :-1]) * dy
    Vx = Vx[:, 1:]
    Vy = Vy[1:, :]

    return Vx, Vy


def draw_wing():
    X, Y = get_xy_mesh()
    i_max = np.min(np.argwhere(X >= WING_RATIO)[:, 0])
    j0 = np.min(np.argwhere(Y >= 0)[:, 1])
    plt.plot([-WING_RATIO, WING_RATIO], [0.0, 0.0])
    plt.plot([-WING_RATIO, WING_RATIO], [1 / M, 1 / M])
    return


def plot_contour(phi, draw_wing_=True, save=False, show=True):
    n_pi = INCLINE_ANGLE / pi
    titl = rf"Velocity Potential $\Phi$ contours: $\Gamma$ = {DELTA}, $\alpha$ = {n_pi} $\pi$"
    X, Y = get_xy_mesh()
    plt.contour(X, Y, phi, levels=40)
    plt.title(titl)
    if draw_wing_:
        draw_wing()
    if save:
        name = f"vel_del_{DELTA}_i_{round(INCLINE_ANGLE, 2)}.png"
        # if ZERO_VX:
        #     name = "noVX_" + name
        # if (FINITE_WIDTH):
        #     name = "fWidth_" + name
        plt.savefig(OUTPUT_PATH + name)
    if show: plt.show()
    return


def plot_velocity(phi):
    X, Y = get_xy_mesh()
    X, Y = X[1:, 1:], Y[1:, 1:]
    Vx, Vy = get_velocity_map(phi)
    plt.quiver(X, Y, Vx, Vy, )
    plt.show()
    return


def plot_velocity2(phi):
    X, Y = get_xy_mesh(cartesian=True)
    X, Y = X[1:, 1:], Y[1:, 1:]
    print(X.shape)

    Vx, Vy = get_velocity_map(phi)
    # print(Vx.shape)

    plt.streamplot(X, Y, Vx, Vy)
    draw_wing()
    plt.show()
    return


def get_stream(phi):
    # x of phi = y of psi, # y of phi = -x of psi
    # psi[i,j] = psi[-j,i]
    psi = np.zeros(shape=phi.shape)
    X, Y = get_xy_mesh()
    i0 = int((N + 1) / 2)
    j0 = int((M + 1) / 2)
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            i_moved = i - i0
            j_moved = j - j0
            psi[i, j] = phi[j0 - j_moved, i]
    return psi


def run(incline=INCLINE_ANGLE, source=SOURCE, save=False, show=True):
    global SOURCE
    SOURCE = source
    init_globals(incline=incline)
    phi, iterations, error = solve_phi()
    print(f'iterations: {iterations}, error: {error}')
    plot_contour(phi, save=save, show=show)
    plot_velocity2(phi)
    return phi


def main():
    phi = run(incline=pi / 4, save=False, show=True)
    # psi = get_stream(phi)
    # plot_contour(psi, save=False,show =True)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
