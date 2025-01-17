# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from typing import Tuple

# Globals - To be set in main (we abuse the dont-touch-globals rule in this code, can wrap using extra module)
OUTPUT_PATH = "/Users/home/Documents/Studies/Semester_B/Mechanics of Continua/Numeric/output/stream_potential/"

X_RANGE = (-0.5, 0.5)
Y_RANGE = (-0.5, 0.5)
N = 100
M = 100

SOURCE = True
WING_RATIO = 0.2 * (X_RANGE[1] - X_RANGE[0])
CIRCULATION = 10.0
INCLINE_ANGLE = 0.25 * pi
U = 1.0

MAX_ITERATIONS = 20000
MAX_ERROR = 1e-6
BLOWN_ERROR = 100.0

# global Psi_S (source term)
PSI_S = None  # This way we ensure we access it only after initiation


def get_xy_mesh():
    """Returns a tuple of matrices (X, Y) where X[i,j] = x coordinate at the index (i,j) and same for Y"""
    # Note - since we want both start and end points we need N + 1 or M + 1 to hold boundary conditions
    # (we divide space to N pieces so we have N + 1 values of x)
    X = np.linspace(X_RANGE[0], X_RANGE[1], N + 1, endpoint=True)
    Y = np.linspace(Y_RANGE[0], Y_RANGE[1], M + 1, endpoint=True)
    return np.meshgrid(X, Y, indexing='ij')


def infinity_boundary(psi1):
    Ux = U * np.cos(INCLINE_ANGLE)
    Uy = U * np.sin(INCLINE_ANGLE)
    dx = abs(X_RANGE[1] - X_RANGE[0]) / N
    dy = abs(Y_RANGE[1] - Y_RANGE[0]) / M
    # Ux:
    #   Left, Right
    psi1[0, :-1] = psi1[0, 1:] + (PSI_S[0, 1:] - PSI_S[0, :-1]) - Ux * dy
    psi1[-1, :-1] = psi1[-1, 1:] + (PSI_S[-1, 1:] - PSI_S[-1, :-1]) - Ux * dy
    #   Up, Down:
    psi1[:, -1] = psi1[:, -2] - (PSI_S[:, -1] - PSI_S[:, -2]) + Ux * dy
    psi1[:, 0] = psi1[:, 1] + (PSI_S[:, 1] - PSI_S[:, 0]) - Ux * dy

    # Uy:
    #  Left, Right
    psi1[0, :] = psi1[1, :] + (PSI_S[1, :] - PSI_S[0, :]) + Uy * dx
    psi1[-1, :] = psi1[-2, :] - (PSI_S[-1, :] - PSI_S[-2, :]) - Uy * dx
    #   Up, Down:
    psi1[:-1, -1] = psi1[1:, -1] + (PSI_S[1:, -1] - PSI_S[:-1, -1]) + Uy * dx
    psi1[:-1, 0] = psi1[1:, 0] + (PSI_S[1:, 0] - PSI_S[:-1, 0]) + Uy * dx

    return psi1


def wing_boundary1(psi1):
    #   Wing length:
    X, Y = get_xy_mesh()
    i_max = np.min(np.argwhere(X >= WING_RATIO)[:, 0])
    j0 = np.min(np.argwhere(Y >= 0)[:, 1])

    # Vx = 0 On sides
    left_x = -i_max - 1
    left_val = 0.5 * (psi1[left_x, j0 + 2] + PSI_S[left_x, j0 + 2] + psi1[left_x, j0 - 1] + PSI_S[left_x, j0 - 1])
    psi1[left_x, j0 - 1:j0 + 2] = left_val - PSI_S[left_x, j0 - 1:j0 + 2]

    right_x = i_max + 1
    right_val = 0.5 * (psi1[right_x, j0 + 2] + PSI_S[right_x, j0 + 2] + psi1[right_x, j0 - 1] + PSI_S[right_x, j0 - 1])
    psi1[right_x, j0 - 1:j0 + 2] = left_val - PSI_S[left_x, j0 - 1:j0 + 2]

    # We want (d/dx)Psi_total = 0, i.e. it's constant on this line segment. We make it the average of end points
    psi_tot = psi1 + PSI_S
    psi1[left_x:right_x+1, j0:j0+2] = 0.5*(psi_tot[left_x-1,j0:j0+2] + psi_tot[right_x+1, j0:j0+2])\
                                      - PSI_S[left_x:right_x+1, j0:j0+2]


    return psi1


def wing_boundary2(psi1):
    psi_tot = np.copy(psi1 + PSI_S)
    #   Wing n_length:
    n_length = int(WING_RATIO * N)
    X, Y = get_xy_mesh()

    i0 = int(N/2)
    i_left = i0 -n_length
    i_right = i0 + n_length

    j0 = int(M/2)
    j_up = j0 + 1
    j_down = j0-1

    i_s = range(i_left, i_right+1)
    j_s =range(j_down, j_up+1)

    # Vy = 0 at top and bottom
    val_down = psi_tot[i_left-1, j_down] + psi_tot[i_right+1, j_down]
    val_up =  psi_tot[i_left-1, j_up] + psi_tot[i_right+1, j_up]
    psi_tot[i_s, j_down:j_up+1] = 0.5 * (psi_tot[i_left-1, j_s] + psi_tot[i_right+1, j_s])

    psi1 = psi_tot - PSI_S
    return psi1


def wing_boundary(psi1):
    
    psi_tot = np.copy(psi1 + PSI_S)
    #   Wing n_length:
    n_length = int(WING_RATIO * N)

    i0 = int(N/2)
    i_left = i0 -n_length
    i_right = i0 + n_length

    j0 = int(M/2)
    j_up = j0 + 1
    j_down = j0-1

    i_s = range(i_left, i_right+1)
    j_s = range(j_down, j_up+1)

    for i in i_s:
        for j in j_s:
            psi_tot[i,j] = 0.1
    psi1 = psi_tot - PSI_S
    return psi1


def impose_boundary(psi1):
    assert (PSI_S is not None)
    # First Note,
    #   The Boundary condition -
    #   B(Psi(x,y)) = B(Psi1 + Psi_S) = 0:
    #   Since Dirichlet and Neumann conditions are linear in psi -
    #   B(Psi1) = -B(Psi_S)

    # Infinity:
    psi1 = infinity_boundary(psi1)

    # Around the Wing:
    if SOURCE:
        psi1 = wing_boundary(psi1)
    return psi1


def new_grid1(psi1_old):
    # Set new Internal Points based on psi1_old
    #    --- Note ---
    #   We override the boundary conditions at y=0 (around the wing). At the end we impose again B.C.

    psi1_new = np.copy(psi1_old)
    psi1_new[1:-1, 1:-1] = 0.25 * (psi1_old[2:, 1:-1] + psi1_old[:-2, 1:-1] + psi1_old[1:-1, 2:] + psi1_old[1:-1, :-2])

    psi1_new = impose_boundary(psi1_new)

    return psi1_new
def new_grid(psi1_old):
    # Set new Internal Points based on psi1_old
    #    --- Note ---
    #   We override the boundary conditions at y=0 (around the wing). At the end we impose again B.C.
    psi_tot = psi1_old + PSI_S
    psi_tot_new = np.copy(psi_tot)
    psi_tot_new[1:-1, 1:-1] = 0.25 * (psi_tot[2:, 1:-1] + psi_tot[:-2, 1:-1] + psi_tot[1:-1, 2:] + psi_tot[1:-1, :-2])

    psi1_new = psi_tot_new - PSI_S
    psi1_new = impose_boundary(psi1_new)
    return psi1_new

def one_step(psi1_old, alpha):
    psi1_new = new_grid(psi1_old)
    psi1_new = psi1_old + alpha * (psi1_new - psi1_old)
    return psi1_new


def init_psi1():
    psi1 = np.ones(shape=(N + 1, M + 1))
    psi1 = impose_boundary(psi1)  # Check that it doesnt copy back and forth. Otherwise, access by reference
    psi1 -= np.average(psi1)
    return psi1


def init_psi_s():
    """ Sets the GLOBAL PSI_S according to CIRCULATION"""
    # Psi_S(r) = -(gamma/(2*pi)) * ln(r)

    global PSI_S

    if not SOURCE:
        return np.zeros(shape=(N + 1, M + 1))

    (X, Y) = get_xy_mesh()
    PSI_S = -(CIRCULATION / (4 * pi)) * np.log(X ** 2 + Y ** 2)
    maskx = X == 0.0
    masky = Y == 0.0
    mask = np.array([[maskx[i, j] and masky[i, j] for j in range(maskx.shape[1])] for i in range(maskx.shape[0])])
    PSI_S[mask] = 0.0
    PSI_S -= np.average(PSI_S)
    return


def solve_psi1(circulation=1.0, alpha=1.0):
    # Note!
    # Psi_Total  = Psi1 + Psi_s, where Psi_s = Psi of Filament Source.
    # We iterate ONLY over Psi1, and at the end impose modified B.C.

    # Initiation
    init_psi_s()
    psi1 = init_psi1()
    iterations = 0
    error = 10.0

    while error > MAX_ERROR:
        psi1_new = one_step(psi1, alpha)
        psi1_new -= np.average(psi1_new)  # Shift back around 0 (avoid big numbers)

        error = np.max(np.abs(psi1_new - psi1))

        psi1 = psi1_new
        iterations += 1

        # breakpoints
        if iterations >= MAX_ITERATIONS:
            print("Reached maximum iterations")
            break
        elif error >= BLOWN_ERROR:
            print("Blown error")
            break
    return psi1, iterations, error


def init_globals(incline=INCLINE_ANGLE, max_iter=MAX_ITERATIONS, wing_length=WING_RATIO, n=N, m=M,
                 circulation=CIRCULATION, xrange=X_RANGE, yrange=Y_RANGE, u=U):
    global N, M, X_RANGE, Y_RANGE, CIRCULATION, WING_RATIO, MAX_ITERATIONS, U, INCLINE_ANGLE

    if wing_length > 1.0 or wing_length < 0.0:
        raise ValueError("Invalid wing length. Should be in [0,1]")

    INCLINE_ANGLE = incline
    N, M = n, m
    X_RANGE, Y_RANGE = xrange, yrange
    CIRCULATION = circulation
    WING_RATIO = wing_length
    MAX_ITERATIONS = max_iter
    U = u
    init_psi_s()
    return


def plot_contour(psi, draw_wing=True, save=False, show=True):
    n_pi = INCLINE_ANGLE / pi
    titl = rf"Stream function $\Psi$ contours: $\Gamma$ = {CIRCULATION}, $\alpha$ = {n_pi} $\pi$"
    X, Y = get_xy_mesh()
    plt.contour(X, Y, psi, levels=70)
    plt.title(titl)
    if draw_wing:
        X, Y = get_xy_mesh()
        i_max = np.min(np.argwhere(X >= WING_RATIO)[:, 0])
        j0 = np.min(np.argwhere(Y >= 0)[:, 1])
        plt.plot([-WING_RATIO, WING_RATIO], [0.0, 0.0])

    if save:
        name = f"stream_gamma_{CIRCULATION}_i_{INCLINE_ANGLE}.png"
        plt.savefig(OUTPUT_PATH + name)
    if show: plt.show()
    return


def constant_vel_stream():
    # Psi = -Vy * x + Vx * y
    X, Y = get_xy_mesh()
    psi = U * (- np.sin(INCLINE_ANGLE) * X + (np.cos(INCLINE_ANGLE) * Y))
    return psi


def run(incline=INCLINE_ANGLE, source=SOURCE, save=False):
    global SOURCE
    SOURCE = source
    init_globals(incline=incline)
    psi1, iterations, error = solve_psi1()
    print(f'iterations: {iterations}, error: {error}')
    plot_contour(PSI_S + psi1, save=save)

    return psi1


def main():
    run(incline=pi / 4, source=True, save=True)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
