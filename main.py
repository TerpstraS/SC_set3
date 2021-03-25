from matplotlib import cm
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp_lin
import scipy.sparse.linalg as sp_lin_sparse
import scipy.sparse as sp_sparse
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable


def construct_M(L, N_x, N_y, circular=False, source=None):
    """ This is a function that makes the matrix of a rectangle. """

    len_M = N_x * N_y
    M = np.zeros((len_M, len_M))

    if circular:
        assert N_x == N_y, "grid should be square for a circular domain"

        x_center, y_center = N_x / 2, N_y / 2

        r = N_x / 2

    # index in M of diagonals of 1. Large is the outer diagonal, small is the diagonal hugging
    # the true matrix diagonal
    M_large_diag = N_x
    M_small_diag = 1

    for i in range(len_M):

        # set values at diagonal to -4
        M[i, i] = -4

        # fill in outermost diagonals
        if M_large_diag >= 0 and M_large_diag < len_M:
            M[i, M_large_diag] = 1
            M[M_large_diag, i] = 1

        # fill in the diagonal hugging the true diagonal
        if not M_small_diag % N_x == 0:
            if M_small_diag >= 0 and M_small_diag < len_M:
                M[i, M_small_diag] = 1
                M[M_small_diag, i] = 1

        M_large_diag += 1
        M_small_diag += 1

    # step over matrix, check if the column is in the circle
    if circular:
        for i in range(len_M):

            # true grid coordinates
            x_grid = i // N_x
            y_grid = i % N_x

            # check if grid point lays outside circle
            if np.sqrt((x_center - x_grid)**2 + (y_center - y_grid)**2) > r:

                # set column to zero (except for the diagonal)
                for j in range(len_M):
                    if not i == j:
                        M[i, j] = 0
                    else:
                        M[i, j] = 1

    M *= (1/(L/N_x)**2)

    if source:

        for i in range(len_M):

            # true grid coordinates
            x_grid = i // N_x
            y_grid = i % N_x
            x_source, y_source = source
            if x_grid == x_source and y_grid == y_source:

                # set column to zero (except for the diagonal)
                for j in range(len_M):
                    if not i == j:
                        M[i, j] = 0
                    else:
                        M[i, j] = 1

    # multiply M by dx
    return M


def solve_eigen_problem(L, N_x, N_y, circular=False, sparse=True, k=6):

    M = construct_M(L, N_x, N_y, circular)

    print("Solving eigenvalues...")

    if sparse:
        eig_vals, eig_vecs = sp_lin_sparse.eigs(M, k=k, which="SM")     # or which="LR"
    else:
        eig_vals, eig_vecs = sp_lin.eig(M)

    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.real.T

    # sort eig_vals and eig_vecs in ascending order
    if not sparse:
        ids = np.flip(np.argsort(eig_vals))
        eig_vals = eig_vals[ids]
        eig_vecs = eig_vecs[ids]

    print("Eigenvalues found...\n")

    return eig_vals, eig_vecs


def time_func(t, labda):
    A, B = 1, 1
    return A * np.cos(labda*t) + B * np.sin(labda*t)

def animate_wave_equation(u, labda):
    u_t = u * time_func(0, labda)
    X = np.arange(0, len(u_t))
    Y = np.arange(0, len(u_t))
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlim(-0.1, 0.1)
    surf = ax.plot_surface(X, Y, u_t, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    # caxes = ax.matshow(u_t, origin='lower')

    # initialization function: plot the background of each frame
    def init():
        line = ax.matshow([])
        return line,

    def animate(t):
        u_t = u * time_func(t, labda)
        plt.cla()
        ax.set_zlim(-0.1, 0.1)
        surf = ax.plot_surface(X, Y, u_t, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
        # return line,

    animation = ani.FuncAnimation(fig, animate, frames=np.arange(0, 0.3*np.pi, 0.01), repeat=True, save_count=700)
    return animation


def diffusion():

    time_start = time.time()

    sparse = True
    N = 100
    r = 2
    x_source, y_source = r + 0.6, r + 1.2
    L = 4
    x_grid_source = int((N / L) * x_source)
    y_grid_source = int((N / L) * y_source)
    M = construct_M(L, N, N, circular=True, source=(x_grid_source, y_grid_source))


    b = np.zeros((N, N))
    b[x_grid_source, y_grid_source] = 1
    b = b.reshape(N * N)

    if sparse:
        c = sp_lin_sparse.spsolve(sp_sparse.csr_matrix(M), b)
    else:
        c = sp_lin.solve(M, b)

    print("Total time: {:.2f} seconds".format(time.time()-time_start))

    plt.rcParams.update({"font.size": 14})
    plt.matshow(c.reshape((N, N)).T, origin="lower", extent=[-2, 2, -2, 2])
    plt.gca().xaxis.tick_bottom()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    # plt.tight_layout()
    plt.savefig("results/diffusion_N{}.png".format(N))
    plt.show()

    return


def plot_drum(N_x, N_y, L, eig_vals, eig_vecs):

    fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
    plt.rcParams.update({"font.size": 14})
    im = None
    for i, ax in enumerate(axs.flatten()):
        # plt.matshow(eig_vec.reshape((N*L, N*L)).T, origin="lower")
        # plt.title("Square drum: eigenvalue {:.0f}".format(eig_vals_square[i]))
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.colorbar()
        im = ax.matshow(eig_vecs[i*2].reshape((N_x*L, N_y*L)).T, origin="lower", extent=[0, L, 0, L])
        ax.set_title("$\lambda = {:.2f}$".format(np.sqrt(-eig_vals[i*2])))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='10%', pad=0.1)
        fig.colorbar(im, cax=cax, orientation="vertical")

    # fig.colorbar(im, ax=axs.ravel().tolist())
    axs[1, 0].xaxis.tick_bottom()
    axs[1, 0].set_xlabel("x")
    axs[1, 1].xaxis.tick_bottom()
    axs[1, 1].set_xlabel("x")
    axs[1, 0].yaxis.tick_left()
    axs[1, 0].set_ylabel("y")
    axs[0, 0].yaxis.tick_left()
    axs[0, 0].set_ylabel("y")
    fig.tight_layout(pad=0)
    plt.savefig("results/drums_circle.png")

    plt.show()


def main():

    # diffusion()
    # return

    time_start = time.time()

    L = 1
    N = 100
    sparse = True
    k = 8

    N_x = N * L
    N_y = N * L

    # eig_vals_square, eig_vecs_square = solve_eigen_problem(L, N*L, N*L, sparse=sparse, k=k)


    # eig_vals_rect, eig_vecs_rect = solve_eigen_problem(L, N*L, N*2*L, sparse=sparse, k=k)
    eig_vals_circle, eig_vecs_circle = solve_eigen_problem(L, N*L, N*L, sparse=sparse, circular=True, k=k)
    plot_drum(N*L, N*L, L, eig_vals_circle, eig_vecs_circle)
    print("Total time: {:.2f} seconds".format(time.time()-time_start))
    return


    # animation
    u = eig_vecs_circle[0].reshape((N*L, N*L)).T
    labda = np.sqrt(-eig_vals_circle[0])

    print('Animating')

    anim = animate_wave_equation(u, labda)
    anim.save('animation.mp4')

    print('Done animating')


    for i, eig_vec in enumerate(eig_vecs_square[:k]):
        plt.matshow(eig_vec.reshape((N*L, N*L)).T, origin="lower")
        plt.title("Square drum: eigenvalue {:.0f}".format(eig_vals_square[i]))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()

    for i, eig_vec in enumerate(eig_vecs_rect[:k]):
        plt.matshow(eig_vec.reshape(N*L, N*2*L).T, origin="lower")
        plt.title("Rectangular drum: eigenvalue {:.0f}".format(eig_vals_rect[i]))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()

    for i, eig_vec in enumerate(eig_vecs_circle[:k]):
        plt.matshow(eig_vec.reshape(N*L, N*L).T, origin="lower")
        plt.title("Circular drum: eigenvalue {:.0f}".format(eig_vals_circle[i]))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()

    plt.show()

    return


if __name__ == '__main__':
    main()
