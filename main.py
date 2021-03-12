import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp_lin
import scipy.sparse.linalg as sp_lin_sparse
import time

# def construct_M(N_x, N_y, circular=False):
#
#     # add 2 to each side for the boundary conditions
#     M = np.zeros(((N_x + 2) * (N_y + 2), (N_x + 2) * (N_y + 2)), dtype=int)
#
#     # loop through each grid point
#     for i in range((N_x + 2) * (N_y + 2)):
#
#         # skip boundaries (where the value should be kept 0, since fixed bounday conditions)
#         if i < N_x + 2 or i >= (N_x + 2) * (N_y + 2) - (N_x + 2) or i % (N_x + 2) == 0 or i % (N_x + 2) == N_x + 1:
#             continue
#
#         # calculate x- and y-coordinate for each grid point
#         x_grid = i % (N_x + 2)
#         y_grid = i // (N_x + 2)
#
#         # determine neighbours and check if they are in the boundaries of the system
#         neighbours_candidates = np.array([[1, 0], [-1, 0], [0, -1], [0, 1]])
#         neighbours = np.full((4, 2), -1)
#         for k, neighbours_candidate in enumerate(neighbours_candidates):
#             if neighbours_candidate[0] + x_grid >= 0 and neighbours_candidate[0] + x_grid < N_x + 2 \
#                 and neighbours_candidate[1] + y_grid >= 0 and neighbours_candidate[1] + y_grid < N_y + 2:
#
#                 neighbours[k, 0] = neighbours_candidate[0] + x_grid
#                 neighbours[k, 1] = neighbours_candidate[1] + y_grid
#
#         for neighbour in neighbours:
#             x_neighbour = neighbour[0]
#             y_neighbour = neighbour[1]
#
#             if x_neighbour != -1 and y_neighbour != -1:
#                 x_M = x_neighbour + y_neighbour * (N_x + 2)
#                 y_M = i
#                 M[x_M, y_M] = 1
#
#         # set diagional to -4
#         M[i, i] = -4
#
#     return M

def construct_M(N_x, N_y, circular=False):

    if circular:
        assert N_x == N_y, "grid should be square for a circular domain"

        x_center, y_center = N_x / 2, N_y / 2
        r = N_x / 2

    M = np.zeros(((N_x) * (N_y), (N_x) * (N_y)), dtype=int)

    # loop through each grid point
    for i in range(N_x * N_y):

        # calculate x- and y-coordinate for each grid point
        x_grid = i % N_x
        y_grid = i // N_x

        # skip if grid point lays outside circle
        if circular and np.sqrt((x_center - x_grid)**2 + (y_center - y_grid)**2) > r:
            continue

        # determine neighbours and check if they are in the boundaries of the system
        neighbours_candidates = np.array([[1, 0], [-1, 0], [0, -1], [0, 1]])
        neighbours = np.full((4, 2), -1)
        for k, neighbours_candidate in enumerate(neighbours_candidates):
            if neighbours_candidate[0] + x_grid >= 0 and neighbours_candidate[0] + x_grid < N_x \
                and neighbours_candidate[1] + y_grid >= 0 and neighbours_candidate[1] + y_grid < N_y:

                neighbours[k, 0] = neighbours_candidate[0] + x_grid
                neighbours[k, 1] = neighbours_candidate[1] + y_grid

        n_neighbours = 0
        for neighbour in neighbours:
            x_neighbour = neighbour[0]
            y_neighbour = neighbour[1]

            if x_neighbour != -1 and y_neighbour != -1:
                n_neighbours += 1
                x_M = x_neighbour + y_neighbour * N_x
                y_M = i
                M[x_M, y_M] = 1

        M[i, i] = -4

    return M


def solve_eigen_problem(L, N_x, N_y, circular=False, sparse=True, k=6):

    M = construct_M(N_x, N_y, circular=circular)
    M = -M / (L /N_x**2)   # apply division by dx^2

    print("Solving eigenvalues...")

    if sparse:
        eig_vals, eig_vecs = sp_lin_sparse.eigs(M, k)
    else:
        eig_vals, eig_vecs = sp_lin.eigh(M)

    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.T

    print("Eigenvalues found...\n")

    return eig_vals, eig_vecs


def time_func(t, labda):
    A, B = 1, 0
    return A * np.cos(labda*t) + B * np.sin(labda*t)


def main():

    time_start = time.time()

    L = 1
    N = 30

    N_x = N * L
    N_y = N * L

    eig_vals_square, eig_vecs_square = solve_eigen_problem(L, N*L, N*L, sparse=False, k=2)
    eig_vals_rect, eig_vecs_rect = solve_eigen_problem(L, N*L, N*2*L, sparse=False, k=2)

    # circle doesn't properly work
    eig_vals_circle, eig_vecs_circle = solve_eigen_problem(L, N*L, N*L, sparse=False, circular=True, k=2)

    print("Total time: {:.2f} seconds".format(time.time()-time_start))

    # u = eig_vecs_square[0].reshape((N*L, N*L)).T
    # labda = np.sqrt(eig_vals_square[0])
    # fig, ax = plt.subplots()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # u_t = u * time_func(0, labda)
    # ax.matshow(u_t, origin="lower")
    #
    # for t in np.arange(0, 0.5*np.pi, 0.01):
    #     u_t = u * time_func(t, labda)
    #     plt.cla()
    #     ax.matshow(u_t, origin="lower")
    #     plt.title("$t = {:.2f}$ seconds".format(t))
    #     plt.pause(0.001)

    for i, eig_vec in enumerate(eig_vecs_square[:2]):
        plt.matshow(eig_vec.reshape((N*L, N*L)).T, origin="lower")
        plt.title("Square drum: eigenvalue {:.0f}".format(eig_vals_square[i]))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()

    for i, eig_vec in enumerate(eig_vecs_rect[:2]):
        plt.matshow(eig_vec.reshape(N*L, N*2*L).T, origin="lower")
        plt.title("Rectangular drum: eigenvalue {:.0f}".format(eig_vals_rect[i]))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()

    for i, eig_vec in enumerate(eig_vecs_circle[:2]):
        plt.matshow(eig_vec.reshape(N*L, N*L).T, origin="lower")
        plt.title("Circular drum: eigenvalue {:.0f}".format(eig_vals_circle[i]))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()

    plt.show()

    return


if __name__ == '__main__':
    main()
