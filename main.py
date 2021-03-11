import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp_lin
import scipy.sparse.linalg as sp_lin_sparse
import time


def construct_M(N_x, N_y):

    # add 2 to each side for the boundary conditions
    M = np.zeros(((N_x) * (N_y), (N_x) * (N_y)), dtype=int)

    # loop through each grid point
    for i in range(N_x * N_y):

        # calculate x- and y-coordinate for each grid point
        x_grid = i % N_x
        y_grid = i // N_x

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


def main():

    L = 1
    N = 50

    N_x = N * L
    N_y = N * L
    M = construct_M(N_x, N_y)
    M = M / (L /N**2)   # apply division by dx^2

    time_start = time.time()
    print("Solving eigenvalues...")

    Ks, vs = sp_lin_sparse.eigs(M, k=10)
    Ks = Ks.real
    vs = vs.real.T
    print(Ks)

    print("Finished")
    print("Total time: {:.2f} seconds".format(time.time()-time_start))

    plt.matshow(vs[0].reshape(N_x, N_y))
    plt.colorbar()
    plt.show()

    return


if __name__ == '__main__':
    main()
