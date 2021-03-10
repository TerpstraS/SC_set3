import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp_lin
import time


def construct_M(N_x, N_y):

    # add 2 to each side for the boundary conditions
    M = np.zeros(((N_x + 2) * (N_y + 2), (N_x + 2) * (N_y + 2)), dtype=int)

    # loop through each grid point
    for i in range((N_x + 2) * (N_y + 2)):

        # skip boundaries (where the value should be kept 0, since fixed bounday conditions)
        if i < N_x + 2 or i >= (N_x + 2) * (N_y + 2) - (N_x + 2) or i % (N_x + 2) == 0 or i % (N_x + 2) == N_x + 1:
            continue

        # calculate x- and y-coordinate for each grid point
        x_grid = i % (N_x + 2)
        y_grid = i // (N_x + 2)

        # determine neighbours and check if they are in the boundaries of the system
        neighbours_candidates = np.array([[1, 0], [-1, 0], [0, -1], [0, 1]])
        neighbours = np.full((4, 2), -1)
        for k, neighbours_candidate in enumerate(neighbours_candidates):
            if neighbours_candidate[0] + x_grid >= 0 and neighbours_candidate[0] + x_grid < N_x + 2 \
                and neighbours_candidate[1] + y_grid >= 0 and neighbours_candidate[1] + y_grid < N_y + 2:

                neighbours[k, 0] = neighbours_candidate[0] + x_grid
                neighbours[k, 1] = neighbours_candidate[1] + y_grid

        for neighbour in neighbours:
            x_neighbour = neighbour[0]
            y_neighbour = neighbour[1]

            if x_neighbour != -1 and y_neighbour != -1:
                x_M = x_neighbour + y_neighbour * (N_x + 2)
                y_M = i
                M[x_M, y_M] = 1

        # set diagional to -4
        M[i, i] = -4

    return M


def eig_vector_to_grid(vector, N_x, N_y):

    grid = np.zeros((N_x + 2, N_y + 2))
    for i in range(len(vector)):
        x_grid = i % (N_x + 2)
        y_grid = i // (N_x + 2)

        grid[x_grid, y_grid] = vector[i]

    return grid


def main():

    L = 1
    N = 25

    N_x = N * L
    N_y = N * L
    M = construct_M(N_x, N_y)
    # M = M / N**2
    # print(M)

    plt.imshow(M)
    plt.colorbar()
    plt.show()
    # return
    # print(M)
    # return

    time_start = time.time()
    print("Solving eigenvalues...")

    # M v = K v
    Ks, vs = sp_lin.eigh(M)
    # eig, eigs, eigh

    print("Finished")
    print("Total time: {:.2f} seconds".format(time.time()-time_start))

    # print(Ks)
    # print(np.sort(Ks))

    print(vs[1])
    grid = eig_vector_to_grid(vs[0], N_x, N_y)
    plt.matshow(grid.T, origin="lower")
    plt.colorbar()
    plt.show()

    return


if __name__ == '__main__':
    main()
