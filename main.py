import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def construct_M(N_x, N_y):

    M = np.zeros((N_x*N_y, N_x*N_y), dtype=int)

    # loop through each grid point
    for i in range(N_x * N_y):

        # calculate x- and y-coordinate for each grid point
        x_grid = i % N_x
        y_grid = i // N_x

        # determine neighbours and check if they are in the boundaries of the system
        neighbours_candidates = [[1, 0], [-1, 0], [0, -1], [0, 1]]
        neighbours = []
        for neighbours_candidate in neighbours_candidates:
            if neighbours_candidate[0] + x_grid >= 0 and neighbours_candidate[0] + x_grid < N_x \
                and neighbours_candidate[1] + y_grid >= 0 and neighbours_candidate[1] + y_grid < N_y:
                neighbours.append([neighbours_candidate[0] + x_grid, neighbours_candidate[1] + y_grid])

        # loop through i-th row of M
        for j in range(N_x * N_y):

            # place diagonal
            if i == j:
                M[i, j] = -len(neighbours)
            else:
                for neighbour in neighbours:
                    x_neighbour = neighbour[0]
                    y_neighbour = neighbour[1]

                    x_M = x_neighbour + y_neighbour * N_x
                    y_M = i

                    M[x_M, y_M] = 1

    return M


def main():

    N_x = 3
    N_y = 3
    M = construct_M(N_x, N_y)

    print(M)

    return


if __name__ == '__main__':
    main()
