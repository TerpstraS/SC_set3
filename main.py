import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def jacobi_iteration_matrix(grid):
    N_x = len(grid)
    N_y = len(grid[0])
    M = np.zeros((N_x*N_y, N_x*N_y))

    for i in range(N_x):
        for j in range(N_y):

            # M_row = np.zeros(N*N)

            # calculate neighbours
            neighbours_candidates = [(1, 0), (-1, 0), (0, -1), (0, 1)]
            neighbours = []
            for neighbours_candidate in neighbours_candidates:
                if neighbours_candidate[0] + i >= 0 and neighbours_candidate[0] + i < N_x \
                    and neighbours_candidate[1] + j >= 0 and neighbours_candidate[1] + j< N_y:
                    neighbours.append(neighbours_candidate)

            print(neighbours)
            # place diagonal
            M[N_y*i + j, N_y*i + j] = -len(neighbours)

            for neighbour in neighbours:
                M[i*N_x + j, j + neighbour[0] + N_x*neighbour[1]] = 1

            print(M)
    # for i in range(len(M)):
    #     for j in range(len(M[0])):
    #         x = j // len(grid)
    #         y = j % len(grid[0])
    #
    #         # calculate neighbours
    #         neighbours_candidates = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    #         neighbours = []
    #         for neighbours_candidate in neighbours_candidates:
    #             if neighbours_candidate[0] >= 0 and neighbours_candidate[0] < N_x \
    #                 and neighbours_candidate[1] >= 0 and neighbours_candidate[1] < N_y:
    #                 neighbours.append(neighbours_candidate)
    #
    #         # place diagonal
    #         if i == j:
    #             M[i, j] = -len(neighbours)
    #         else:
    #             if
    #         # print(x, y)

    return M


def main():

    x = 3
    y = 3
    grid = np.zeros((x, y))
    matrix = jacobi_iteration_matrix(grid)

    print(matrix)

    # L = 1
    # dx = 1/4
    # dy = 1/4
    # N_x = int(1 / dx)
    # N_y = int(1 / dy)
    #
    # M = np.zeros((N_x*N_y, N_x*N_y))
    #
    # for i in range(N_x):
    #     for j in range(N_y):
    #
    #         # M_row = np.zeros(N*N)
    #
    #         # calculate neighbours
    #         neighbours_candidates = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    #         neighbours = []
    #         for neighbours_candidate in neighbours_candidates:
    #             if neighbours_candidate[0] >= 0 and neighbours_candidate[0] < N_x \
    #                 and neighbours_candidate[1] >= 0 and neighbours_candidate[1] < N_y:
    #                 neighbours.append(neighbours_candidate)
    #
    #         # place diagonal
    #         M[N_x*i + j, N_x*i + j] = -len(neighbours)
    #
    # print(M)

    return


if __name__ == '__main__':
    main()
