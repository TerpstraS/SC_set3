import matplotlib.pyplot as plt
import numpy as np
import numba


@numba.njit
def leapfrog(time_total, dt, k, f_t=False, omega=0.01, m=1):
    time_arr = np.zeros(int(time_total / dt))
    x = np.zeros(int(time_total / dt))
    v = np.zeros(int(time_total / dt))
    x[0] = 1
    v[0] = 0

    for t in range(1, int(time_total / dt)):
        x[t] = dt * v[t-1] + x[t-1]
        if f_t:
            v[t] = dt * (np.sin(omega * x[t]) - k * x[t]) / m + v[t-1]
        else:
            v[t] = dt * (-k * x[t]/m) + v[t-1]
        time_arr[t] = t * dt

    return time_arr, x, v


def leapfrog_hooke():
    dt = 0.001
    m = 1
    k = 0.05
    time_total = 50

    time_arr, x, v = leapfrog(time_total, dt, k, f_t=False, m=m)

    plt.figure()
    plt.plot(v, x)

    plt.rcParams.update({"font.size": 14})
    fig, ax1 = plt.subplots()
    plt.title("Hooke")
    color = "black"
    ax1.plot(time_arr, x, color=color)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("x (m)", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()
    color = "blue"
    ax2.plot(time_arr, v, color=color)
    ax2.set_ylabel("v (m/s)", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    fig.tight_layout()
    # plt.savefig("./results/leapfrog/x_v_k{}.png".format(k))


def leapfrog_sinusiodal():

    dt = 0.001
    m = 1
    k = 0.3
    time_total = 100
    omega = 0.1

    time_arr, x, v = leapfrog(time_total, dt, k, f_t=True, omega=omega, m=m)
    #
    plt.figure()
    plt.plot(v, x)

    plt.rcParams.update({"font.size": 14})
    fig, ax1 = plt.subplots()
    plt.title("Omega")
    color = "black"
    ax1.plot(time_arr, x, color=color)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("x (m)", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()
    color = "blue"
    ax2.plot(time_arr, v, color=color)
    ax2.set_ylabel("v (m/s)", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    fig.tight_layout()
    # plt.savefig("./results/leapfrog/sinusiodal_x_v_k{}.png".format(k))


def main():

    leapfrog_hooke()

    leapfrog_sinusiodal()

    plt.show()

    return


if __name__ == '__main__':
    main()
