import matplotlib.pyplot as plt
import numpy as np


def main():

    dt = 0.001
    m = 1
    k = 0.05

    time_total = 50
    time_arr = np.zeros(int(time_total / dt))
    x = np.zeros(int(time_total / dt), dtype=float)
    v = np.zeros(int(time_total / dt), dtype=float)
    x[0] = 1
    v[0] = 0

    for t in range(1, int(time_total / dt)):
        x[t] = dt * v[t-1] + x[t-1]
        v[t] = dt * (-k * x[t]/m) + v[t-1]
        time_arr[t] = t * dt

    plt.rcParams.update({"font.size": 14})
    fig, ax1 = plt.subplots()
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
    plt.savefig("./results/leapfrog/x_v_k{}.png".format(k))
    plt.show()

    return


if __name__ == '__main__':
    main()
