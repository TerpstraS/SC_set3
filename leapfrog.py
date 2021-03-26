import matplotlib.pyplot as plt
import numpy as np
import numba
from scipy.integrate import solve_ivp


def runge_kutta(k, time_total, dt):
    def harmonic_oscillator(t, z, k):
        x, v = z
        return [v, -k * x]

    sol = solve_ivp(harmonic_oscillator, [0, time_total], [1, 0], t_eval=np.linspace(0, time_total, int(time_total / dt)), args=(k, ))


    plt.rcParams.update({"font.size": 14})
    fig, ax1 = plt.subplots()
    plt.title("RK45")
    color = "black"
    ax1.plot(sol.t, sol.y[0], color=color)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("x (m)", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()
    color = "blue"
    ax2.plot(sol.t, sol.y[1], color=color)
    ax2.set_ylabel("v (m/s)", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    fig.tight_layout()

    return sol.t, sol.y[0], sol.y[1]


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


def leapfrog_hooke(k, time_total, dt):
    m = 1

    time_arr, x, v = leapfrog(time_total, dt, k, f_t=False, m=m)

    plt.rcParams.update({"font.size": 14})
    fig, ax1 = plt.subplots()
    # plt.title("Leapfrog")
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
    plt.savefig("./results/leapfrog/leapfrog_dt{}_k{}.png".format(dt, k))

    return time_arr, x, v

def leapfrog_sinusiodal(k, omega, time_total, dt):

    m = 1

    time_arr, x, v = leapfrog(time_total, dt, k, f_t=True, omega=4, m=m)
    #
    # plt.figure()
    # plt.plot(v, x)
    # plt.show()

    plt.rcParams.update({"font.size": 14})
    fig, ax1 = plt.subplots()
    # plt.title("Leapfrog sinusiodal force $\omega={}$".format(omega))
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
    plt.savefig("./results/leapfrog/sinusiodal_x_v_k{}.png".format(k))

    # omegas = np.linspace(0, 10, 5)
    # plt.figure()
    # # plt.title("Phase plot leapfrog sinusiodal force")
    # for omega1 in omegas:
    #     time_arr, x, v = leapfrog(time_total, dt, k, f_t=True, omega=omega1, m=m)
    #     plt.plot(x, v, label="$\omega={}$".format(omega1))
    #
    # plt.xlabel("x (m)")
    # plt.ylabel("v (m/s)")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("results/leapfrog/phase_plots.png")

    return time_arr, x, v

def main():

    dt = 0.01
    k = 2
    time_total = 25
    # time_arr, x_leapfrog, v_leapfrog = leapfrog_hooke(k, time_total, dt)
    # time_arr, x_rk45, v_rk45 = runge_kutta(k, time_total, dt)
    # # print(x_leapfrog)
    # # print(x_rk45)
    # print(np.sum((x_leapfrog - x_rk45)))

    omega = 2.5
    time_arr, x, v = leapfrog_sinusiodal(k, omega, time_total, dt)

    plt.show()

    return


if __name__ == '__main__':
    main()
