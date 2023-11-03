import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from ellipsoids import plot_ellipse_from_matrices
from SDP_procedure import ellipsoidal_cadro, generate_data, tau
from robust_optimization import solve_robust_quadratic_loss


def sdp_sigma_mu():
    warnings.filterwarnings("ignore", category=UserWarning)
    mu = 0
    rico = 2
    m_test = 1000
    nb_tries = 50
    ms = (20, 30, 40, 50, 60, 70, 80)
    sigmas = (0.05, 0.2, 0.35, 0.5, 1, 2, 3)
    loss_diffs_median = np.empty((len(ms), len(sigmas)))
    loss_diffs_25 = np.empty((len(ms), len(sigmas)))
    loss_diffs_75 = np.empty((len(ms), len(sigmas)))
    theta_diffs_median = np.empty((len(ms), len(sigmas)))
    theta_diffs_25 = np.empty((len(ms), len(sigmas)))
    theta_diffs_75 = np.empty((len(ms), len(sigmas)))
    for i, m in enumerate(ms):
        for j, sigma in enumerate(sigmas):
            theta_diffs = np.empty(nb_tries)
            loss_diffs = np.empty(nb_tries)
            for _ in range(nb_tries):
                x = np.linspace(-2, 2, m)
                np.random.shuffle(x)
                y = generate_data(x, rico, pdf="normal", mu=mu, sigma=sigma)
                data = np.vstack([x, y])

                # outliers: for m / 7 random points, set y = rico * x +/- 10 * sigma
                indices = np.random.choice(m, size=int(m / 7), replace=False)
                data[1, indices] = rico * data[0, indices] + np.random.choice([-1, 1], size=int(m / 7)) * 10 * sigma

                data_test = np.array([np.linspace(-2, 2, m_test),
                                      generate_data(np.linspace(-2, 2, m_test), rico, pdf="normal", mu=mu,
                                                    sigma=sigma)])
                reporting = ellipsoidal_cadro(data, data_test, tau, plot=False, report=False, mu=0.01, nu=0.8,
                                              scaling_factor_ellipse=1, ellipse_alg="lj")
                theta_diffs[_] = reporting["theta_difference"] / reporting["theta_0"]
                loss_diffs[_] = reporting["loss_difference"] / reporting["test_loss_0"]

            theta_median = np.median(theta_diffs)
            theta25 = np.percentile(theta_diffs, 25)
            theta75 = np.percentile(theta_diffs, 75)
            loss_median = np.median(loss_diffs)
            loss25 = np.percentile(loss_diffs, 25)
            loss75 = np.percentile(loss_diffs, 75)
            print("m = ", m, ", sigma = ", sigma)
            print(
                f"theta difference is {theta_median * 100:.4f} % (percentile 25: {theta25 * 100:.4f}, percentile 75: {theta75 * 100:.4f})")
            print(
                f"loss difference is {loss_median * 100:.4f} % (percentile 25: {loss25 * 100:.4f}, percentile 75: {loss75 * 100:.4f})")
            with open("results.txt", "a") as file:
                file.write(f"m = {m}, sigma = {sigma}\n")
                file.write(f"theta difference is {theta_median * 100:.4f} % (percentile 25: {theta25 * 100:.4f}, "
                           f"percentile 75: {theta75 * 100:.4f})\n ")
                file.write(f"loss difference is {loss_median * 100:.4f} % (percentile 25: {loss25 * 100:.4f}, percentile "
                           f"75: {loss75 * 100:.4f})")
                file.write("(mean loss difference: {:.4f} % +- {:.4f} %)\n".format(np.mean(loss_diffs) * 100,
                                                                                   np.std(loss_diffs) * 100))
                file.write("(mean theta difference: {:.4f} % +- {:.4f} %)\n".format(np.mean(theta_diffs) * 100,
                                                                                    np.std(theta_diffs) * 100))
                file.write("\n")
            theta_diffs_median[i, j] = theta_median
            theta_diffs_25[i, j] = theta25
            theta_diffs_75[i, j] = theta75
            loss_diffs_median[i, j] = loss_median
            loss_diffs_25[i, j] = loss25
            loss_diffs_75[i, j] = loss75

    # make a new subplot for every m
    fig, axs = plt.subplots(np.ceil(np.sqrt(len(ms))).astype(int), np.ceil(np.sqrt(len(ms))).astype(int))
    fig2, axs2 = plt.subplots(np.ceil(np.sqrt(len(ms))).astype(int), np.ceil(np.sqrt(len(ms))).astype(int))
    fig.suptitle("Loss difference as a function of sigma for different m")
    for i, m in enumerate(ms):
        row, col = i // np.ceil(np.sqrt(len(ms))).astype(int), i % np.ceil(np.sqrt(len(ms))).astype(int)
        # plot the loss difference as a function of sigma including the 25th and 75th percentile
        axs[row, col].errorbar(sigmas, loss_diffs_median[i, :], yerr=[abs(loss_diffs_median[i, :] - loss_diffs_25[i, :]),
                                                                      abs(loss_diffs_75[i, :] - loss_diffs_median[i, :])],
                               fmt='o-')
        axs[row, col].set_title(f"m = {m}")
        if col == 0:
            axs[row, col].set_ylabel("loss difference")
        axs[row, col].set_xlabel("sigma")
        axs[row, col].set_xticks(np.linspace(0, max(sigmas), 5))
        axs2[row, col].set_xticklabels(np.linspace(0, max(sigmas), 5))
        axs[row, col].grid()

        # plot the theta difference as a function of sigma
        axs2[row, col].errorbar(sigmas, theta_diffs_median[i, :], yerr=[abs(theta_diffs_median[i, :] - theta_diffs_25[i, :]),
                                                                      abs(theta_diffs_75[i, :] - theta_diffs_median[i, :])],
                                fmt='o-')
        axs2[row, col].set_title(f"m = {m}")
        if col == 0:
            axs2[row, col].set_ylabel("theta difference")
        axs2[row, col].set_xlabel("sigma")
        axs2[row, col].set_xticks(np.linspace(0, max(sigmas), 5))
        axs2[row, col].set_xticklabels(np.linspace(0, max(sigmas), 5))
        axs2[row, col].grid()

    plt.subplots_adjust(hspace=1, wspace=1, left=0.2, right=0.9, top=0.9, bottom=0.1)
    plt.show()

    # save resulting matrices to file
    np.savez("results.npz", loss_diffs_mean=loss_diffs_median, loss_diffs_25=loss_diffs_25, loss_diffs_75=loss_diffs_75,
             theta_diffs_median=theta_diffs_median, theta_diffs_25=theta_diffs_25, theta_diffs_75=theta_diffs_75,
             ms=ms, sigmas=sigmas)


def sdp_with_plots():
    warnings.filterwarnings("ignore", category=UserWarning)
    mu = 0
    rico = 1.5
    m_test = 1000
    m = 30
    sigma = 1
    x = np.linspace(-2, 2, m)
    np.random.shuffle(x)
    y = generate_data(x, rico, pdf="normal", mu=mu, sigma=sigma)
    data = np.vstack([x, y])
    # generate outliers: for m / 7 random points, set y = rico * x +/- 5 * sigma
    indices = np.random.choice(m, size=int(m / 7), replace=False)
    data[1, indices] = rico * data[0, indices] + np.random.choice([-1, 1], size=int(m / 7)) * 5 * sigma
    data_test = np.array([np.linspace(-2, 2, m_test), generate_data(np.linspace(-2, 2, m_test), rico, pdf="normal",
                                                                    mu=mu, sigma=sigma)])

    reporting = ellipsoidal_cadro(data, data_test, tau, plot=False, report=False, mu=0.01, nu=0.8,
                                  scaling_factor_ellipse=1, ellipse_alg="lj")
    A, a, c = reporting["A"], reporting["a"], reporting["c"]
    a = a.value
    theta_r, loss = solve_robust_quadratic_loss(A, a, c)
    theta_star = reporting["theta_star"]
    theta_0 = reporting["theta_0"]
    x_range = (np.min(x), np.max(x))
    plt.figure(0)
    plt.scatter(data[0, :], data[1, :], label="data", marker=".")
    plt.plot(x_range, theta_star * np.array(x_range), label=r"$\theta^*$", color="red")
    plt.plot(x_range, theta_r * np.array(x_range), label=r"$\theta_r$")
    plt.plot(x_range, theta_0 * np.array(x_range), label=r"$\theta_0$", linestyle="--")
    plot_ellipse_from_matrices(A, a, c, theta_0, x_range[1], x_range[0], np.max(y), np.min(y), 100, 100)
    plt.legend()
    plt.grid()
    plt.show()


def robust_problem():
    # test the function
    vertical_axis = 1
    horizontal_axis = 3
    rotation_angle = np.pi / 4
    A = np.array([[- 1 / horizontal_axis ** 2, 0], [0, - 1 / vertical_axis ** 2]])
    R = np.array(
        [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
    A = R.T @ A @ R
    a = np.array([[0], [0]])
    c = np.array([[1]])  # -1 to have a non-empty ellipsoid
    theta, value = solve_robust_quadratic_loss(A, a, c)
    print("theta: ", theta)
    print("optimal value: ", value)
    # plot the ellipsoid
    ellipse = Ellipse(xy=(0, 0), width=2 * horizontal_axis, height=2 * vertical_axis, edgecolor='r', fc='None',
                      angle=np.rad2deg(rotation_angle))
    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    ax.add_artist(ellipse)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid()
    # add the optimal theta
    x = np.linspace(-2, 2, 100)
    y = theta * x
    plt.plot(x, y, label=r"$\theta^*$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # sdp_with_plots()
    sdp_sigma_mu()
    # robust_problem()