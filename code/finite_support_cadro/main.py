import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from ellipsoids import plot_ellipse_from_matrices, plot_circle_from_matrices
from SDP_procedure import ellipsoidal_cadro, generate_data, tau
from robust_optimization import solve_robust_quadratic_loss


def sdp_sigma_m():
    warnings.filterwarnings("ignore", category=UserWarning)
    mu = 0
    rico = 2
    m_test = 1000
    nb_tries = 50
    ms = (10, 20, 30, 40, 50)
    sigmas = (0.1, 0.5, 0.75, 1, 3)
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
                                              scaling_factor_ellipse=1, ellipse_alg="circ")
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
                f"theta difference is {theta_median * 100:.4f} % (percentile 25: {theta25 * 100:.4f}, percentile 75: "
                f"{theta75 * 100:.4f})")
            print(
                f"loss difference is {loss_median * 100:.4f} % (percentile 25: {loss25 * 100:.4f}, percentile 75: "
                f"{loss75 * 100:.4f})")
            with open("results_circ.txt", "a") as file:
                file.write(f"m = {m}, sigma = {sigma}\n")
                file.write(f"theta difference is {theta_median * 100:.4f} % (percentile 25: {theta25 * 100:.4f}, "
                           f"percentile 75: {theta75 * 100:.4f})\n ")
                file.write(
                    f"loss difference is {loss_median * 100:.4f} % (percentile 25: {loss25 * 100:.4f}, percentile "
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
        axs[row, col].errorbar(sigmas, loss_diffs_median[i, :],
                               yerr=[abs(loss_diffs_median[i, :] - loss_diffs_25[i, :]),
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
        axs2[row, col].errorbar(sigmas, theta_diffs_median[i, :],
                                yerr=[abs(theta_diffs_median[i, :] - theta_diffs_25[i, :]),
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

    fig.savefig("figures/loss_diffs_circ.png")
    fig2.savefig("figures/theta_diffs_circ.png")
    plt.show()

    # save resulting matrices to file
    np.savez("results_circ.npz", loss_diffs_mean=loss_diffs_median, loss_diffs_25=loss_diffs_25, loss_diffs_75=loss_diffs_75,
             theta_diffs_median=theta_diffs_median, theta_diffs_25=theta_diffs_25, theta_diffs_75=theta_diffs_75,
             ms=ms, sigmas=sigmas)


def sdp_with_plots():
    mu = 0
    rico = 3
    m_test = 1000
    m = 10
    sigma = 1
    ellipse_alg = "circ"
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
                                  scaling_factor_ellipse=1, ellipse_alg=ellipse_alg)
    A, a, c = reporting["A"], reporting["a"], reporting["c"]
    a = a.value
    theta_r, loss = solve_robust_quadratic_loss(A, a, c)
    theta_star = reporting["theta_star"]
    theta_0 = reporting["theta_0"]
    alpha = reporting["alpha"]
    lambda_ = reporting["lambda"]
    print("theta_r = ", theta_r)
    print("theta_star = ", theta_star)
    print("theta_0 = ", theta_0)
    print("alpha = ", alpha)
    print("lambda = ", lambda_)
    x_range = (np.min(x), np.max(x))

    if ellipse_alg == "lj":
        plt.figure(0)
        plt.scatter(data[0, :], data[1, :], label="data", marker=".")
        plt.plot(x_range, theta_r * np.array(x_range), label=r"$\theta_r = {:.4f}$".format(theta_r))
        plt.plot(x_range, theta_star * np.array(x_range), label=r"$\theta^* = {:.4f}$".format(theta_star),
                 color="green", linestyle="--")
        plt.plot(x_range, theta_0 * np.array(x_range), label=r"$\theta_0 = {:.4f}$".format(theta_0), linestyle="--")
        plot_ellipse_from_matrices(A, a, c, theta_0, x_range[1], x_range[0], np.max(y), np.min(y), 200, 300)
        plt.legend()
        plt.grid()
    elif ellipse_alg == "circ":
        fig, ax = plt.subplots()
        ax.scatter(data[0, :], data[1, :], label="data", marker=".")
        ax.plot(x_range, theta_r * np.array(x_range), label=r"$\theta_r = {:.4f}$".format(theta_r))
        ax.plot(x_range, theta_star * np.array(x_range), label=r"$\theta^* = {:.4f}$".format(theta_star), color="green",
                linestyle="--")
        ax.plot(x_range, theta_0 * np.array(x_range), label=r"$\theta_0 = {:.4f}$".format(theta_0), linestyle="--")
        plot_circle_from_matrices(ax, A, a, c)
        ax.legend()
        ax.grid()
    plt.axis("equal")
    plt.savefig("circ_alldata.png")
    plt.show()


def robust_or_saa():
    """
    Script used to generate boxplots for theta_r, theta_star and theta_0 for different values of sigma. The boxplots
    are saved in the folder figures/sigma_collapse. It also showcases the collapse rate for different values of sigma,
    i.e. the percentage of times theta_star is closer to theta_r than to theta_0.
    """
    mu = 0
    rico = 3
    m_test = 1000
    m = 30
    nb_tries = 50
    sigmas = (0.1, 0.5, 0.75, 1, 1.5, 2, 2.5, 3)
    theta_r = np.empty((len(sigmas), nb_tries))
    theta_star = np.empty((len(sigmas), nb_tries))
    theta_0 = np.empty((len(sigmas), nb_tries))
    nb_collapses = np.empty((len(sigmas)))
    for i, sigma in enumerate(sigmas):
        for j in range(nb_tries):
            x = np.linspace(-2, 2, m)
            np.random.shuffle(x)
            y = generate_data(x, rico, pdf="normal", mu=mu, sigma=sigma)
            data = np.vstack([x, y])
            # generate outliers: for m / 7 random points, set y = rico * x +/- 5 * sigma
            indices = np.random.choice(m, size=int(m / 7), replace=False)
            data[1, indices] = rico * data[0, indices] + np.random.choice([-1, 1], size=int(m / 7)) * 5 * sigma
            data_test = np.array([np.linspace(-2, 2, m_test), generate_data(np.linspace(-2, 2, m_test), rico,
                                                                            pdf="normal", mu=mu, sigma=sigma)])
            results = ellipsoidal_cadro(data, data_test, tau, plot=False, report=False, mu=0.01, nu=0.8,
                                        scaling_factor_ellipse=1, ellipse_alg="lj")
            theta_robust, _ = solve_robust_quadratic_loss(results["A"], results["a"].value, results["c"])
            theta_r[i, j] = theta_robust
            theta_star[i, j] = results["theta_star"]
            theta_0[i, j] = results["theta_0"]
        # if theta_star is closer to theta_r than to theta_0, then the solution collapsed
        nb_collapses[i] = np.sum(np.abs(theta_star[i, :] - theta_r[i, :]) < np.abs(theta_star[i, :] - theta_0[i, :]))
        nb_collapses[i] /= nb_tries
        print("sigma = ", sigma)
        print("collapse rate: ", nb_collapses[i])
        print("median theta_star: ", np.median(theta_star[i, :]))

        # make boxplots for theta_r, theta_star and theta_0
        fig, axs = plt.subplots(1, 3)
        fig.suptitle(f"sigma = {sigma}. Collapse rate: {nb_collapses[i] * 100:.2f} %")
        axs[0].boxplot(theta_r[i, :])
        axs[0].set_title(r"$\theta_r$")
        axs[1].boxplot(theta_star[i, :])
        axs[1].set_title(r"$\theta^*$")
        axs[2].boxplot(theta_0[i, :])
        axs[2].set_title(r"$\theta_0$")
        # fig.savefig(f"figures/theta_boxplots_lj/boxplots_sigma_{sigma}.png")
        plt.show()




if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    sdp_with_plots()
    # sdp_sigma_m()
    # robust_or_saa()