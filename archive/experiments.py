

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

