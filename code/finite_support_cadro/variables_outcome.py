import warnings
import numpy as np
import matplotlib.pyplot as plt
from ellipsoids import Ellipsoid
from robust_optimization import RobustOptimization
from one_dimension_cadro import CADRO1DLinearRegression
from utils.data_generator import ScalarDataGenerator


def sdp_with_plots(seed=0):
    mu = 0
    rico = 3
    m_test = 1000
    m = 30
    sigma = 1
    ellipse_alg = "princ"
    R = np.array([[np.cos(np.pi / 3), -np.sin(np.pi / 3)], [np.sin(np.pi / 3), np.cos(np.pi / 3)]])
    # R = np.array([[1, 0], [0, 1]])
    lengths = np.array([[1], [1.5]])
    x = np.linspace(-2, 2, m)
    train_data_gen = ScalarDataGenerator(x, seed)
    y = train_data_gen.generate_linear_norm_disturbance(mu, sigma, rico, outliers=True)
    data = np.vstack([x, y])

    x_test = np.linspace(-2, 2, m_test)
    test_data_gen = ScalarDataGenerator(x_test, seed)
    y_test = test_data_gen.generate_linear_norm_disturbance(mu, sigma, rico, outliers=True)
    data_test = np.vstack([x_test, y_test])

    # construct the ellipsoid
    ellipsoid = Ellipsoid.lj_ellipsoid(data, 3)

    # set up CADRO problem
    problem = CADRO1DLinearRegression(data, ellipsoid)
    problem.solve()
    problem.set_theta_r()

    problem.print_results(include_robust=True)
    results = problem.results
    theta_r = results["theta_r"]
    theta_star = results["theta"]
    theta_0 = results["theta_0"]

    plt.figure(0)
    plt.scatter(data[0, :], data[1, :], label="data", marker=".")
    plt.plot(x, theta_r * x, label=r"$\theta_r = {:.4f}$".format(theta_r))
    plt.plot(x, theta_star * x, label=r"$\theta^* = {:.4f}$".format(theta_star), color="green", linestyle="--")
    plt.plot(x, theta_0 * x, label=r"$\theta_0 = {:.4f}$".format(theta_0), linestyle="--")
    problem.ellipsoid.plot(ax=plt.gca(), color="red", label="ellipsoid")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.show()


def robust_or_saa():
    """
    Script used to generate plots displaying the realized values of theta_r, theta_star and theta_0 for different
    values of sigma and m. Also plots the realized values of alpha and lambda for different values of sigma and m.
    """
    mu = 0
    rico = 3
    R = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])

    ms = (20, 50, 100, 200, 300)
    nb_tries = 100
    sigmas = (0.5, 0.75, 1, 1.5, 2, 2.5, 3)
    theta_r_array = np.empty((len(ms), len(sigmas), nb_tries))
    theta_star_array = np.empty((len(ms), len(sigmas), nb_tries))
    theta_0_array = np.empty((len(ms), len(sigmas), nb_tries))
    collapses = np.empty((len(ms), len(sigmas), nb_tries))
    alphas_array = np.empty((len(ms), len(sigmas), nb_tries))
    lambdas_array = np.empty((len(ms), len(sigmas), nb_tries))

    x_test = np.linspace(-2, 2, 50)
    test_data_gen = ScalarDataGenerator(x_test, seed=0)
    y_test = test_data_gen.generate_linear_norm_disturbance(mu, 4, rico, outliers=True)
    data_test = np.vstack([x_test, y_test])
    ellipsoid = Ellipsoid.from_principal_axes(R, data_test)
    for k, m in enumerate(ms):
        for i, sigma in enumerate(sigmas):
            for j in range(nb_tries):
                x = np.linspace(-2, 2, m)
                np.random.shuffle(x)
                data_gen = ScalarDataGenerator(x, seed=0)
                data_gen.generate_linear_norm_disturbance(mu, sigma, rico, outliers=True)
                data_gen.contain_within_ellipse(ellipsoid)
                y = data_gen.__y
                data = np.vstack([x, y])
                problem = CADRO1DLinearRegression(data, ellipsoid)
                results = problem.solve()
                theta_0 = results["theta_0"]
                theta_star = results["theta"]
                robust_problem = RobustOptimization(ellipsoid)
                robust_results = robust_problem.solve_1d_linear_regression()
                theta_robust = robust_results["theta"]
                theta_r_array[k, i, j] = theta_robust
                theta_star_array[k, i, j] = theta_star
                theta_0_array[k, i, j] = theta_0
                collapses[k, i, j] = np.abs(theta_0 - theta_star) > np.abs(theta_robust - theta_star)
                alphas_array[k, i, j] = results["alpha"]
                lambdas_array[k, i, j] = results["lambda"]

            # make boxplots for theta_r, theta_star and theta_0
            plt.figure()
            nb_collapses = np.mean(collapses[k, i, :])
            plt.title(f"sigma = {sigma}, m = {m}. Collapse rate: {nb_collapses * 100:.2f} %")
            # plot a cross for every theta_r, theta_star and theta_0
            plt.scatter(np.ones(nb_tries), theta_r_array[k, i, :], marker="x", label=r"$\theta_r$")
            plt.scatter(2 * np.ones(nb_tries), theta_star_array[k, i, :], marker="x", label=r"$\theta^*$")
            plt.scatter(3 * np.ones(nb_tries), theta_0_array[k, i, :], marker="x", label=r"$\theta_0$")
            plt.xticks([1, 2, 3], [r"$\theta_r$", r"$\theta^*$", r"$\theta_0$"])
            plt.legend()
            plt.xlim(0, 4)
            plt.grid()
            plt.savefig(f"figures/rotated ellipse/princ_thetas_sigma_{sigma}_m_{m}.png")

            # get the indices of lambdas which are close to 0 (between -0.1 and 0.1)
            plt.figure()
            ind_lambdas = np.where(np.abs(lambdas_array[k, i, :]) < 0.1)[0]
            frac_lambdas = len(ind_lambdas) / nb_tries
            plt.title(f"sigma = {sigma}, m = {m}. Collapse rate: {nb_collapses * 100:.2f} %")
            # plot a cross for every lambda and alpha
            # every alpha corresponding to a lambda close to 0 is plotted in red, otherwise in blue
            plt.scatter(np.ones(len(ind_lambdas)), alphas_array[k, i, ind_lambdas], marker='x', label=r"$\alpha_r$", color="r")
            plt.scatter(np.ones(nb_tries - len(ind_lambdas)), alphas_array[k, i, np.delete(np.arange(nb_tries), ind_lambdas)], marker='x', color='b', label=r"$\alpha_0$")
            plt.scatter(2 * np.ones(nb_tries), lambdas_array[k, i, :], marker="x", label=r"$\lambda$")
            # add a text label with the fraction of lambdas close to 0
            plt.text(2, 0.1, f"{frac_lambdas * 100:.2f} %", horizontalalignment="center", verticalalignment="center")
            plt.text(2, 0.9, f"{(1 - frac_lambdas) * 100:.2f} %", horizontalalignment="center",
                     verticalalignment="center")
            plt.xticks([1, 2], [r"$\alpha$", r"$\lambda$"])
            plt.xlim(0, 3)
            plt.legend()
            plt.grid()
            plt.savefig(f"figures/rotated ellipse/princ_alphas_lambdas_sigma_{sigma}_m_{m}.png")

            plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    # sdp_with_plots()
    robust_or_saa()
