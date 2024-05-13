import warnings
import numpy as np
import matplotlib.pyplot as plt
from ellipsoids import Ellipsoid
from robust_optimization import RobustOptimization
from one_dimension_cadro import CADRO1DLinearRegression
from utils.data_generator import ScalarDataGenerator
import pandas as pd
from datetime import datetime


def experiment1(seed):
    """
    For an LJ ellipsoid and a circle, plot the realized values of theta_r, theta_star and theta_0 for sigma=1 and
    m=30.
    """
    plt.rcParams.update({'font.size': 15})

    n = 50
    sigma = 1
    rico = 3

    x = np.linspace(0, 1, n)
    data_gen = ScalarDataGenerator(x, seed)
    y = data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
    data = np.vstack([x, y])
    lj_ellipsoid = Ellipsoid.lj_ellipsoid(data, scaling_factor=2)
    circle = Ellipsoid.smallest_enclosing_sphere(data)

    # set up problem
    m = 60
    sigma = 1
    x_train = np.linspace(0, 1, m)
    train_data_gen = ScalarDataGenerator(x_train, seed)

    # 2.1 Löwner-John ellipsoid
    train_data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
    train_data_gen.contain_within_ellipse(lj_ellipsoid)
    data_train = np.vstack([x_train, train_data_gen.y])
    problem = CADRO1DLinearRegression(data_train, lj_ellipsoid)
    problem.solve(nb_theta_0=2)
    problem.set_theta_r()

    results = problem.results
    theta_r = results["theta_r"]
    theta_star = results["theta"]
    theta_1 = results["theta_0"][0]
    theta_2 = results["theta_0"][1]

    plt.figure()
    plt.scatter(data_train[0, :], data_train[1, :], label="data", marker=".")
    plt.plot(x_train, theta_r * x_train, label=r"$\theta_r = {:.2f}$".format(theta_r), linestyle="-", color="blue")
    plt.plot(x_train, theta_star * x_train, label=r"$\theta^* = {:.2f}$".format(theta_star), color="red",
             linestyle="--")
    plt.plot(x_train, theta_1 * x_train, label=r"$\theta_1 = {:.2f}$".format(theta_1), linestyle="--", color="grey")
    plt.plot(x_train, theta_2 * x_train, label=r"$\theta_2 = {:.2f}$".format(theta_2), linestyle="--", color="grey")
    problem.ellipsoid.plot(ax=plt.gca(), color="red", label="ellipsoid")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.title("Löwner-John ellipsoid")
    plt.tight_layout()
    plt.savefig("thesis_figures/1d_linreg_multiple/theta_r_star_lj.png")

    # 2.2 Smallest enclosing circle
    train_data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
    train_data_gen.contain_within_ellipse(circle)
    data_train = np.vstack([x_train, train_data_gen.y])
    problem = CADRO1DLinearRegression(data_train, circle)
    problem.solve(nb_theta_0=2)
    problem.set_theta_r()

    results = problem.results
    theta_r = results["theta_r"]
    theta_star = results["theta"]
    theta_1 = results["theta_0"][0]
    theta_2 = results["theta_0"][1]

    plt.figure()
    plt.scatter(data_train[0, :], data_train[1, :], label="data", marker=".")
    x_plot = np.linspace(-5, 5, 10)
    x_plot2 = np.linspace(-2.5, 2.5, 10)
    plt.plot(x_plot, theta_r * x_plot, label=r"$\theta_r = {:.2f}$".format(theta_r), linestyle="solid", color="blue")
    plt.plot(x_plot, theta_star * x_plot, label=r"$\theta^* = {:.2f}$".format(theta_star), color="red", linestyle="--")
    plt.plot(x_plot, theta_1 * x_plot, label=r"$\theta_1 = {:.2f}$".format(theta_1), linestyle="--", color="grey")
    plt.plot(x_plot2, theta_2 * x_plot2, label=r"$\theta_2 = {:.2f}$".format(theta_2), linestyle="--", color="grey")
    problem.ellipsoid.plot(ax=plt.gca(), color="red", label="ellipsoid")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.title("Circular support")
    plt.tight_layout()
    plt.savefig("thesis_figures/1d_linreg_multiple/theta_r_star_circle.png")

    plt.show()


def experiment3(seed):
    """
    Experiment 3: for a LJ ellipsoid and a circle, plot the realized values of alpha and lambda for illustrative
    purposes. We use a separate i.i.d. data set to generate the ellipsoids.
    Experiment 4: plot the loss for sigma = 1 and different m
    """
    plt.rcParams.update({'font.size': 10})

    # 3.0 generate ellipsoids
    m = 50
    rico = 3
    sigma = 1
    x = np.linspace(0, 1, m)
    data_gen = ScalarDataGenerator(x, seed)
    y = data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=False)
    data = np.vstack([x, y])
    lj_ellipsoid = Ellipsoid.lj_ellipsoid(data, scaling_factor=2)
    circle = Ellipsoid.smallest_enclosing_sphere(data)

    ms = (15, 20, 30, 50, 100, 150, 200)
    sigmas = (0.2, 0.5, 1, 2)
    nb_tries = 100
    nb_theta_0 = 2

    experiment3_loop(lj_ellipsoid, "lj", ms, sigmas, nb_tries, rico, seed, excel=False, nb_theta0=nb_theta_0)

    experiment3_loop(circle, "circle", ms, sigmas, nb_tries, rico, seed, excel=False, nb_theta0=nb_theta_0)


def experiment3_loop(ellipsoid, type, ms, sigmas, nb_tries, rico, seed, excel=True, nb_theta0=1):
    # create figures
    theta_fig, theta_axs = plt.subplots(len(ms), len(sigmas))
    alpha_fig, alpha_axs = plt.subplots(len(ms), len(sigmas))
    # set figure size to A4
    alpha_fig.set_size_inches(8.27, 11.69)
    theta_fig.set_size_inches(8.27, 11.69)

    # get robust cost
    problem = RobustOptimization(ellipsoid)
    results_r = problem.solve_1d_linear_regression()
    robust_cost = results_r["cost"]

    # initialize arrays
    theta_r_array = np.zeros((len(ms), len(sigmas), nb_tries))
    theta_star_array = np.zeros((len(ms), len(sigmas), nb_tries))
    theta_0_array = np.zeros((len(ms), len(sigmas), nb_theta0, nb_tries))
    alpha_array = np.zeros((len(ms), len(sigmas), nb_theta0, nb_tries))
    lambda_array = np.zeros((len(ms), len(sigmas), nb_theta0, nb_tries))
    loss_0_array = np.zeros((len(ms), len(sigmas), nb_tries))
    loss_star_array = np.zeros((len(ms), len(sigmas), nb_tries))
    loss_r_array = np.zeros((len(ms), len(sigmas), nb_tries))

    # initialize arrays for median and quantiles
    theta_star_median = np.zeros((len(ms), len(sigmas)))
    theta_star_upq = np.zeros((len(ms), len(sigmas)))
    theta_star_downq = np.zeros((len(ms), len(sigmas)))
    theta_0_median = np.zeros((len(ms), len(sigmas), nb_theta0))
    theta_0_upq = np.zeros((len(ms), len(sigmas), nb_theta0))
    theta_0_downq = np.zeros((len(ms), len(sigmas), nb_theta0))
    theta_r_median = np.zeros((len(ms), len(sigmas)))
    alpha_median = np.zeros((len(ms), len(sigmas), nb_theta0))
    alpha_upq = np.zeros((len(ms), len(sigmas), nb_theta0))
    alpha_downq = np.zeros((len(ms), len(sigmas), nb_theta0))
    loss_0_median = np.zeros((len(ms), len(sigmas)))  # we average the losses over the different theta_0
    loss_0_upq = np.zeros((len(ms), len(sigmas)))
    loss_0_downq = np.zeros((len(ms), len(sigmas)))
    loss_star_median = np.zeros((len(ms), len(sigmas)))
    loss_star_upq = np.zeros((len(ms), len(sigmas)))
    loss_star_downq = np.zeros((len(ms), len(sigmas)))
    loss_r_median = np.zeros((len(ms), len(sigmas)))

    if excel:
        # create dataframes for the results. The columns are the sigmas, the rows are the ms
        theta_star_df = pd.DataFrame(columns=sigmas, index=ms)
        theta_0_df = pd.DataFrame(columns=sigmas, index=ms)
        theta_r_df = pd.DataFrame(columns=sigmas, index=ms)
        alpha_df = pd.DataFrame(columns=sigmas, index=ms)
        collapses_df = pd.DataFrame(columns=sigmas, index=ms)
        loss_0_df = pd.DataFrame(columns=sigmas, index=ms)
        loss_star_df = pd.DataFrame(columns=sigmas, index=ms)

    x_test = np.linspace(0, 1, 1000)
    test_data_gen = ScalarDataGenerator(x_test, seed)

    for k, m in enumerate(ms):
        print(f"{datetime.now()} - {ellipsoid.type} - m = {m}")
        x_train = np.linspace(0, 1, m)
        data_gen = ScalarDataGenerator(x_train, seed)
        for i, sigma in enumerate(sigmas):
            y_test = test_data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
            data_test = np.vstack([x_test, y_test])
            for j in range(nb_tries):
                # solve the problem
                data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
                data_gen.contain_within_ellipse(ellipsoid)
                data_train = np.vstack([x_train, data_gen.y])
                problem = CADRO1DLinearRegression(data_train, ellipsoid)
                results = problem.solve(nb_theta_0=nb_theta0)
                problem.set_theta_r()
                # fill in the arrays
                theta_r_array[k, i, j] = problem.theta_r
                theta_star_array[k, i, j] = results["theta"]
                for l in range(nb_theta0):
                    theta_0_array[k, i, l, j] = results["theta_0"][l]
                    alpha_array[k, i, l, j] = results["alpha"][l]
                    lambda_array[k, i, l, j] = results["lambda"][l]
                loss_0_array[k, i, j] = np.mean([problem.test_loss(data_test, "theta_0", l)
                                                 for l in range(nb_theta0)])
                loss_star_array[k, i, j] = problem.test_loss(data_test, "theta")
                loss_r_array[k, i, j] = problem.test_loss(data_test, "theta_r")

            # fill in the median and quantiles
            theta_star_median[k, i] = np.median(theta_star_array[k, i, :])
            theta_star_upq[k, i] = np.quantile(theta_star_array[k, i, :], 0.75)
            theta_star_downq[k, i] = np.quantile(theta_star_array[k, i, :], 0.25)

            for l in range(nb_theta0):
                theta_0_median[k, i, l] = np.median(theta_0_array[k, i, l, :])
                theta_0_upq[k, i, l] = np.quantile(theta_0_array[k, i, l, :], 0.75)
                theta_0_downq[k, i, l] = np.quantile(theta_0_array[k, i, l, :], 0.25)
                alpha_median[k, i, l] = np.median(alpha_array[k, i, l, :])
                alpha_upq[k, i, l] = np.quantile(alpha_array[k, i, l, :], 0.75)
                alpha_downq[k, i, l] = np.quantile(alpha_array[k, i, l, :], 0.25)

            theta_r_median[k, i] = np.median(theta_r_array[k, i, :])
            loss_0_median[k, i] = np.median(loss_0_array[k, i, :])
            loss_0_upq[k, i] = np.quantile(loss_0_array[k, i, :], 0.75)
            loss_0_downq[k, i] = np.quantile(loss_0_array[k, i, :], 0.25)
            loss_star_median[k, i] = np.median(loss_star_array[k, i, :])
            loss_star_upq[k, i] = np.quantile(loss_star_array[k, i, :], 0.75)
            loss_star_downq[k, i] = np.quantile(loss_star_array[k, i, :], 0.25)
            loss_r_median[k, i] = np.median(loss_r_array[k, i, :])

            # plot theta_r, theta_star and theta_0
            theta_axs[k, i].scatter(np.ones(nb_tries), theta_r_array[k, i, :], marker=".", label=r"$\theta_r$")
            theta_axs[k, i].scatter(2 * np.ones(nb_tries), theta_star_array[k, i, :], marker=".", label=r"$\theta^*$")
            theta_axs[k, i].set_xticks([1, 2] + list(range(3, 3 + nb_theta0)))
            for l in range(nb_theta0):
                theta_axs[k, i].scatter((l + 3) * np.ones(nb_tries), theta_0_array[k, i, l, :], marker=".",
                                        label=r"$\theta_{}$".format(l + 1))
            theta_axs[k, i].set_xticklabels(
                [r"$\theta_r$", r"$\theta^*$"] + [r"$\theta_{}$".format(l) for l in range(1, nb_theta0 + 1)])
            theta_axs[k, i].set_title(r"$m = {}, \sigma = {}$".format(m, sigma))
            theta_axs[k, i].grid()

            # plot alpha and lambda
            # draw a black line at the robust cost
            alpha_axs[k, i].axhline(y=robust_cost, linestyle="--", color="black")
            alpha_axs[k, i].set_xticks(list(range(1, nb_theta0 + 1)))
            alpha_axs[k, i].set_xticklabels([r"$\alpha_{}$".format(l) for l in range(1, nb_theta0 + 1)])
            for l in range(nb_theta0):
                # get all indices of lambdas which are close to 0
                ind_lambdas_0 = np.where(np.abs(lambda_array[k, i, l, :]) < 0.1)[0]
                ind_lambdas_1 = np.setdiff1d(np.arange(nb_tries), ind_lambdas_0)
                alpha_axs[k, i].scatter((l + 1) * np.ones(len(ind_lambdas_0)), alpha_array[k, i, l, ind_lambdas_0],
                                        marker=".",
                                        label=r"$\alpha^0_{}$".format(l + 1),
                                        color="red")
                alpha_axs[k, i].scatter((l + 1) * np.ones(len(ind_lambdas_1)), alpha_array[k, i, l, ind_lambdas_1],
                                        marker=".",
                                        label=r"$\alpha^1_{}$".format(l + 1),
                                        color="blue")
            alpha_axs[k, i].set_title(r"$m = {}, \sigma = {}$".format(m, sigma))
            alpha_axs[k, i].grid()

            if excel:
                # fill in the table for thetas and alpha, and collapse rates
                # fill in with mean and quantiles as mean (25: 0.25 quantile, 75: 0.75 quantile)
                theta_star_df.loc[m, sigma] = "{:.4f} ({:.4f}, {:.4f})".format(theta_star_median[k, i],
                                                                               theta_star_downq[k, i],
                                                                               theta_star_upq[k, i])
                theta_0_df.loc[m, sigma] = "{:.4f} ({:.4f}, {:.4f})".format(theta_0_median[k, i],
                                                                            theta_0_downq[k, i],
                                                                            theta_0_upq[k, i])
                theta_r_df.loc[0, 0] = "{:.4f}".format(theta_r_median[k, i])
                theta_r_df.loc[0, 1] = "{:.4f}".format(loss_r_median[k, i])
                alpha_df.loc[m, sigma] = "{:.4f} ({:.4f}, {:.4f})".format(alpha_median[k, i],
                                                                          alpha_downq[k, i],
                                                                          alpha_upq[k, i])
                # collapse if lambda close to zero
                collapses = np.mean(np.abs(lambda_array[k, i, :]) < 0.1) * nb_tries
                collapses_df.loc[m, sigma] = "{:.4f}".format(collapses)
                loss_0_df.loc[m, sigma] = "{:.4f} ({:.4f}, {:.4f})".format(loss_0_median[k, i],
                                                                           loss_0_downq[k, i],
                                                                           loss_0_upq[k, i])
                loss_star_df.loc[m, sigma] = "{:.4f} ({:.4f}, {:.4f})".format(loss_star_median[k, i],
                                                                              loss_star_downq[k, i],
                                                                              loss_star_upq[k, i])

    # set layout and save figures
    alpha_fig.tight_layout()
    theta_fig.tight_layout()

    # save figures
    alpha_fig.savefig("thesis_figures/1d_linreg_multiple/{}_alpha.png".format(type))
    theta_fig.savefig("thesis_figures/1d_linreg_multiple/{}_theta.png".format(type))
    plt.show()

    plt.rcParams.update({'font.size': 15})

    # plot the loss for sigma = 1 and different m
    # collect the losses for sigma = 1
    sigma_index = 2
    loss_0 = loss_0_median[:, sigma_index]
    loss_0_up = loss_0_upq[:, sigma_index]
    loss_0_down = loss_0_downq[:, sigma_index]

    loss_star = loss_star_median[:, sigma_index]
    loss_star_up = loss_star_upq[:, sigma_index]
    loss_star_down = loss_star_downq[:, sigma_index]

    plt.figure()
    plt.errorbar(ms, loss_0, yerr=[loss_0 - loss_0_down, loss_0_up - loss_0], fmt='o-', label=r"$\ell(\theta_0)$")
    plt.errorbar(ms, loss_star, yerr=[loss_star - loss_star_down, loss_star_up - loss_star], fmt='o-',
                 label=r"$\ell(\theta^*)$")
    plt.xlabel("m")
    # x in log scale
    plt.xscale("log")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.title(f"Loss for $\sigma = 1$ ({type})")
    plt.savefig("thesis_figures/1d_linreg_multiple/{}_loss.png".format(type))
    plt.show()

    if excel:
        # save tables to xlsx file
        with pd.ExcelWriter("thesis_figures/1d_linreg/{}_results.xlsx".format(type)) as writer:
            theta_star_df.to_excel(writer, sheet_name="theta_star")
            theta_0_df.to_excel(writer, sheet_name="theta_0")
            theta_r_df.to_excel(writer, sheet_name="theta_r")
            alpha_df.to_excel(writer, sheet_name="alpha")
            collapses_df.to_excel(writer, sheet_name="collapses")
            loss_0_df.to_excel(writer, sheet_name="loss_0")
            loss_star_df.to_excel(writer, sheet_name="loss_star")
            writer.close()

        # save to latex tables
        theta_star_df.to_latex("thesis_figures/1d_linreg/{}_theta_star.tex".format(type))
        theta_0_df.to_latex("thesis_figures/1d_linreg/{}_theta_0.tex".format(type))
        theta_r_df.to_latex("thesis_figures/1d_linreg/{}_theta_r.tex".format(type))
        alpha_df.to_latex("thesis_figures/1d_linreg/{}_alpha.tex".format(type))
        collapses_df.to_latex("thesis_figures/1d_linreg/{}_collapses.tex".format(type))
        loss_0_df.to_latex("thesis_figures/1d_linreg/{}_loss_0.tex".format(type))
        loss_star_df.to_latex("thesis_figures/1d_linreg/{}_loss_star.tex".format(type))


def experiment4(seed):
    """
    Plot the loss function for the circle and LJ ellipsoid sigma = 1 and m = 50
    """
    plt.rcParams.update({'font.size': 15})

    m = 50
    sigma = 1

    x_ellipse = np.linspace(0, 1, 2 * m)
    data_gen = ScalarDataGenerator(x_ellipse, seed)
    y_ellipse = data_gen.generate_linear_norm_disturbance(0, sigma, 3, outliers=True)
    data_ellipse = np.vstack([x_ellipse, y_ellipse])
    # lj_ellipsoid = Ellipsoid.lj_ellipsoid(data_ellipse, scaling_factor=2)
    circle = Ellipsoid.smallest_enclosing_sphere(data_ellipse)

    x = np.linspace(0, 1, m)
    data_gen = ScalarDataGenerator(x, seed)
    data_gen.generate_linear_norm_disturbance(0, sigma, 3, outliers=True)
    # data_gen.contain_within_ellipse(lj_ellipsoid)  # activate for LJ ellipsoid
    data_gen.contain_within_ellipse(circle)
    data = np.vstack([x, data_gen.y])

    x_test = np.linspace(0, 1, 1000)
    test_data_gen = ScalarDataGenerator(x_test, seed)
    test_data_gen.generate_linear_norm_disturbance(0, sigma, 3, outliers=True)
    test_data = np.vstack([x_test, test_data_gen.y])

    theta = np.linspace(-2.5, 4.5, 130)
    # problem = CADRO1DLinearRegression(data, lj_ellipsoid)  # activate for LJ ellipsoid
    problem = CADRO1DLinearRegression(data, circle)
    objective = np.zeros(len(theta))
    test_loss = np.zeros(len(theta))

    # get theta_0, theta_r and theta_star
    problem.solve(nb_theta_0=2)
    theta_0 = problem.results["theta_0"]
    theta_1, theta_2 = theta_0
    theta_star = problem.results["theta"]
    problem.set_theta_r()
    theta_r = problem.theta_r

    for i in range(len(theta)):
        problem.reset()
        results = problem.solve(theta=theta[i], nb_theta_0=2, theta0=theta_0)
        objective[i] = results['objective']
        test_loss[i] = problem.test_loss(test_data)

    # plot the loss function
    plt.figure()
    plt.plot(theta, objective, label="objective")
    plt.plot(theta, test_loss * 10, label="test loss x 10")
    # add a vertical line for theta_0, theta_r and theta_star
    plt.axvline(x=theta_1, linestyle='--', color='k')
    plt.axvline(x=theta_2, linestyle='--', color='k')
    # add a vertical line for theta_r and theta_star
    plt.axvline(x=theta_r, linestyle='--', color='r')
    plt.axvline(x=theta_star, linestyle='--', color='g')

    # plt.xticks([theta_1, theta_2, theta_r, theta_star] + [-2, -1, 0, 1, 2, 3, 4],
    #            [r"         $\theta_1,\theta^\star$", r"$\theta_2$", r"$\theta_r$", None] + [None, -1, 0, 1, 2, 3, None])

    plt.xticks([theta_1, theta_2, theta_r, theta_star] + [-2, -1, 0, 1, 2, 3, 4],
               [r"$\theta_1$", r"$\theta_2$", r"$\theta_r, \theta^\star$", None] + [None, -1, None, 1, 2, 3, None])

    plt.legend()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\ell(\theta, \xi)$")
    # plt.title("Loss function for LJ ellipsoid")
    plt.title("Loss function for circular support")
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    seed = 42
    experiment1(seed)
    # experiment3(seed)
    # experiment4(seed)
