import warnings
import numpy as np
import matplotlib.pyplot as plt
from ellipsoids import Ellipsoid
from robust_optimization import RobustOptimization
from one_dimension_cadro import CADRO1DLinearRegression
from utils.data_generator import ScalarDataGenerator
import pandas as pd


def experiment1(seed):
    """
    Experiment 1: create ellipsoids for illustration purposes
    """
    plt.rcParams.update({'font.size': 15})

    rico = 2
    m = 30
    x = np.linspace(0, 1, m)
    data_gen = ScalarDataGenerator(x, seed)
    y = data_gen.generate_linear_norm_disturbance(0, 1, rico, outliers=True)
    data = np.vstack([x, y])

    # 1.1 Löwner-John ellipsoid
    plt.figure()
    ellipsoid = Ellipsoid.lj_ellipsoid(data)
    ellipsoid.plot()
    plt.scatter(data[0, :], data[1, :], label="data", marker=".")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.title("Löwner-John ellipsoid")
    plt.savefig("thesis_figures/1d_linreg/lj_ellipsoid.png")
    plt.show()

    # 1.2 Principal ellipsoid
    plt.figure()
    angle = np.pi / 3
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    ellipsoid = Ellipsoid.from_principal_axes(R, data)
    ellipsoid.plot()
    plt.scatter(data[0, :], data[1, :], label="data", marker=".")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.title("Principal ellipsoid")
    plt.savefig("thesis_figures/1d_linreg/principal_ellipsoid.png")
    plt.show()

    # 1.3 Circle
    plt.figure()
    ellipsoid = Ellipsoid.smallest_enclosing_sphere(data)
    ellipsoid.plot()
    plt.scatter(data[0, :], data[1, :], label="data", marker=".")
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.title("Circle")
    plt.savefig("thesis_figures/1d_linreg/circle.png")
    plt.show()


def experiment2(seed):
    """
    Experiment 2: for a LJ ellipsoid and a circle, plot the realized values of theta_r, theta_star and theta_0 for
    illustrative purposes. We use a seperate i.i.d. data set to generate the ellipsoids.
    """
    plt.rcParams.update({'font.size': 15})

    # 2.0 generate ellipsoids
    m = 50
    rico = 3
    sigma = 1
    x = np.linspace(0, 1, m)
    data_gen = ScalarDataGenerator(x, seed)
    y = data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
    data = np.vstack([x, y])
    lj_ellipsoid = Ellipsoid.lj_ellipsoid(data)
    circle = Ellipsoid.smallest_enclosing_sphere(data)

    # set up problem
    n = 30
    x_train = np.linspace(0, 1, n)
    train_data_gen = ScalarDataGenerator(x_train, seed)

    # 2.1 Löwner-John
    train_data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
    train_data_gen.contain_within_ellipse(lj_ellipsoid)
    data_train = np.vstack([x_train, train_data_gen.y])
    problem = CADRO1DLinearRegression(data_train, lj_ellipsoid)
    problem.solve()
    problem.set_theta_r()
    theta_r = problem.theta_r
    theta_star = problem.results["theta"]
    theta_0 = problem.results["theta_0"][0]
    x_plot = np.linspace(-0.5, 1.5, 10)
    plt.figure()
    plt.scatter(data_train[0, :], data_train[1, :], label="data", marker=".")
    plt.plot(x_plot, theta_r * x_plot, label=r"$\theta_r = {:.4f}$".format(theta_r))
    plt.plot(x_plot, theta_star * x_plot, label=r"$\theta^* = {:.4f}$".format(theta_star), color="green",
             linestyle="--")
    plt.plot(x_plot, theta_0 * x_plot, label=r"$\theta_0 = {:.4f}$".format(theta_0), linestyle=":")
    problem.ellipsoid.plot(ax=plt.gca(), color="red", label="ellipsoid")
    plt.legend()
    plt.grid()
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.title("Löwner-John ellipsoid")
    plt.savefig("thesis_figures/1d_linreg/exp2_lj_ellipsoid.png")
    plt.show()

    # 2.2 Circle
    train_data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
    train_data_gen.contain_within_ellipse(circle)
    data_train = np.vstack([x_train, train_data_gen.y])
    problem = CADRO1DLinearRegression(data_train, circle)
    problem.solve()
    problem.set_theta_r()
    theta_r = problem.theta_r
    theta_star = problem.results["theta"]
    theta_0 = problem.results["theta_0"][0]
    plt.figure()
    plt.scatter(data_train[0, :], data_train[1, :], label="data", marker=".")
    plt.plot(x_plot, theta_r * x_plot, label=r"$\theta_r = {:.4f}$".format(theta_r))
    plt.plot(x_plot, theta_star * x_plot, label=r"$\theta^* = {:.4f}$".format(theta_star), color="green",
             linestyle="--")
    plt.plot(x_plot, theta_0 * x_plot, label=r"$\theta_0 = {:.4f}$".format(theta_0), linestyle=":")
    problem.ellipsoid.plot(ax=plt.gca(), color="red", label="ellipsoid")
    plt.legend()
    plt.grid()
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.title("Circle")
    plt.savefig("thesis_figures/1d_linreg/exp2_circle.png")
    plt.show()


def experiment3(seed):
    """
    Experiment 3: for a LJ ellipsoid and a circle, plot the realized values of alpha and lambda for illustrative
    purposes. We use a separate i.i.d. data set to generate the ellipsoids.
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

    ms = (5, 10, 15, 20, 30, 50, 100)
    sigmas = (0.2, 0.5, 1, 2, 3, 5)
    nb_tries = 100

    experiment3_loop(lj_ellipsoid, "lj", ms, sigmas, nb_tries, rico, seed)

    experiment3_loop(circle, "circle", ms, sigmas, nb_tries, rico, seed)


def experiment3_loop(ellipsoid, type, ms, sigmas, nb_tries, rico, seed, excel=True):

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
    theta_0_array = np.zeros((len(ms), len(sigmas), nb_tries))
    alpha_array = np.zeros((len(ms), len(sigmas), nb_tries))
    lambda_array = np.zeros((len(ms), len(sigmas), nb_tries))
    loss_0_array = np.zeros((len(ms), len(sigmas), nb_tries))
    loss_star_array = np.zeros((len(ms), len(sigmas), nb_tries))
    loss_r_array = np.zeros((len(ms), len(sigmas), nb_tries))

    # initialize arrays for median and quantiles
    theta_star_median = np.zeros((len(ms), len(sigmas)))
    theta_star_upq = np.zeros((len(ms), len(sigmas)))
    theta_star_downq = np.zeros((len(ms), len(sigmas)))
    theta_0_median = np.zeros((len(ms), len(sigmas)))
    theta_0_upq = np.zeros((len(ms), len(sigmas)))
    theta_0_downq = np.zeros((len(ms), len(sigmas)))
    theta_r_median = np.zeros((len(ms), len(sigmas)))
    alpha_median = np.zeros((len(ms), len(sigmas)))
    alpha_upq = np.zeros((len(ms), len(sigmas)))
    alpha_downq = np.zeros((len(ms), len(sigmas)))
    loss_0_median = np.zeros((len(ms), len(sigmas)))
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
                results = problem.solve()
                problem.set_theta_r()
                # fill in the arrays
                theta_r_array[k, i, j] = problem.theta_r
                theta_star_array[k, i, j] = results["theta"]
                theta_0_array[k, i, j] = results["theta_0"][0]
                alpha_array[k, i, j] = results["alpha"][0]
                lambda_array[k, i, j] = results["lambda"]
                loss_0_array[k, i, j] = problem.test_loss(data_test, "theta_0")
                loss_star_array[k, i, j] = problem.test_loss(data_test, "theta")
                loss_r_array[k, i, j] = problem.test_loss(data_test, "theta_r")

            # fill in the median and quantiles
            theta_star_median[k, i] = np.median(theta_star_array[k, i, :])
            theta_star_upq[k, i] = np.quantile(theta_star_array[k, i, :], 0.75)
            theta_star_downq[k, i] = np.quantile(theta_star_array[k, i, :], 0.25)
            theta_0_median[k, i] = np.median(theta_0_array[k, i, :])
            theta_0_upq[k, i] = np.quantile(theta_0_array[k, i, :], 0.75)
            theta_0_downq[k, i] = np.quantile(theta_0_array[k, i, :], 0.25)
            theta_r_median[k, i] = np.median(theta_r_array[k, i, :])
            alpha_median[k, i] = np.median(alpha_array[k, i, :])
            alpha_upq[k, i] = np.quantile(alpha_array[k, i, :], 0.75)
            alpha_downq[k, i] = np.quantile(alpha_array[k, i, :], 0.25)
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
            theta_axs[k, i].scatter(3 * np.ones(nb_tries), theta_0_array[k, i, :], marker=".", label=r"$\theta_0$")
            theta_axs[k, i].set_xticks([1, 2, 3])
            theta_axs[k, i].set_xticklabels([r"$\theta_r$", r"$\theta^*$", r"$\theta_0$"])
            theta_axs[k, i].set_title(r"$m = {}, \sigma = {}$".format(m, sigma))
            theta_axs[k, i].grid()

            # plot alpha and lambda
            # get all indices where lambda is close to 0
            ind_lambdas_0 = np.where(np.abs(lambda_array[k, i, :]) < 0.1)[0]
            ind_lambdas_1 = np.setdiff1d(np.arange(nb_tries), ind_lambdas_0)
            alpha_axs[k, i].scatter(np.ones(len(ind_lambdas_0)), alpha_array[k, i, ind_lambdas_0], marker=".",
                                    label=r"$\alpha$",
                                    color="red")
            alpha_axs[k, i].scatter(np.ones(len(ind_lambdas_1)), alpha_array[k, i, ind_lambdas_1], marker=".",
                                    label=r"$\alpha$",
                                    color="blue")
            # draw a black line at the robust cost
            alpha_axs[k, i].axhline(y=robust_cost, linestyle="--", color="black")
            alpha_axs[k, i].set_xticks([1])
            alpha_axs[k, i].set_xticklabels([r"$\alpha$"])
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
    alpha_fig.savefig("thesis_figures/1d_linreg/{}_alpha.png".format(type))
    theta_fig.savefig("thesis_figures/1d_linreg/{}_theta.png".format(type))
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


def experiment45(seed):
    """
    Experiment 4: for one specific value of m and sigma, plot the realized values of theta_r, theta_star,
    theta_0 and alpha
    """
    plt.rcParams.update({'font.size': 15})

    rico = 3
    m = 50
    sigma = 1
    nb_tries = 100

    x = np.linspace(0, 1, m)
    data_gen = ScalarDataGenerator(x, seed)
    y = data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
    data = np.vstack([x, y])
    lj_ellipse = Ellipsoid.lj_ellipsoid(data)
    circle = Ellipsoid.smallest_enclosing_sphere(data)
    ellipses = (lj_ellipse, circle)

    theta_r_array = np.zeros((len(ellipses)))
    theta_star_array = np.zeros((len(ellipses), nb_tries))
    theta_0_array = np.zeros((len(ellipses), nb_tries))
    alpha_array = np.zeros((len(ellipses), nb_tries))
    collapses = np.zeros((len(ellipses), nb_tries))
    lambda_array = np.zeros((len(ellipses), nb_tries))

    for k in range(len(ellipses)):
        ellipse = ellipses[k]
        type = "lj" if k == 0 else "circle"
        for j in range(nb_tries):
            data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
            data_gen.contain_within_ellipse(ellipse)
            data = np.vstack([x, data_gen.y])
            problem = CADRO1DLinearRegression(data, ellipse)
            results = problem.solve()
            problem.set_theta_r()
            if j == 0:
                theta_r_array[k] = problem.theta_r
            theta_star_array[k, j] = results["theta"]
            theta_0_array[k, j] = results["theta_0"][0]
            alpha_array[k, j] = results["alpha"][0]
            collapses[k, j] = np.abs(results["theta_0"][0] - results["theta"]) > \
                              np.abs(results["theta"] - problem.theta_r)
            lambda_array[k, j] = results["lambda"]

        plt.figure()
        plt.scatter(np.ones(nb_tries), theta_r_array[k] * np.ones(nb_tries), marker=".", label=r"$\theta_r$")
        plt.scatter(2 * np.ones(nb_tries), theta_star_array[k, :], marker=".", label=r"$\theta^*$")
        plt.scatter(3 * np.ones(nb_tries), theta_0_array[k, :], marker=".", label=r"$\theta_0$")
        plt.xticks([1, 2, 3], [r"$\theta_r$", r"$\theta^*$", r"$\theta_0$"])
        plt.title(f"{'Circle' if type == 'circle' else 'Löwner-John ellipsoid'}")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig("thesis_figures/1d_linreg/exp4_{}_theta.png".format(type))
        plt.show()

        # calculate robust cost
        robust_problem = RobustOptimization(ellipse)
        robust_results = robust_problem.solve_1d_linear_regression()
        cost = robust_results["cost"]
        plt.figure()
        # get all indices at which lambda is closer to 1 than to 0, and vice versa
        ind_lambdas_0 = np.where(np.abs(lambda_array[k, :]) < 0.1)[0]
        ind_alphas_1 = np.setdiff1d(np.arange(nb_tries), ind_lambdas_0)
        plt.scatter(np.ones(len(ind_lambdas_0)), alpha_array[k, ind_lambdas_0], marker=".", label=r"$\alpha_{[0]}$",
                    color='red')
        plt.scatter(np.ones(len(ind_alphas_1)), alpha_array[k, ind_alphas_1], marker=".", label=r"$\alpha_{[1]}$",
                    color='blue')
        plt.xticks([1], [r"$\alpha$"])
        # draw horizontal line for robust cost
        plt.axhline(y=cost, linestyle="--", color="black", label="robust cost")
        fig = plt.gcf()
        fig.set_size_inches(3, 6)
        plt.title(f"{'Circle' if type == 'circle' else 'Löwner-John ellipsoid'}")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig("thesis_figures/1d_linreg/exp4_{}_alpha.png".format(type))
        plt.show()

    # Experiment 5: plot the loss function for the circle and the LJ ellipsoid for different values of theta
    # setup data
    x_test = np.linspace(0, 1, 1000)
    test_data_gen = ScalarDataGenerator(x_test, seed)
    data_test = np.vstack([x_test, test_data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)])
    for k in range(len(ellipses)):
        ellipse = ellipses[k]
        data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
        data_gen.contain_within_ellipse(ellipse)
        data = np.vstack([x, data_gen.y])
        if k == 0:
            theta = np.linspace(2, 8, 100)
        else:
            theta = np.linspace(-1, 5, 100)

        problem = CADRO1DLinearRegression(data, ellipse)
        objective = np.zeros(len(theta))
        test_loss = np.zeros(len(theta))

        for i in range(len(theta)):
            problem.reset()
            results = problem.solve(theta=theta[i], nb_theta_0=1)
            objective[i] = results["objective"]
            test_loss[i] = problem.test_loss(data_test)

        problem.solve()
        theta_star = problem.results["theta"]
        theta_0 = problem.results["theta_0"][0]
        problem.set_theta_r()
        theta_r = problem.theta_r

        plt.figure()
        plt.plot(theta, objective, label="objective")
        plt.plot(theta, test_loss * 10, label="test loss x 10")
        plt.axvline(x=theta_0, linestyle="--", color="black")
        plt.axvline(x=theta_r, linestyle="--", color="black")
        plt.axvline(x=theta_star, linestyle="--", color="red", label=r"$\theta^*$")
        if k == 0:
            plt.xticks([theta_0, theta_r] + [2, 3, 4, 5, 6, 7, 8],
                       [r"$\theta_0$", r"$\theta_r$"] + ['2', '3', '4', '5', '6', '7', '8'])
        elif k == 1:
            plt.xticks([theta_0, theta_r] + [-1, 1, 2, 3, 4, 5],
                       [r"$\theta_0$", r"$\theta_r$"] + ['-1', '1', '2', '3', '4', '5'])
        plt.legend()
        plt.grid()
        plt.title(f"{'Circle' if k == 1 else 'Löwner-John ellipsoid'}")
        plt.tight_layout()
        plt.savefig(f"thesis_figures/1d_linreg/exp5_{k}_loss.png")
        plt.show()



def experiment6():
    """
    plot the loss function for the circle and the LJ ellipsoid for different values of m
    and for both theta_0 and theta_star
    """
    plt.rcParams.update({'font.size': 15})

    lj_star = np.array([4.3915, 4.4084, 4.4264, 4.4394, 4.4873, 4.4917, 4.4749])
    lj_star_upq = np.array([4.3915, 4.4084, 4.4264, 4.4394, 4.7893, 5.0232, 4.5896])
    lj_star_downq = np.array([4.3195, 4.4084, 4.4262, 4.4394, 4.4190, 4.43616, 4.4199])

    lj_0 = np.array([6.6304, 17.6625, 25.8751, 18.4294, 6.4817, 4.5337, 4.4749])
    lj_0_downq = np.array([4.9547, 6.1798, 12.4730, 8.4875, 4.7309, 4.3703, 4.4199])
    lj_0_upq = np.array([9.8649, 36.0025, 108.5167, 34.9789, 10.9431, 5.4329, 4.5896])


    circ_star = np.array([6.7928, 7.0101, 7.4317, 7.1130, 7.3620, 7.2082, 4.4447])
    circ_star_downq = np.array([6.7928, 7.0101, 7.4317, 7.1130, 5.7603, 4.8139, 4.4071])
    circ_star_upq = np.array([6.79287, 7.0101, 7.4318, 7.1130, 7.3621, 7.2084, 4.5810])

    circ_0 = np.array([6.7024, 13.4722, 39.8733, 19.8666, 5.5588, 4.5478, 4.4446])
    circ_0_downq = np.array([4.9390, 5.9448, 13.6888, 8.1248, 4.7207, 4.3661, 4.4071])
    circ_0_upq = np.array([10.97, 31.51, 83.82, 35.0677, 9.25, 5.6358, 4.5339])

    m = [5, 10, 15, 20, 30, 50, 100]

    plt.figure()
    plt.errorbar(m, lj_star, yerr=[lj_star - lj_star_downq, lj_star_upq - lj_star], label=r"LJ, $\theta^*$", fmt='o-')
    plt.errorbar(m, lj_0, yerr=[lj_0 - lj_0_downq, lj_0_upq - lj_0], label=r"LJ, $\theta_0$", fmt='o-')
    plt.grid()
    plt.xlabel("m")
    plt.ylabel("loss")
    plt.ylim(0, 50)
    plt.title(r"Loss for $\sigma=1$ (lj)")
    # x-axis is logarithmic
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("thesis_figures/1d_linreg/exp6_lj_loss.pdf")
    plt.show()

    plt.figure()
    plt.errorbar(m, circ_star, yerr=[circ_star - circ_star_downq, circ_star_upq - circ_star], label=r"Circle, $\theta^*$", marker="o")
    plt.errorbar(m, circ_0, yerr=[circ_0 - circ_0_downq, circ_0_upq - circ_0], label=r"Circle, $\theta_0$", marker="o")
    plt.grid()
    plt.xlabel("m")
    plt.ylabel("loss")
    plt.title(r"Loss for $\sigma=1$ (circle)")
    plt.ylim(0, 50)
    # x-axis is logarithmic
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("thesis_figures/1d_linreg/exp6_circ_loss.pdf")
    plt.show()


def experiment7(seed):
    """
    Experiment 7: plot a histogram of the realized values of theta, and of the loss.
    """
    plt.rcParams.update({'font.size': 15})

    rico = 3
    m = 50
    sigma = 1
    nb_tries = 300

    x = np.linspace(0, 1, m)
    data_gen = ScalarDataGenerator(x, seed)
    y = data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
    data = np.vstack([x, y])
    lj_ellipse = Ellipsoid.lj_ellipsoid(data)
    circle = Ellipsoid.smallest_enclosing_sphere(data)
    ellipses = (lj_ellipse, circle)

    theta_r_array = np.zeros((len(ellipses)))
    theta_star_array = np.zeros((len(ellipses), nb_tries))
    theta_0_array = np.zeros((len(ellipses), nb_tries))

    cost_r_array = np.zeros((len(ellipses)))
    cost_star_array = np.zeros((len(ellipses), nb_tries))
    cost_0_array = np.zeros((len(ellipses), nb_tries))

    collapse_count = [0] * len(ellipses)
    noncollapse_array = [0] * len(ellipses)
    robustcost_array = [0] * len(ellipses)

    test_x = np.linspace(0, 1, 1000)
    test_data_gen = ScalarDataGenerator(test_x, seed)
    test_data = np.vstack([test_x, test_data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)])

    for k in range(len(ellipses)):
        ellipse = ellipses[k]
        type = "LJ" if k == 0 else "Circle"
        for j in range(nb_tries):
            data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
            data_gen.contain_within_ellipse(ellipse)
            data = np.vstack([x, data_gen.y])
            problem = CADRO1DLinearRegression(data, ellipse)
            results = problem.solve()
            problem.set_theta_r()
            if j == 0:
                theta_r_array[k] = problem.theta_r
                cost_r_array[k] = problem.test_loss(test_data, "theta_r")
            theta_star_array[k, j] = results["theta"]
            theta_0_array[k, j] = results["theta_0"][0]

            if results["lambda"] < 0.5:
                collapse_count[k] += 1

            cost_star_array[k, j] = problem.test_loss(test_data, "theta")
            cost_0_array[k, j] = problem.test_loss(test_data, "theta_0")

        noncollapse_array[k] = [theta_r_array[k]] * collapse_count[k]
        robustcost_array[k] = [cost_r_array[k]] * collapse_count[k]

    # Experiment 7: plot a histogram of the values of theta_0 and theta_star
    for k in range(len(ellipses)):
        ellipse = ellipses[k]
        plt.figure()
        plt.hist(theta_0_array[k, :], bins=20, label=r"$\theta_0$", range=(-5, 12))
        plt.hist(theta_star_array[k, :], bins=20, label=r"$\theta^*$", range=(-5, 12), rwidth=0.7)
        plt.hist(noncollapse_array[k], bins=20, range=(-5, 12), rwidth=0.7)
        plt.legend()
        plt.grid()
        plt.title(f"{'Circle' if k == 1 else 'Löwner-John ellipsoid'}")
        plt.savefig(f"thesis_figures/1d_linreg/exp7_{k}_hist.png")
        plt.show()
        print(f"For {'Circle' if k == 1 else 'Löwner-John ellipsoid'}: theta_r = {theta_r_array[k]}")

    # Experiment 7a: plot a histogram of the values of the test loss for theta_0 and theta_star
    for k in range(len(ellipses)):
        ellipse = ellipses[k]
        plt.figure()
        plt.hist(cost_0_array[k, :], bins=30, label=r"$J(\theta_0)$", range=(0, 40))
        plt.hist(cost_star_array[k, :], bins=30, label=r"$J(\theta^*)$", rwidth=0.7, range=(0, 40))
        plt.hist(robustcost_array[k], bins=30, range=(0, 40), rwidth=0.7)
        plt.legend()
        plt.grid()
        plt.title(f"{'Circle' if k == 1 else 'Löwner-John ellipsoid'}")
        plt.savefig(f"thesis_figures/1d_linreg/exp7_{k}_hist_testloss.png")
        plt.show()
        print(f"For {'Circle' if k == 1 else 'Löwner-John ellipsoid'}: cost_r = {cost_r_array[k]}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    seed = 0
    # experiment1(seed)
    # experiment2(seed)
    # experiment3(seed)
    experiment45(seed)
    # experiment6()
    # experiment7(seed)