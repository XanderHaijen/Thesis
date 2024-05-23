import numpy as np
import matplotlib.pyplot as plt
from ellipsoids import Ellipsoid
from robust_optimization import RobustOptimization
from multiple_dimension_cadro import LeastSquaresCadro
import cvxpy as cp
from utils.data_generator import MultivariateDataGenerator as MDG
import utils.multivariate_experiments as aux
import pandas as pd
from time import time
from datetime import datetime
from moment_dro import MomentDRO


def experiment1(seed):
    """
    Experiment 1: Plot a test setup for a 3D CADRO model
    """
    plt.rcParams.update({'font.size': 15})

    generator = np.random.default_rng(seed)
    n = 20
    slope = 2 * np.ones((2,))
    # sample uniformly from the unit hypercube
    data = generator.uniform(size=(2, n))
    y = np.array([np.dot(data[:, i], slope) + generator.normal(scale=1) for i in range(n)])
    data = np.vstack((data, y))
    ellipsoid = Ellipsoid.lj_ellipsoid(data)
    robust_optimization = RobustOptimization(ellipsoid)
    robust_optimization.solve_least_squares()
    theta_r = robust_optimization.theta

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[0, :], data[1, :], data[2, :])
    # draw the direction plane
    x = np.linspace(0, 2, 10)
    y = np.linspace(0, 2, 10)
    X, Y = np.meshgrid(x, y)
    Z_star = np.zeros(X.shape)
    Z_r = np.zeros(X.shape)
    plt.xlabel("x")
    plt.ylabel("y")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z_star[i, j] = np.dot(np.array([X[i, j], Y[i, j]]), slope)
            Z_r[i, j] = np.dot(np.array([X[i, j], Y[i, j]]), theta_r)
    ax.plot_surface(X, Y, Z_star, alpha=0.2, color='r', label="Slope")
    ax.plot_surface(X, Y, Z_r, alpha=0.2, color='g', label=r"$\theta_r$")

    # plot theta_r as a vector
    ax.quiver(0, 0, 0, 1, theta_r[0], theta_r[1], color='g', label=r"$\theta_r$")
    plt.tight_layout()
    plt.show()


def experiment2(seed):
    """
    For different values of d, m and sigma, solve the CADRO problem and plot the values for alpha and the loss
    """
    plt.rcParams.update({'font.size': 8})
    generator = np.random.default_rng(seed)
    dimensions = [2, 5, 10, 15, 20, 25, 30]
    a, b = 0, 10  # side lengths of the hypercube
    assert b > a

    for n_d, d in enumerate(dimensions):
        slope = [i + 1 for i in range(d - 1)]

        # we generate the bounding ellipses based on a bounding box around the data which we calculate a priori
        print(f"{datetime.now()} - Dimension {d} - Ellipsoid construction")
        emp_slope = slope + np.clip(generator.normal(scale=0.5, size=(d - 1,)), -0.5, 0.5)  # random disturbance
        lj = Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)), -4, 4, theta=emp_slope,
                                            scaling_factor=1.05)
        lj.type = "LJ"

        delta_w = (b - a) / 2
        ses = Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)), -delta_w, delta_w,
                                             theta=emp_slope, scaling_factor=1.05)
        ses.type = "SCC"  # smallest circumscribed cube

        r = np.dot(slope, b * np.ones((d - 1,))) + 4
        ses = Ellipsoid.ellipse_from_corners(-r * np.ones((d - 1,)), r * np.ones((d - 1,)), -r, r, theta=np.zeros((d - 1,)),
                                             scaling_factor=1.05)
        ses.type = "SES"  # smallest enclosing sphere

        # conduct experiments for Löwner-John ellipsoid
        print(f"{datetime.now()} - Dimension {d}, LJ ellipsoid")
        alpha_data, loss_0_data, loss_star_data, loss_r = experiment2_loop(d, lj, generator, slope, a, b)

        # write the dataframe to a text file as latex tables and to an Excel file
        with pd.ExcelWriter(f"thesis_figures/multivariate_ls/full_exp/d{d}_experiment2_lj.xlsx") as writer:
            alpha_data.to_excel(writer, sheet_name='alpha')
            loss_0_data.to_excel(writer, sheet_name='loss_0')
            loss_star_data.to_excel(writer, sheet_name='loss_star')

        with open(f"thesis_figures/multivariate_ls/full_exp/d{d}_experiment2_lj.txt", "w") as f:
            f.write("Alpha data \n")
            f.write(alpha_data.to_latex(float_format="%.0f"))
            f.write("\n")
            f.write("Loss 0 data \n")
            f.write(loss_0_data.to_latex(float_format="%.0f"))
            f.write("\n")
            f.write("Loss star data \n")
            f.write(loss_star_data.to_latex(float_format="%.0f"))
            f.write("\n")
            f.write(f"Robust cost: {loss_r}")

        # # conduct experiments for smallest enclosing sphere
        print(f"{datetime.now()} - Dimension {d}, SES ellipsoid")
        alpha_data, loss_0_data, loss_star_data, loss_r = experiment2_loop(d, ses, generator, slope, a, b)

        # write the dataframe to a text file as latex tables and to an Excel file
        with pd.ExcelWriter(f"thesis_figures/multivariate_ls/full_exp/d{d}_experiment2_{ses.type}.xlsx") as writer:
            alpha_data.to_excel(writer, sheet_name='alpha')
            loss_0_data.to_excel(writer, sheet_name='loss_0')
            loss_star_data.to_excel(writer, sheet_name='loss_star')

        with open(f"thesis_figures/multivariate_ls/full_exp/d{d}_experiment2_{ses.type}.txt", "w") as f:
            f.write("Alpha data \n")
            f.write(alpha_data.to_latex(float_format="%.0f"))
            f.write("\n")
            f.write("Loss 0 data \n")
            f.write(loss_0_data.to_latex(float_format="%.0f"))
            f.write("\n")
            f.write("Loss star data \n")
            f.write(loss_star_data.to_latex(float_format="%.0f"))
            f.write("\n")
            f.write(f"Robust cost: {loss_r}")


def experiment2_loop(dimension, ellipsoid, generator, slope, a, b):
    # general setup
    data_size = lambda d: [2 * d, 5 * d, 8 * d, 12 * d, 25 * d]
    sigmas = [0.2, 0.5, 1, 2]
    nb_tries = 100

    alpha_array = np.zeros((len(data_size(dimension)), len(sigmas), nb_tries))
    lambda_array = np.zeros((len(data_size(dimension)), len(sigmas), nb_tries))
    test_loss_0_array = np.zeros((len(data_size(dimension)), len(sigmas), nb_tries))
    test_loss_star_array = np.zeros((len(data_size(dimension)), len(sigmas), nb_tries))
    test_loss_r_array = np.zeros((len(data_size(dimension)), len(sigmas)))

    # get the independent variable samples
    test_x = (b - a) * MDG.uniform_unit_hypercube(generator, dimension - 1, 1000) + a

    # solve the robust optimization problem
    robust_opt = RobustOptimization(ellipsoid)
    robust_opt.solve_least_squares()
    loss_r = robust_opt.cost

    for i, m in enumerate(data_size(dimension)):
        print(f"{datetime.now()} - m = {m}")
        for j, sigma in enumerate(sigmas):

            # generate i.i.d. test data
            test_y = (np.array([np.dot(test_x[:, k], slope) for k in range(1000)]) +
                      MDG.normal_disturbance(generator, sigma, 1000, True))
            test_data = np.vstack([test_x, test_y])

            for k in range(nb_tries):
                # training data
                x = MDG.uniform_unit_hypercube(generator, dimension - 1, m)
                y = (np.array([np.dot(x[:, i], slope) for i in range(m)]) +
                     MDG.normal_disturbance(generator, sigma, m, True))
                training_data = np.vstack([x, y])
                MDG.contain_in_ellipsoid(generator, training_data, ellipsoid, slope)

                # define problem
                problem = LeastSquaresCadro(training_data, ellipsoid)

                # solve
                problem.solve()

                # fill in loss arrays
                test_loss_0 = problem.test_loss(test_data, 'theta_0')
                test_loss_star = problem.test_loss(test_data, 'theta')
                test_loss_0_array[i, j, k] = test_loss_0
                test_loss_star_array[i, j, k] = test_loss_star
                if k == 0:
                    test_loss_r_array[i, j] = problem.test_loss(test_data, 'theta_r')

                # fill in lambda array
                lambda_array[i, j, k] = problem.results["lambda"]

                # fill in alpha array
                alpha_array[i, j, k] = problem.results["alpha"][0]

    # make the plot for the alphas: boxplot combined with scatterplot
    fig, ax = plt.subplots(len(data_size(dimension)), len(sigmas))
    for i in range(len(data_size(dimension))):
        for j in range(len(sigmas)):
            ind = np.where(test_loss_0_array[i, j, :] < 10 * np.median(test_loss_0_array[i, j, :]))
            alpha_plot = alpha_array[i, j, ind][0]
            lambda_plot = lambda_array[i, j, ind][0]
            aux.plot_alphas(ax[i, j], alpha_plot, lambda_plot, loss_r,
                            title=r"$m = " + str(data_size(dimension)[i]) + r", \sigma = " + str(sigmas[j]) + r"$",
                            boxplot=True)

    fig.suptitle(f"Dimension {dimension} - {ellipsoid.type} ellipsoid")
    plt.tight_layout()
    plt.savefig("thesis_figures/multivariate_ls/full_exp/alphas_d" + str(dimension) + "_" + ellipsoid.type + ".png")
    plt.close()

    # make the plot for the loss histograms: overlaying histograms for loss_0 and loss_star
    fig, ax = plt.subplots(len(data_size(dimension)), len(sigmas))
    for i in range(len(data_size(dimension))):
        for j in range(len(sigmas)):
            # only keep the losses where the loss_0 is not too large
            ind = np.where(test_loss_0_array[i, j, :] < 10 * np.median(test_loss_0_array[i, j, :]))
            test_loss_0_plot = test_loss_0_array[i, j, ind][0]
            test_loss_star_plot = test_loss_star_array[i, j, ind][0]
            aux.plot_loss_histograms(ax[i, j], test_loss_0_plot, test_loss_star_plot, test_loss_r_array[i, j],
                                     title=r"$m = " + str(data_size(dimension)[i]) + r", \sigma = " + str(
                                         sigmas[j]) + r"$",
                                     bins=20)

    fig.suptitle(f"Dimension {dimension} - {ellipsoid.type} ellipsoid")
    plt.tight_layout()
    plt.savefig("thesis_figures/multivariate_ls/full_exp/hist_loss_d" + str(dimension) + "_" + ellipsoid.type + ".png")
    plt.close()

    # get all numeric data
    median_alpha = np.median(alpha_array, axis=2)
    upq_alpha = np.percentile(alpha_array, 75, axis=2)
    downq_alpha = np.percentile(alpha_array, 25, axis=2)
    median_loss_0 = np.median(test_loss_0_array, axis=2)
    upq_loss_0 = np.percentile(test_loss_0_array, 75, axis=2)
    downq_loss_0 = np.percentile(test_loss_0_array, 25, axis=2)
    median_loss_star = np.median(test_loss_star_array, axis=2)
    upq_loss_star = np.percentile(test_loss_star_array, 75, axis=2)
    downq_loss_star = np.percentile(test_loss_star_array, 25, axis=2)

    # construct dataframes
    df_alpha = pd.DataFrame(columns=sigmas, index=data_size(dimension))
    df_loss_0 = pd.DataFrame(columns=sigmas, index=data_size(dimension))
    df_loss_star = pd.DataFrame(columns=sigmas, index=data_size(dimension))

    # formatting: median (lower quantile, upper quantile)
    for i in range(len(data_size(dimension))):
        for j in range(len(sigmas)):
            df_alpha.loc[data_size(dimension)[i], sigmas[
                j]] = f"{round(median_alpha[i, j], 3)} ({round(downq_alpha[i, j], 3)}, {round(upq_alpha[i, j], 3)})"
            df_loss_0.loc[data_size(dimension)[i], sigmas[
                j]] = f"{round(median_loss_0[i, j], 3)} ({round(downq_loss_0[i, j], 3)}, {round(upq_loss_0[i, j], 3)})"
            df_loss_star.loc[data_size(dimension)[i], sigmas[
                j]] = f"{round(median_loss_star[i, j], 3)} ({round(downq_loss_star[i, j], 3)}, {round(upq_loss_star[i, j], 3)})"

    # return the numeric data
    return df_alpha, df_loss_0, df_loss_star, [loss_r, np.mean(test_loss_r_array)]


def experiment3_2d(seed):
    """
    Test the effect of rotating the LJ ellipsoid on the CADRO method for the 2D case
    """
    # generate the data
    d = 2
    m = 30
    sigma = 1
    rico = 1
    nb_tries = 100
    a, b = -5, 5
    assert b > a
    generator = np.random.default_rng(seed)
    slope = rico * np.ones((d - 1,))

    emp_slope = slope
    lj = Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)), -4, 4, theta=emp_slope,
                                        scaling_factor=1.05)
    if d == 2:
        plt.figure()
        lj.plot(color='r', label="LJ")
        # plot a line along the actual slope
        plt.plot([a, b], [a * slope[0], b * slope[0]], color='g', label="Slope")

    angles_pos = np.linspace(np.deg2rad(10), np.deg2rad(70), 7)
    angles_neg = - np.linspace(np.deg2rad(70), np.deg2rad(10), 7)
    # concatenate and include 0
    angles = np.concatenate((angles_neg, [0], angles_pos))
    if d == 2:
        colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))
    ellipsoids = []

    test_x = MDG.uniform_unit_hypercube(generator, d - 1, 1000)
    test_y = np.array([np.dot(test_x[:, i], slope) for i in range(1000)]) + MDG.normal_disturbance(generator, sigma,
                                                                                                   1000, True)
    test_data = np.vstack([test_x, test_y])
    for angle in angles:
        R = aux.rotation_matrix(d, angle, components=[0, 1])
        # for higher dimensions, we do not need the shape and center matrices
        shape = R.T @ lj.shape @ R if lj.shape is not None else None
        center = R.T @ lj.center if lj.center is not None else None
        ellipsoid = Ellipsoid(R.T @ lj.A @ R, R.T @ lj.a, lj.c, shape, center)
        ellipsoid.type = "LJ_rotated"
        ellipsoids.append(ellipsoid)
        if d == 2 and angle != 0:
            ellipsoid.plot(color=colors[np.where(angles == angle)[0][0]], alpha=0.5)

    if d == 2:
        plt.legend()
        plt.grid()
        plt.savefig("thesis_figures/multivariate_ls/rotations/ellipsoids.png")

    alpha_array = np.zeros((len(ellipsoids), nb_tries))
    lambda_array = np.zeros((len(ellipsoids), nb_tries))
    loss_r_array = np.zeros((len(ellipsoids)))
    test_loss_r_array = np.zeros((len(ellipsoids)))

    test_loss_0_array = np.zeros((len(ellipsoids), nb_tries))
    test_loss_star_array = np.zeros((len(ellipsoids), nb_tries))

    for i, ellipsoid in enumerate(ellipsoids):

        # get the robust cost
        robust_opt = RobustOptimization(ellipsoid)
        robust_opt.solve_least_squares()
        loss_r_array[i] = robust_opt.cost

        for k in range(nb_tries):
            # sample uniformly from the unit hypercube
            x = MDG.uniform_unit_hypercube(generator, d - 1, m)
            y = np.array([np.dot(x[:, i], slope) for i in range(m)]) + MDG.normal_disturbance(generator, sigma, m)
            data = np.vstack([x, y])
            MDG.contain_in_ellipsoid(generator, data, ellipsoid, slope)

            # solve the CADRO problem
            problem = LeastSquaresCadro(data, ellipsoid, solver=cp.MOSEK)
            problem.solve()

            # collect the results
            alpha_array[i, k] = problem.results["alpha"][0]
            lambda_array[i, k] = problem.results["lambda"][0]
            test_loss_0_array[i, k] = problem.test_loss(test_data, 'theta_0')
            test_loss_star_array[i, k] = problem.test_loss(test_data, 'theta')

            if k == 0:
                test_loss_r_array[i] = problem.test_loss(test_data, 'theta_r')

        print("theta_r", problem.theta_r)

    # make the plot for the loss histograms
    print("histograms")
    width, height = 5, 3
    assert width * height >= len(ellipsoids)  # make sure we have enough subplots
    fig, ax = plt.subplots(ncols=width, nrows=height)
    for i in range(len(ellipsoids)):
        # remove outliers: only keep the losses where the loss_0 is not too large
        # indices = np.where(test_loss_0_array < 10 * np.median(test_loss_0_array))
        aux.plot_loss_histograms(ax[i // width, i % width],
                                 test_loss_0_array[i, :], test_loss_star_array[i, :],
                                 test_loss_r_array[i], title=r"$\phi = {}$".format(round(np.rad2deg(angles[i]))),
                                 bins=30)
        # set xlim to [15, 18]
        ax[i // width, i % width].set_xlim([15, 18])
        # x ticks at 16, 17 and 18
        ax[i // width, i % width].set_xticks([15, 16, 17, 18])
    plt.tight_layout()
    plt.savefig("thesis_figures/multivariate_ls/rotations/hist_loss_all_d" + str(d) + ".png")
    plt.show()

    # make the plot for the alphas: boxplot combined with scatterplot
    print("alphas")
    fig, ax = plt.subplots(ncols=len(ellipsoids))
    # set all axes to the same limits
    max_limit = np.max(alpha_array)
    min_limit = np.min(alpha_array)
    for i in range(len(ellipsoids)):
        aux.plot_alphas(ax[i], alpha_array[i, :], lambda_array[i, :], loss_r_array[i],
                        title=str(round(np.rad2deg(angles[i]))) + r"$^{\circ}$",
                        boxplot=True)
        # log scale for y
        ax[i].set_yscale('log')
        ax[i].set_ylim([min_limit, max_limit])

        if i > 0:
            ax[i].set_yticks([])

    plt.tight_layout()
    plt.savefig(f"thesis_figures/multivariate_ls/rotations/alphas_d{d}.png")
    plt.show()

    # plot the robust test loss as a function of the angle
    print("loss")
    plt.figure()
    plt.plot(np.rad2deg(angles), test_loss_r_array, marker='o')
    plt.xlabel(r"$\phi$ (degrees)")
    plt.ylabel(r"$\ell(\theta_r)$")
    plt.grid()
    plt.tight_layout()
    plt.savefig("thesis_figures/multivariate_ls/rotations/loss_angle_d" + str(d) + ".png")
    plt.show()


def experiment3_md(seed, d: int = 3):
    """
    Test the effect of rotating the LJ ellipsoid on the CADRO method for the 2D case
    """
    # generate the data
    m = 12 * d
    sigma = 1
    actual_rico = 1
    actual_slope = actual_rico * np.ones((d - 1,))
    nb_tries = 100
    a, b = -5, 5
    assert b > a
    generator = np.random.default_rng(seed)

    indices = range(-10, 10)

    ellipsoids = []
    ricos = []

    for index in indices:
        rico_rot = (index + np.sqrt(d - 1)) / np.sqrt(d - 1)
        slope_rot = rico_rot * np.ones((d - 1,))
        ellipsoids.append(Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)),
                                                         -2, 2, theta=slope_rot, scaling_factor=1.05))
        ricos.append(rico_rot)

    alpha_array = np.zeros((len(ellipsoids), nb_tries))
    lambda_array = np.zeros((len(ellipsoids), nb_tries))

    loss_r_array = np.zeros((len(ellipsoids)))
    test_loss_r_array = np.zeros((len(ellipsoids)))

    test_x = a + (b - a) * MDG.uniform_unit_hypercube(generator, d - 1, 1000)
    test_y = np.array([np.dot(test_x[:, i], actual_slope) for i in range(1000)]) + \
             MDG.normal_disturbance(generator, sigma, 1000, True)
    test_data = np.vstack([test_x, test_y])

    # figure for the loss histograms
    width, height = 5, 4
    assert width * height >= len(ellipsoids)  # make sure we have enough subplots
    fig, ax = plt.subplots(ncols=width, nrows=height)

    print(f"{datetime.now()} - Dimension {d} - Starting the loop")
    for i in range(len(ellipsoids)):
        # setup
        slope = ricos[i] * np.ones((d - 1,))
        ellipsoid = ellipsoids[i]
        test_loss_0_array = np.zeros((nb_tries))
        test_loss_star_array = np.zeros((nb_tries))

        # get the robust cost
        robust_opt = RobustOptimization(ellipsoid)
        robust_opt.solve_least_squares()
        loss_r_array[i] = robust_opt.cost

        print(f"{datetime.now()} - Slope {ricos[i]}")
        for k in range(nb_tries):
            # generate training data
            x = a + (b - a) * MDG.uniform_unit_hypercube(generator, d - 1, m)
            y = (np.array([np.dot(x[:, i], actual_slope) for i in range(m)]) +
                 MDG.normal_disturbance(generator, sigma, m))
            data = np.vstack([x, y])
            MDG.contain_in_ellipsoid(generator, data, ellipsoid, slope)

            # solve the CADRO problem
            problem = LeastSquaresCadro(data, ellipsoid, solver=cp.MOSEK)
            problem.solve()

            # collect the results
            alpha_array[i, k] = problem.results["alpha"][0]
            lambda_array[i, k] = problem.results["lambda"][0]
            test_loss_0_array[k] = problem.test_loss(test_data, 'theta_0')
            test_loss_star_array[k] = problem.test_loss(test_data, 'theta')

            if k == 0:
                test_loss_r_array[i] = problem.test_loss(test_data, 'theta_r')

        # update figure
        indices = np.where(test_loss_0_array <= 10 * np.median(test_loss_0_array))
        aux.plot_loss_histograms(ax[i // width, i % width],
                                 test_loss_0_array[indices], test_loss_star_array[indices],
                                 test_loss_r_array[i], title=f"{round(ricos[i], 2)}", bins=20)

    plt.tight_layout()
    plt.savefig("thesis_figures/multivariate_ls/rotations/hist_loss_all_d" + str(d) + ".png")
    plt.show()

    # make the plot for the alphas: boxplot combined with scatterplot
    print(f"{datetime.now()} - Alphas")
    fig, ax = plt.subplots(ncols=len(ellipsoids))
    # set all axes to the same limits
    max_limit = np.max(alpha_array)
    min_limit = np.min(alpha_array)
    for i in range(len(ellipsoids)):
        title = str(round(ricos[i], 2)) if i % 3 == 0 else None
        aux.plot_alphas(ax[i], alpha_array[i, :], lambda_array[i, :], loss_r_array[i],
                        boxplot=True, title=title)
        ax[i].set_yscale('log')
        ax[i].set_ylim([min_limit, max_limit])

        if i > 0:
            ax[i].set_yticks([])

    plt.tight_layout()
    plt.savefig(f"thesis_figures/multivariate_ls/rotations/alphas_d{d}.png")
    plt.show()

    # plot the robust test loss as a function of the angle
    print(f"{datetime.now()} - Loss")
    plt.figure()
    plt.plot(ricos, test_loss_r_array, marker='o')
    plt.xlabel(r"slope")
    plt.ylabel(r"$\ell(\theta_r)$")
    plt.grid()
    plt.tight_layout()
    plt.savefig("thesis_figures/multivariate_ls/rotations/loss_angle_d" + str(d) + ".png")
    # plot a vertical line at the actual angle
    plt.axvline(actual_rico, color='r', linestyle='--')
    plt.show()


def experiment4(seed, solver, dimensions):
    """
    Time the CADRO method for different dimensions
    """
    generator = np.random.default_rng(seed)
    timings_mean_array = np.zeros((len(dimensions)))
    timings_std_array = np.zeros((len(dimensions)))
    nb_tries = 2
    nb_warmup = 2
    a, b = 0, 6
    assert b > a

    for n_d, d in enumerate(dimensions):
        with open("progress.txt", "a") as f:
            f.write('------------------------------------------------------\n')
            f.write(f"{datetime.now()} - Dimension {d} - Solver {solver}\n")

        # generate a dataset to calculate the supporting ellipses
        ubx = b * np.ones((d - 1,))
        lbx = a * np.ones((d - 1,))
        lbw, ubw = -2, 2
        slope = np.ones((d - 1,)) + np.clip(generator.normal(scale=0.5, size=(d - 1,)), -0.5, 0.5)
        ellipse = Ellipsoid.ellipse_from_corners(lbx, ubx, lbw, ubw, theta=slope, scaling_factor=1.05)

        if n_d == 0:
            for _ in range(nb_warmup):
                x = MDG.uniform_unit_hypercube(generator, d - 1, int(5 * d))
                y = np.array([np.dot(x[:, i], np.ones((d - 1,))) for i in range(int(5 * d))]) + \
                    MDG.normal_disturbance(generator, 1, int(5 * d), True)
                data = np.vstack([x, y])
                problem = LeastSquaresCadro(data, ellipse, solver=cp.MOSEK)
                problem.solve()

        robust_opt = RobustOptimization(ellipse)
        robust_opt.solve_least_squares()
        loss_r = robust_opt.cost

        timings_array = np.zeros((nb_tries))
        loss_0_array = np.zeros((nb_tries))
        loss_star_array = np.zeros((nb_tries))

        for k in range(nb_tries):
            # sample uniformly from the unit hypercube
            x = MDG.uniform_unit_hypercube(generator, d - 1, int(2 * d * np.log(d)))
            y = np.array([np.dot(x[:, i], np.ones((d - 1,))) for i in range(int(2 * d * np.log(d)))]) + \
                MDG.normal_disturbance(generator, 1, int(2 * d * np.log(d)), True)
            data = np.vstack([x, y])

            try:
                # time the CADRO method
                t1 = time()
                problem = LeastSquaresCadro(data, ellipse, solver=solver)
                problem.solve()
                t2 = time()

                # collect timings
                timings_array[k] = t2 - t1
                loss_0_array[k] = problem.test_loss(data, 'theta_0')
                loss_star_array[k] = problem.test_loss(data, 'theta')
            except cp.error.SolverError:
                timings_array[k] = np.nan
                loss_0_array[k] = np.nan
                loss_star_array[k] = np.nan

        # calculate the mean and standard deviation of the timings
        if sum(np.isnan(timings_array)) > int(0.5 * nb_tries):

            if sum(np.isnan(timings_array)) == nb_tries:
                mean_time = np.nan
                std_time = np.nan
            else:
                mean_time = np.mean(timings_array, where=~np.isnan(timings_array))
                std_time = np.std(timings_array, where=~np.isnan(timings_array))

            with open("progress.txt", "a") as f:
                f.write(f"{datetime.now()} - Plotting ... \n")
                f.write(f"Time for dimension {d}: {mean_time} (+/- {std_time}) (s) \n")
                f.write(f"Solver error in {np.sum(np.isnan(timings_array))} cases. Aborting. \n")
                return
        else:
            mean_time = np.mean(timings_array, where=~np.isnan(timings_array))
            std_time = np.std(timings_array, where=~np.isnan(timings_array))
            timings_mean_array[n_d] = mean_time
            timings_std_array[n_d] = std_time

            with open("progress.txt", "a") as f:
                f.write(f"{datetime.now()} - Plotting ... \n")
                f.write(f"Time for dimension {d}: {mean_time} (+/- {std_time}) (s) \n")
                f.write(f"Solver error in {np.sum(np.isnan(timings_array))} cases. Continuing. \n")

            # plot the timings
            plt.rcParams.update({'font.size': 15})
            aux.plot_timings(timings_mean_array[:n_d + 1], timings_std_array[:n_d + 1], dimensions[:n_d + 1])
            plt.grid()
            plt.title(solver)
            plt.tight_layout()
            plt.savefig(f"timings_{solver}.png")
            plt.close()
            plt.rcParams.update({'font.size': 10})

            # make the plot for the loss histograms: overlaying histograms for loss_0 and loss_star
            loss_0_array = loss_0_array[~np.isnan(loss_0_array)]
            loss_star_array = loss_star_array[~np.isnan(loss_star_array)]
            plt.figure()
            aux.plot_loss_histograms(plt.gca(), loss_0_array, loss_star_array, loss_r, bins=100,
                                     title=f"Dimension {d} - {solver}")
            plt.tight_layout()
            plt.savefig("hist_loss_d" + str(d) + "_" + ellipse.type + "_" + solver + ".png")
            plt.close()


def experiment5(seed):
    """
    Plot the difference between theta_star and theta_0/theta_r. Also plot separate figures for the loss histograms
    and alpha values.
    """
    plt.rcParams.update({'font.size': 15})
    dimensions = [10]
    # a, b = -5, 5
    a, b = -2, 2
    assert b > a
    generator = np.random.default_rng(seed)
    nb_tries = 100

    # data_size = lambda d: [2 * d, 5 * d, 8 * d, 12 * d, 20 * d, 25 * d]
    # sigmas = [0.2, 0.5] #, 1, 2]
    # sigmas = [0.5, 1]
    data_size = lambda d: 5 * np.logspace(1, 6, 15, dtype=int)
    sigmas = [1]

    for n_d, d in enumerate(dimensions):
        ms = data_size(d)
        slope = np.ones((d - 1,))

        print(f"{datetime.now()} - Dimension {d} - Ellipsoid construction")
        emp_slope = slope + np.clip(generator.normal(scale=0.5, size=(d - 1,)), -0.5, 0.5)  # random disturbance
        lj = Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)), -2, 2, theta=emp_slope,
                                            scaling_factor=1.05)
        lj.type = "LJ"

        delta_w = (b - a) / 2
        ses = Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)), -delta_w, delta_w,
                                             theta=emp_slope, scaling_factor=1.05)
        ses.type = "SCC"

        sigma_subg = aux.subgaussian_parameter(d, a, b, 2, -2, emp_slope)

        ellipsoids = [lj, ses]

        for ellipsoid in ellipsoids:
            dist_star_0 = np.zeros((len(ms), len(sigmas), nb_tries))
            dist_star_r = np.zeros((len(ms), len(sigmas), nb_tries))
            dist_r_0 = np.zeros((len(ms), len(sigmas), nb_tries))

            test_loss_0 = np.zeros((len(ms), len(sigmas), nb_tries))
            test_loss_star = np.zeros((len(ms), len(sigmas), nb_tries))
            test_loss_dro = np.zeros((len(ms), len(sigmas), nb_tries))
            test_loss_r = np.zeros((len(ms), len(sigmas)))

            alpha_array = np.zeros((len(ms), len(sigmas), nb_tries))
            lambda_array = np.zeros((len(ms), len(sigmas), nb_tries))

            # get robust cost
            robust_opt = RobustOptimization(ellipsoid)
            robust_opt.solve_least_squares()
            theta_r = robust_opt.theta
            cost_r = robust_opt.cost

            for i, m in enumerate(ms):
                print(f"{datetime.now()} - m = {m}")
                for j, sigma in enumerate(sigmas):
                    print(f"{datetime.now()} - sigma = {sigma}")
                    test_x = (b - a) * MDG.uniform_unit_hypercube(generator, d - 1, 1000) + a
                    test_y = np.array([np.dot(test_x[:, k], slope) for k in range(1000)]) + \
                             MDG.normal_disturbance(generator, sigma, 1000, True)
                    test_data = np.vstack([test_x, test_y])
                    for k in range(nb_tries):
                        # sample uniformly from the unit hypercube
                        x = (b - a) * MDG.uniform_unit_hypercube(generator, d - 1, m) + a
                        y = np.array([np.dot(x[:, i], slope) for i in range(m)]) + \
                            MDG.normal_disturbance(generator, 2, m, True)
                        data = np.vstack([x, y])
                        MDG.contain_in_ellipsoid(generator, data, ellipsoid, slope)

                        # solve the CADRO problem
                        problem = LeastSquaresCadro(data, ellipsoid, solver=cp.MOSEK)
                        problem.solve()

                        # solve the moment DRO problem
                        dro = MomentDRO(ellipsoid, data, confidence=0.05, solver=cp.MOSEK, sigmaG=sigma_subg)
                        theta_dro = dro.solve(check_data=False)

                        # fill in the distance arrays
                        dist_star_0[i, j, k] = np.linalg.norm(problem.results["theta"] - problem.results["theta_0"])
                        dist_star_r[i, j, k] = np.linalg.norm(problem.results["theta"] - theta_r)
                        dist_r_0[i, j, k] = np.linalg.norm(problem.results["theta_r"] - problem.results["theta_0"])

                        # fill in the loss arrays
                        test_loss_0[i, j, k] = problem.test_loss(test_data, 'theta_0')
                        test_loss_star[i, j, k] = problem.test_loss(test_data, 'theta')
                        test_loss_dro[i, j, k] = dro.test_loss(test_data, theta_dro)
                        if k == 0:
                            test_loss_r[i, j] = problem.test_loss(test_data, 'theta_r')

                        # fill in lambda array and alpha array
                        lambda_array[i, j, k] = problem.results["lambda"][0]
                        alpha_array[i, j, k] = problem.results["alpha"][0]

                    # # remove outliers:
                    # # get the indices where the distances are not too large (w.r.t. dist_star_0)
                    # ind = np.where(dist_star_0[i, j, :] < 10 * np.median(dist_star_0[i, j, :]))
                    # dist_star_0_plot = dist_star_0[i, j, ind]
                    # dist_star_r_plot = dist_star_r[i, j, ind]
                    #
                    # # plot the resulting distance-distance plot
                    # plt.figure()
                    # # respectively
                    # dist_r_0_median = np.median(dist_r_0[i, j, :])
                    # dist_r_0_p25 = np.percentile(dist_r_0[i, j, :], 25)
                    # dist_r_0_p75 = np.percentile(dist_r_0[i, j, :], 75)
                    #
                    # # plot the median distance between theta_r and theta_0 as a line
                    # line = [[0, dist_r_0_median], [dist_r_0_median, 0]]
                    # plt.plot(line[0], line[1], color='r', label=r'$\|\theta_r - \theta_0\|$')
                    # # plot the 25th and 75th percentiles
                    # line = [[0, dist_r_0_p25], [dist_r_0_p25, 0]]
                    # plt.plot(line[0], line[1], color='r', linestyle='--')
                    # line = [[0, dist_r_0_p75], [dist_r_0_p75, 0]]
                    # plt.plot(line[0], line[1], color='r', linestyle='--')
                    #
                    # # ||theta_star - theta_0|| and ||theta_star - theta_r|| on the x and y-axis
                    # plt.scatter(dist_star_0_plot, dist_star_r_plot, marker='x')
                    #
                    # # layout
                    # plt.xlabel(r"$\|\theta^* - \theta_0\|$")
                    # plt.ylabel(r"$\|\theta^* - \theta_r\|$")
                    # plt.title(f"Dimension {d} - {ellipsoid.type} - m = {m} - sigma = {sigma}")
                    # maximum = (max(np.max(dist_star_0_plot), 5), max(np.max(dist_star_r_plot), 5))
                    # plt.ylim([0, maximum[1]])
                    # plt.xlim([0, maximum[0]])
                    # plt.grid()
                    # plt.legend()
                    # plt.tight_layout()
                    # plt.savefig(
                    #     f"thesis_figures/multivariate_ls/distances/distances_d{d}_{ellipsoid.type}_m{m}_sigma{sigma}.png")
                    # plt.show()
                    # plt.close()

                    # plot the loss histograms
                    # fig, ax = plt.subplots()
                    # aux.plot_loss_histograms(ax, test_loss_0[i, j, :], test_loss_star[i, j, :], test_loss_r[i, j],
                    #                          title=f"Dimension {d} - {ellipsoid.type} - m = {m} - sigma = {sigma}",
                    #                          bins=30)
                    #
                    # plt.tight_layout()
                    # plt.savefig(
                    #     f"thesis_figures/multivariate_ls/histograms/hist_loss_d{d}_{ellipsoid.type}_m{m}_sigma{sigma}.png")
                    # plt.close()
                    #
                    # # make the plot for the alphas: boxplot combined with scatterplot
                    # plt.rcParams.update({'font.size': 10})
                    # fig, ax = plt.subplots()
                    # aux.plot_alphas(ax, alpha_array[i, j, :], lambda_array[i, j, :], cost_r,
                    #                 title=None, boxplot=True, marker='o')
                    # fig.set_size_inches(4, 6)
                    # # make sure all labels are visible
                    # plt.tight_layout()
                    # plt.savefig(
                    #     f"thesis_figures/multivariate_ls/alphas/alphas_d{d}_{ellipsoid.type}_m{m}_sigma{sigma}.png")
                    # plt.close()
                    # plt.rcParams.update({'font.size': 15})

            # plot the average loss in function of m for every sigma

            # save the results in a file
            np.save(f"results_d{d}_{ellipsoid.type}_loss_0.npy", test_loss_0)
            np.save(f"results_d{d}_{ellipsoid.type}_loss_star.npy", test_loss_star)
            np.save(f"results_d{d}_{ellipsoid.type}_loss_dro.npy", test_loss_dro)
            np.save(f"results_d{d}_{ellipsoid.type}_loss_r.npy", test_loss_r)
            np.save("ms.npy", ms)

            for j, sigma in enumerate(sigmas):
                aux.plot_loss_m(plt.gca(), np.median(test_loss_0[:, j, :], axis=1),
                                np.percentile(test_loss_0[:, j, :], 75, axis=1),
                                np.percentile(test_loss_0[:, j, :], 25, axis=1),
                                np.median(test_loss_star[:, j, :], axis=1),
                                np.percentile(test_loss_star[:, j, :], 75, axis=1),
                                np.percentile(test_loss_star[:, j, :], 25, axis=1),
                                None, None, None,
                                ms, title=None)
                plt.xscale('log')
                plt.grid()

                M = 4_956_137
                indices_valid = np.where(ms > M)[0]
                indices_invalid = np.where(ms <= M)[0]
                indices_invalid = np.append(indices_invalid, indices_valid[0])
                invalid_dro_p50 = np.median(test_loss_dro[indices_invalid, j, :], axis=1)
                invalid_dro_p25 = np.percentile(test_loss_dro[indices_invalid, j, :], 25, axis=1)
                invalid_dro_p75 = np.percentile(test_loss_dro[indices_invalid, j, :], 75, axis=1)

                valid_dro = np.median(test_loss_dro[indices_valid, j, :], axis=1)
                valid_dro_p25 = np.percentile(test_loss_dro[indices_valid, j, :], 25, axis=1)
                valid_dro_p75 = np.percentile(test_loss_dro[indices_valid, j, :], 75, axis=1)

                plt.errorbar(ms[indices_invalid], invalid_dro_p50, yerr=[invalid_dro_p50 - invalid_dro_p25, invalid_dro_p75 - invalid_dro_p50],
                                fmt='o', color='r', label="invalid bounds")
                plt.errorbar(ms[indices_valid], valid_dro, yerr=[valid_dro - valid_dro_p25, valid_dro_p75 - valid_dro],
                                fmt='o', color='g', label="DRO")


                # draw a horizontal line at the robust solution
                plt.axhline(test_loss_r[0, j], color='black', linestyle='--', label="robust")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    f"loss_dro_m_d{d}_{ellipsoid.type}_sigma{sigma}.png")
                plt.close()

    plt.rcParams.update({'font.size': 10})


if __name__ == '__main__':
    seed = 0
    # experiment1(seed)
    # experiment2(seed)
    # experiment3_2d(seed)
    # dimensions = [4, 10, 20]
    # for d in dimensions:
    #     experiment3_md(seed, d)
    experiment5(seed)

    # solvers = ['SCS', 'CVXOPT', 'CLARABEL', 'MOSEK']
    # dimensions_lists = [[5, 10, 15, 20],
    #                     [5, 10, 15, 20, 25, 30, 35, 40, 100],
    #                     [5, 10, 15, 20, 25, 30, 35, 40, 100],
    #                     [5, 10, 15, 20, 25, 30, 35, 40, 100, 200, 500, 1000]]
    #
    # for i in range(len(solvers)):
    #     experiment4(seed, solvers[i], dimensions_lists[i])
