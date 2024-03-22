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
from sklearn.decomposition import PCA


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
    dimensions = [2, 5] #, 10, 15, 20, 25, 30]
    a, b = 0, 20  # side lengths of the hypercube
    assert b > a

    for n_d, d in enumerate(dimensions):
        slope = [i + 1 for i in range(d - 1)]

        # we generate the bounding ellipses based on a bounding box around the data which we calculate a priori

        # get the corners of the hypercube [a, b]^(d-1)
        corners_x = aux.hypercube_corners(a, b, d - 1, 1e6, generator)


        print(f"{datetime.now()} - Dimension {d} - Ellipsoid construction")
        emp_slope = slope + np.clip(generator.normal(scale=0.5, size=(d - 1,)), -0.5, 0.5)  # random disturbance
        lj = Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)), -4, 4, theta=emp_slope,
                                            scaling_factor=1.05)
        lj.type = "LJ"
        ses = Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)), (a - b) / 2, (b - a) / 2,
                                             theta=emp_slope, scaling_factor=1.05)
        ses.type = "SES"

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

        # conduct experiments for smallest enclosing sphere
        # print(f"{datetime.now()} - Dimension {d}, SES ellipsoid")
        # alpha_data, loss_0_data, loss_star_data, loss_r = experiment2_loop(d, ses, generator, slope, a, b)
        #
        # # write the dataframe to a text file as latex tables and to an Excel file
        # with pd.ExcelWriter(f"thesis_figures/multivariate_ls/full_exp/d{d}_experiment2_ses.xlsx") as writer:
        #     alpha_data.to_excel(writer, sheet_name='alpha')
        #     loss_0_data.to_excel(writer, sheet_name='loss_0')
        #     loss_star_data.to_excel(writer, sheet_name='loss_star')
        #
        # with open(f"thesis_figures/multivariate_ls/full_exp/d{d}_experiment2_ses.txt", "w") as f:
        #     f.write("Alpha data \n")
        #     f.write(alpha_data.to_latex(float_format="%.0f"))
        #     f.write("\n")
        #     f.write("Loss 0 data \n")
        #     f.write(loss_0_data.to_latex(float_format="%.0f"))
        #     f.write("\n")
        #     f.write("Loss star data \n")
        #     f.write(loss_star_data.to_latex(float_format="%.0f"))
        #     f.write("\n")
        #     f.write(f"Robust cost: {loss_r}")


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


def experiment3(seed):
    """
    Construct rotated ellipses based on PCA

    Drop this altogether?
    """
    d = 2
    m = 20
    sigma = 1.5
    rico = 4
    a, b = 0, 10
    assert b > a

    generator = np.random.default_rng(seed)
    slope = rico * np.ones((d - 1,))

    # support generating data uniformly from the unit hypercube
    x = 3 * MDG.uniform_unit_hypercube(generator, d - 1, 4 * m)
    y = np.array([np.dot(x[:, i], slope) for i in range(4 * m)]) + MDG.normal_disturbance(generator, sigma, 4 * m,
                                                                                          False)

    data = np.vstack([x, y])
    # get the principal components
    pca = PCA(n_components=d)
    pca.fit(data.T)
    R = pca.components_
    # add a disturbance to R
    # put the columns in reverse order
    R = R[:, ::-1]
    R += generator.normal(scale=5, size=(d, d))

    # construct orthogonal matrix from R
    Q, _ = np.linalg.qr(R)

    ellipsoid = Ellipsoid.from_principal_axes(R, data, solver=cp.MOSEK, verbose=True, max_length=10, scaling_factor=1.5)
    lj = Ellipsoid.lj_ellipsoid(data)
    if d == 2:
        # plot data and ellipsoid
        plt.figure()
        plt.scatter(data[0, :], data[1, :])
        ellipsoid.plot(color='r')
        lj.plot(color='g')
        plt.show()


def experiment3a(seed):
    """
    Test the effect of rotating the LJ ellipsoid on the CADRO method
    """
    # generate the data
    d = 2
    m = 40
    sigma = 2
    rico = 1
    nb_tries = 100
    a, b = -5, 5
    assert b > a
    generator = np.random.default_rng(seed)
    slope = rico * np.ones((d - 1,))

    x = (b - a) * MDG.uniform_unit_hypercube(generator, d - 1, m) + a
    y = np.array([np.dot(x[:, i], slope) for i in range(m)]) + MDG.normal_disturbance(generator, sigma, m,
                                                                                      outliers=True)
    # subtract the mean from the data
    data = np.vstack([x, y])

    # get the ellipsoid
    corners_x = aux.hypercube_corners(a, b, d - 1, 1e6, generator)
    emp_slope = slope + np.clip(generator.normal(scale=0.1, size=(d - 1,)), -0.2, 0.2)  # random disturbance
    lj, corners = aux.ellipse_from_corners(corners_x.T, theta=emp_slope, ub=8, lb=8, kind="lj", return_corners=True)

    plt.figure()
    plt.scatter(data[0, :], data[1, :], marker='.')
    lj.plot(color='r', label="LJ")
    # plot a line along the actual slope

    angles = - np.linspace(np.deg2rad(5), np.deg2rad(50), 10)
    colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))
    ellipsoids = [lj]

    test_x = MDG.uniform_unit_hypercube(generator, d - 1, 1000)
    test_y = np.array([np.dot(test_x[:, i], slope) for i in range(1000)]) + MDG.normal_disturbance(generator, sigma,
                                                                                                   1000, True)
    test_data = np.vstack([test_x, test_y])
    for angle in angles:
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ellipsoid = Ellipsoid(R.T @ lj.A @ R, R.T @ lj.a, lj.c, R.T @ lj.shape @ R, R.T @ lj.center)
        ellipsoids.append(ellipsoid)
        ellipsoid.plot(label=r'$\phi = ' + str(round(np.rad2deg(angle))) + r'^\circ$',
                       color=colors[np.where(angles == angle)[0][0]])
    plt.legend()
    plt.grid()
    plt.show()

    alpha_array = np.zeros((len(ellipsoids), nb_tries))
    lambda_array = np.zeros((len(ellipsoids), nb_tries))
    loss_r_array = np.zeros((len(ellipsoids)))
    test_loss_r_array = np.zeros((len(ellipsoids)))
    angles = np.array([0] + list(angles))

    for i, ellipsoid in enumerate(ellipsoids):
        test_loss_0_array = np.zeros((nb_tries))
        test_loss_star_array = np.zeros((nb_tries))

        # get the robust cost
        robust_opt = RobustOptimization(ellipsoid)
        robust_opt.solve_least_squares()
        loss_r_array[i] = robust_opt.cost

        for k in range(nb_tries):
            # sample uniformly from the unit hypercube
            x = MDG.uniform_unit_hypercube(generator, d - 1, m)
            y = np.array([np.dot(x[:, i], slope) for i in range(m)]) + MDG.normal_disturbance(generator, sigma, m)
            data = np.vstack([x, y])

            # solve the CADRO problem
            problem = LeastSquaresCadro(data, ellipsoid, solver=cp.MOSEK)
            problem.solve()

            # collect the results
            alpha_array[i, k] = problem.results["alpha"][0]
            lambda_array[i, k] = problem.results["lambda"]
            test_loss_0_array[k] = problem.test_loss(test_data, 'theta_0')
            test_loss_star_array[k] = problem.test_loss(test_data, 'theta')

            if k == 0:
                test_loss_r_array[i] = problem.test_loss(test_data, 'theta_r')

        # make the plot for the loss histograms: overlaying histograms for loss_0 and loss_star
        title = r"$\phi = " + str(round(np.rad2deg(angles[i]))) + r"^{\circ}$"
        fig, ax = plt.subplots()
        aux.plot_loss_histograms(ax, test_loss_0_array, test_loss_star_array, test_loss_r_array[i], title=title,
                                 bins=20)
        plt.tight_layout()
        plt.savefig("thesis_figures/multivariate_ls/rotations/hist_loss_" + str(round(np.rad2deg(angles[i]))) + ".png")
        plt.show()

        print("theta_r", problem.theta_r)

    # make the plot for the alphas: boxplot combined with scatterplot
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
    plt.savefig("thesis_figures/multivariate_ls/rotations/alphas.png")
    plt.show()

    # define a rotation matrix (3d). First rotate around the z-axis, then around the y-axis
    # phi = np.deg2rad(30)
    # psi = np.deg2rad(20)
    # R = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]) @ \
    #     np.array([[np.cos(psi), 0, np.sin(psi)], [0, 1, 0], [-np.sin(psi), 0, np.cos(psi)]])

    # generate a random orthogonal matrix
    # R = np.random.rand(d, d)
    # R, _ = np.linalg.qr(R)


def experiment4(seed):
    """
    Time the CADRO method for different dimensions
    """
    dimensions = [5, 10, 15, 20, 25, 30, 35, 40, 100, 250, 500, 1000]
    generator = np.random.default_rng(seed)
    timings_mean_array = np.zeros((len(dimensions)))
    timings_std_array = np.zeros((len(dimensions)))
    nb_tries = 1000
    nb_warmup = 1000
    a, b = 0, 6
    assert b > a

    for n_d, d in enumerate(dimensions):
        with open("progress.txt", "a") as f:
            f.write(f"{datetime.now()} - Dimension {d} \n")

        # generate a dataset to calculate the supporting ellipses
        # we do not use the corner method here due to scalability issues but rather generate a large dataset
        # and limit the disturbance
        support_generating_x = (b - a) * MDG.uniform_unit_hypercube(generator, d - 1, int(5 * d * np.log(d))) + a
        support_generating_y = np.array([np.dot(support_generating_x[:, i], np.ones((d - 1,))) for i in
                                         range(int(5 * d * np.log(d)))])
        disturbance = MDG.normal_disturbance(generator, 1, int(5 * d * np.log(d)), False)
        disturbance = np.clip(disturbance, -2, 2)
        support_generating_y += disturbance
        support_generating = np.vstack([support_generating_x, support_generating_y])

        # calculate the LJ ellipsoid
        lj = Ellipsoid.lj_ellipsoid(support_generating)
        timings_array = np.zeros((nb_tries))
        loss_0_array = np.zeros((nb_tries))
        loss_star_array = np.zeros((nb_tries))

        if n_d == 0:
            for _ in range(nb_warmup):
                x = MDG.uniform_unit_hypercube(generator, d - 1, int(5 * d * np.log(d)))
                y = np.array([np.dot(x[:, i], np.ones((d - 1,))) for i in range(int(5 * d * np.log(d)))])
                data = np.vstack([x, y])
                problem = LeastSquaresCadro(data, lj, solver=cp.MOSEK)
                problem.solve()

        robust_opt = RobustOptimization(lj)
        robust_opt.solve_least_squares()
        loss_r = robust_opt.cost
        for k in range(nb_tries):
            # sample uniformly from the unit hypercube
            x = MDG.uniform_unit_hypercube(generator, d - 1, int(2 * d * np.log(d)))
            y = np.array([np.dot(x[:, i], np.ones((d - 1,))) for i in range(int(2 * d * np.log(d)))]) + \
                MDG.normal_disturbance(generator, 1, int(2 * d * np.log(d)), True)
            data = np.vstack([x, y])

            # time the CADRO method
            t1 = time()
            problem = LeastSquaresCadro(data, lj, solver=cp.MOSEK)
            problem.solve()
            t2 = time()

            # collect timings
            timings_array[k] = t2 - t1

            # collect losses
            loss_0_array[k] = problem.test_loss(data, 'theta_0')
            loss_star_array[k] = problem.test_loss(data, 'theta')

        # calculate the mean and standard deviation of the timings
        mean_time = np.mean(timings_array)
        std_time = np.std(timings_array)
        timings_mean_array[n_d] = mean_time
        timings_std_array[n_d] = std_time

        # plot the timings
        with open("progress.txt", "a") as f:
            f.write(f"{datetime.now()} - Plotting Dimension {d} \n")
        plt.rcParams.update({'font.size': 15})
        aux.plot_timings(timings_mean_array[:n_d + 1], timings_std_array[:n_d + 1], dimensions[:n_d + 1])
        plt.rcParams.update({'font.size': 10})

        # make the plot for the loss histograms: overlaying histograms for loss_0 and loss_star
        plt.figure()
        hist_range = (min(np.min(loss_0_array), np.min(loss_star_array)),
                      max(np.max(loss_0_array), np.max(loss_star_array)))
        plt.hist(loss_0_array, bins=100, alpha=0.5, label=r"$\theta_0$", range=hist_range)
        plt.hist(loss_star_array, bins=100, alpha=0.5, label=r"$\theta$", range=hist_range)
        # add a vertical line for the robust cost if it is in the picture
        if hist_range[1] > loss_r > hist_range[0]:
            plt.axvline(loss_r, color='black', linestyle='dashed', linewidth=1)
        plt.title(f"Dimension {d} - {lj.type} ellipsoid")
        plt.legend()
        plt.tight_layout()
        plt.savefig("hist_loss_d" + str(d) + "_" + lj.type + ".png")
        plt.close()


def experiment5(seed):
    """
    Plot the difference between theta_star and theta_0/theta_r. Also plot separate figures for the loss histograms
    and alpha values.
    """
    plt.rcParams.update({'font.size': 15})
    dimensions = [5, 10, 15]
    a, b = 0, 5
    assert b > a
    generator = np.random.default_rng(seed)
    nb_tries = 100

    data_size = lambda d: [2 * d, 3 * d, 4 * d, 5 * d, 9 * d, 14 * d]

    sigmas = [0.5, 1, 2]

    for n_d, d in enumerate(dimensions):
        ms = data_size(d)
        slope = np.ones((d - 1,))
        corners_x = aux.hypercube_corners(a, b, d - 1, 1e6, generator)
        lj = aux.ellipse_from_corners(corners_x.T, theta=slope, ub=6, lb=6, kind="lj")
        ses = aux.ellipse_from_corners(corners_x.T, theta=slope, ub=6, lb=6, kind="ses")
        ellipsoids = [lj, ses]

        for ellipsoid in ellipsoids:
            dist_star_0 = np.zeros((len(ms), len(sigmas), nb_tries))
            dist_star_r = np.zeros((len(ms), len(sigmas), nb_tries))

            test_loss_0 = np.zeros((len(ms), len(sigmas), nb_tries))
            test_loss_star = np.zeros((len(ms), len(sigmas), nb_tries))
            test_loss_r = np.zeros((len(ms), len(sigmas)))

            alpha_array = np.zeros((len(ms), len(sigmas), nb_tries))
            lambda_array = np.zeros((len(ms), len(sigmas), nb_tries))

            # get robust cost
            robust_opt = RobustOptimization(ellipsoid)
            robust_opt.solve_least_squares()
            theta_r = robust_opt.theta
            cost_r = robust_opt.cost

            for i, m in enumerate(ms):
                for j, sigma in enumerate(sigmas):
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
                        problem = LeastSquaresCadro(data, ellipsoid, solver=cp.MOSEK)
                        problem.solve()

                        # fill in the distance arrays
                        dist_star_0[i, j, k] = np.linalg.norm(problem.results["theta"] - problem.results["theta_0"])
                        dist_star_r[i, j, k] = np.linalg.norm(problem.results["theta"] - theta_r)

                        # fill in the loss arrays
                        test_loss_0[i, j, k] = problem.test_loss(test_data, 'theta_0')
                        test_loss_star[i, j, k] = problem.test_loss(test_data, 'theta')
                        if k == 0:
                            test_loss_r[i, j] = problem.test_loss(test_data, 'theta_r')

                        # fill in lambda array and alpha array
                        lambda_array[i, j, k] = problem.results["lambda"]
                        alpha_array[i, j, k] = problem.results["alpha"][0]

                    # remove outliers:
                    # get the indices where the distances are not too large (w.r.t. dist_star_0)
                    ind = np.where(dist_star_0[i, j, :] < 10 * np.median(dist_star_0[i, j, :]))
                    dist_star_0_plot = dist_star_0[i, j, ind]
                    dist_star_r_plot = dist_star_r[i, j, ind]

                    # plot the results: ||theta_star - theta_0|| and ||theta_star - theta_r|| on the x and y axis
                    # respectively
                    plt.figure()
                    plt.scatter(dist_star_0_plot, dist_star_r_plot, marker='x')
                    plt.xlabel(r"$||\theta^* - \theta_0||$")
                    plt.ylabel(r"$||\theta^* - \theta_r||$")
                    plt.title(f"Dimension {d} - {ellipsoid.type} - m = {m} - sigma = {sigma}")
                    maximum = (np.max(dist_star_0_plot), max(np.max(dist_star_r_plot), 50))
                    plt.ylim([0, maximum[1]])
                    plt.xlim([0, maximum[0]])
                    plt.grid()
                    plt.tight_layout()
                    plt.savefig(
                        f"thesis_figures/multivariate_ls/distances/distances_d{d}_{ellipsoid.type}_m{m}_sigma{sigma}.png")
                    plt.close()

                    # plot the loss histograms
                    fig, ax = plt.subplots()
                    aux.plot_loss_histograms(ax, test_loss_0[i, j, :], test_loss_star[i, j, :], test_loss_r[i, j],
                                             title=f"Dimension {d} - {ellipsoid.type} - m = {m} - sigma = {sigma}",
                                             bins=30)

                    plt.tight_layout()
                    plt.savefig(
                        f"thesis_figures/multivariate_ls/histograms/hist_loss_d{d}_{ellipsoid.type}_m{m}_sigma{sigma}.png")
                    plt.close()

                    # make the plot for the alphas: boxplot combined with scatterplot
                    plt.rcParams.update({'font.size': 10})
                    fig, ax = plt.subplots()
                    aux.plot_alphas(ax, alpha_array[i, j, :], lambda_array[i, j, :], cost_r,
                                    title=None, boxplot=True, marker='o')
                    fig.set_size_inches(4, 6)
                    # make sure all labels are visible
                    plt.tight_layout()
                    plt.savefig(
                        f"thesis_figures/multivariate_ls/alphas/alphas_d{d}_{ellipsoid.type}_m{m}_sigma{sigma}.png")
                    plt.close()
                    plt.rcParams.update({'font.size': 15})

        # plot the average loss in function of m for every sigma
        for j, sigma in enumerate(sigmas):
            fig, ax = plt.subplots()
            aux.plot_loss_m(ax, np.median(test_loss_0[:, j, :], axis=1),
                            np.percentile(test_loss_0[:, j, :], 75, axis=1),
                            np.percentile(test_loss_0[:, j, :], 25, axis=1), np.median(test_loss_star[:, j, :], axis=1),
                            np.percentile(test_loss_star[:, j, :], 75, axis=1),
                            np.percentile(test_loss_star[:, j, :], 25, axis=1),
                            ms, title=f"Dimension {d} - {ellipsoid.type} - sigma = {sigma}")

            plt.tight_layout()
            plt.savefig(
                f"thesis_figures/multivariate_ls/loss_m/loss_m_d{d}_{ellipsoid.type}_sigma{sigma}.png")
            plt.close()

    plt.rcParams.update({'font.size': 10})


if __name__ == '__main__':
    seed = 0
    # experiment1(seed)
    experiment2(seed)
    # experiment3a(seed)
    # experiment4(seed)
    # experiment5(seed)
