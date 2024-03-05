import numpy as np
import matplotlib.pyplot as plt
from ellipsoids import Ellipsoid
from robust_optimization import RobustOptimization
from multiple_dimension_cadro import LeastSquaresCadro
import cvxpy as cp
from utils.data_generator import MultivariateDataGenerator as MDG
import pandas as pd
from time import time
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
    dimensions = [5, 10, 15, 20, 25, 30]

    loss_r_array = np.zeros((2, len(dimensions)))

    for n_d, d in enumerate(dimensions):
        # generate a dataset to calculate the supporting ellipses
        support_generating_x = MDG.uniform_unit_hypercube(generator, d - 1, int(5 * d * np.log(d)))
        support_generating_y = (np.array([np.dot(support_generating_x[:, i], np.ones((d - 1,))) for i in
                                          range(int(5 * d * np.log(d)))])
                                + MDG.normal_disturbance(generator, 1, int(5 * d * np.log(d)), True))
        support_generating = np.vstack([support_generating_x, support_generating_y])

        # calculate the two supporting ellipsoids: Löwner-John and smallest enclosing sphere
        lj = Ellipsoid.lj_ellipsoid(support_generating)
        ses = Ellipsoid.smallest_enclosing_sphere(support_generating)

        # conduct experiments for Löwner-John ellipsoid
        alpha_data, loss_0_data, loss_star_data, loss_r = experiment2_loop(d, lj, generator, excel=True)

        loss_r_array[0, n_d] = loss_r

        # write the dataframe to a text file as latex tables and to an Excel file
        with pd.ExcelWriter(f"thesis_figures/multivariate_ls/d{d}_experiment2_lj.xlsx") as writer:
            alpha_data.to_excel(writer, sheet_name='alpha')
            loss_0_data.to_excel(writer, sheet_name='loss_0')
            loss_star_data.to_excel(writer, sheet_name='loss_star')

        with open(f"thesis_figures/multivariate_ls/d{d}_experiment2_lj.txt", "w") as f:
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
        alpha_data, loss_0_data, loss_star_data, loss_r = experiment2_loop(d, ses, generator, excel=True)

        loss_r_array[1, n_d] = loss_r

        # write the dataframe to a text file as latex tables and to an Excel file
        with pd.ExcelWriter(f"thesis_figures/multivariate_ls/d{d}_experiment2_ses.xlsx") as writer:
            alpha_data.to_excel(writer, sheet_name='alpha')
            loss_0_data.to_excel(writer, sheet_name='loss_0')
            loss_star_data.to_excel(writer, sheet_name='loss_star')

        with open(f"thesis_figures/multivariate_ls/d{d}_experiment2_ses.txt", "w") as f:
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


def plot_timings(timings_mean_array, timings_std_array, dimensions):
    # linear scale
    fig, ax = plt.subplots()
    ax.plot(dimensions, timings_mean_array, marker='o', linestyle='-', color='b')
    ax.fill_between(dimensions, timings_mean_array - timings_std_array, timings_mean_array + timings_std_array,
                    alpha=0.2, color='b')
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Time (s)")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.tight_layout()
    plt.savefig("timings.png")
    plt.close()


def experiment2_loop(dimension, ellipsoid, generator, excel=False):
    # general setup
    data_size = lambda d: [int(d * np.log(d)), int(2 * d * np.log(d)), int(5 * d * np.log(d))]
    sigmas = [0.5, 1, 2, 3]
    nb_tries = 100
    slope = [i + 1 for i in range(dimension - 1)]
    slope = np.array(slope) / np.linalg.norm(slope)

    alpha_array = np.zeros((len(data_size(dimension)), len(sigmas), nb_tries))
    lambda_array = np.zeros((len(data_size(dimension)), len(sigmas), nb_tries))
    test_loss_0_array = np.zeros((len(data_size(dimension)), len(sigmas), nb_tries))
    test_loss_star_array = np.zeros((len(data_size(dimension)), len(sigmas), nb_tries))

    # get the independent variable samples
    test_x = MDG.uniform_unit_hypercube(generator, dimension - 1, 1000)

    # solve the robust optimization problem
    robust_opt = RobustOptimization(ellipsoid)
    robust_opt.solve_least_squares()
    loss_r = robust_opt.cost

    for i, m in enumerate(data_size(dimension)):
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

                # fill in lambda array
                lambda_array[i, j, k] = problem.results["lambda"]

                # fill in alpha array
                alpha_array[i, j, k] = problem.results["alpha"][0]

    # make the plot for the alphas: boxplot combined with scatterplot
    fig, ax = plt.subplots(len(data_size(dimension)), len(sigmas))
    for i in range(len(data_size(dimension))):
        for j in range(len(sigmas)):
            ind_lambdas_1 = np.where(lambda_array[i, j, :] > 0.9)
            ind_lambdas_0 = np.where(lambda_array[i, j, :] < 0.1)
            ind_lambdas_else = np.where((lambda_array[i, j, :] <= 0.9) & (lambda_array[i, j, :] >= 0.1))
            # plot a horizontal line at the robust cost
            ax[i, j].axhline(loss_r, color='black', linestyle='dashed', linewidth=1)
            # boxplot, overlayed with the actual values of alpha
            ax[i, j].boxplot(alpha_array[i, j, :], showfliers=False)
            ax[i, j].scatter(np.ones(len(ind_lambdas_1[0])), alpha_array[i, j, ind_lambdas_1],
                             label=r"$\lambda \approx 1$", color='b', marker='.')
            ax[i, j].scatter(np.ones(len(ind_lambdas_0[0])), alpha_array[i, j, ind_lambdas_0],
                             label=r"$\lambda \approx 0$", color='r', marker='.')
            ax[i, j].scatter(np.ones(len(ind_lambdas_else[0])), alpha_array[i, j, ind_lambdas_else],
                             label=r"$\lambda$ otherwise", color='g', marker='.')
            ax[i, j].set_title(r"$m = " + str(data_size(dimension)[i]) + r", \sigma = " + str(sigmas[j]) + r"$")
            # remove x ticks
            ax[i, j].set_xticks([])

    fig.suptitle(f"Dimension {dimension} - {ellipsoid.type} ellipsoid")
    plt.tight_layout()
    plt.savefig("thesis_figures/multivariate_ls/alphas_d" + str(dimension) + "_" + ellipsoid.type + ".png")

    # make the plot for the loss histograms: overlaying histograms for loss_0 and loss_star
    fig, ax = plt.subplots(len(data_size(dimension)), len(sigmas))
    for i in range(len(data_size(dimension))):
        for j in range(len(sigmas)):
            hist_range = (min(np.min(test_loss_0_array[i, j, :]), np.min(test_loss_star_array[i, j, :])),
                          max(np.max(test_loss_0_array[i, j, :]), np.max(test_loss_star_array[i, j, :])))
            ax[i, j].hist(test_loss_0_array[i, j, :], bins=10, alpha=0.5, label=r"$\theta_0$", range=hist_range)
            ax[i, j].hist(test_loss_star_array[i, j, :], bins=10, alpha=0.5, label=r"$\theta$", range=hist_range)
            # add a vertical line for the robust cost if it is in the picture
            if hist_range[1] > loss_r > hist_range[0]:
                ax[i, j].axvline(loss_r, color='black', linestyle='dashed', linewidth=1)
            ax[i, j].set_title(r"$m = " + str(data_size(dimension)[i]) + r", \sigma = " + str(sigmas[j]) + r"$")

    fig.suptitle(f"Dimension {dimension} - {ellipsoid.type} ellipsoid")
    plt.tight_layout()
    plt.savefig("thesis_figures/multivariate_ls/hist_loss_d" + str(dimension) + "_" + ellipsoid.type + ".png")

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
    return df_alpha, df_loss_0, df_loss_star, loss_r


def experiment3(seed):
    """
    Perform one instance of the CADRO method
    """
    d = 2
    m = 20
    sigma = 1.5
    rico = 4
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
    R += 0.1 * generator.normal(size=(d, d), scale=7)

    # construct orthogonal matrix from R
    Q, _ = np.linalg.qr(R)

    ellipsoid = Ellipsoid.from_principal_axes(R, data, solver=cp.CVXOPT, verbose=True)
    lj = Ellipsoid.lj_ellipsoid(data)
    if d == 2:
        # plot data and ellipsoid
        plt.figure()
        plt.scatter(data[0, :], data[1, :])
        ellipsoid.plot(color='r')
        lj.plot(color='g')
        plt.show()


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

    for n_d, d in enumerate(dimensions):
        # generate a dataset to calculate the supporting ellipses
        support_generating_x = MDG.uniform_unit_hypercube(generator, d - 1, int(5 * d * np.log(d)))
        support_generating_y = (np.array([np.dot(support_generating_x[:, i], np.ones((d - 1,))) for i in
                                          range(int(5 * d * np.log(d)))])
                                + MDG.normal_disturbance(generator, 1, int(5 * d * np.log(d)), True))
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
                problem = LeastSquaresCadro(data, lj, solver=cp.SCS)
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
            problem = LeastSquaresCadro(data, lj, solver=cp.SCS)
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
        plt.rcParams.update({'font.size': 15})
        plot_timings(timings_mean_array[:n_d + 1], timings_std_array[:n_d + 1], dimensions[:n_d + 1])
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
    Plot the difference between theta_star and theta_0/theta_r
    """
    dimensions = [2, 5]
    generator = np.random.default_rng(seed)
    nb_tries = 100
    m = 30

    for n_d, d in enumerate(dimensions):
        # generate a dataset to calculate the supporting ellipses
        support_generating_x = MDG.uniform_unit_hypercube(generator, d - 1, 2 * m)
        support_generating_y = (np.array([np.dot(support_generating_x[:, i], np.ones((d - 1,))) for i in
                                          range(2 * m)])
                                + MDG.normal_disturbance(generator, 2, int(2 * m), True))
        support_generating = np.vstack([support_generating_x, support_generating_y])

        # calculate the LJ ellipsoid and the smallest enclosing sphere
        lj = Ellipsoid.lj_ellipsoid(support_generating)
        ses = Ellipsoid.smallest_enclosing_sphere(support_generating)
        ellipsoids = [lj, ses]

        for ellipsoid in ellipsoids:
            dist_star_0 = np.zeros((nb_tries))
            dist_star_r = np.zeros((nb_tries))

            # get robust cost
            robust_opt = RobustOptimization(ellipsoid)
            robust_opt.solve_least_squares()
            theta_r = robust_opt.theta

            for k in range(nb_tries):
                # sample uniformly from the unit hypercube
                x = MDG.uniform_unit_hypercube(generator, d - 1, m)
                y = np.array([np.dot(x[:, i], np.ones((d - 1,))) for i in range(m)]) + \
                    MDG.normal_disturbance(generator, 2, m, True)
                data = np.vstack([x, y])
                problem = LeastSquaresCadro(data, ellipsoid, solver=cp.MOSEK)
                problem.solve()
                dist_star_0[k] = np.linalg.norm(problem.results["theta"] - problem.results["theta_0"])
                dist_star_r[k] = np.linalg.norm(problem.results["theta"] - theta_r)

            # plot the results: ||theta_star - theta_0|| and ||theta_star - theta_r|| on the x and y axis respectively
            plt.figure()
            plt.scatter(dist_star_0, dist_star_r, marker='x')
            plt.xlabel(r"$||\theta^* - \theta_0||$")
            plt.ylabel(r"$||\theta^* - \theta_r||$")
            plt.title(f"Dimension {d} - {ellipsoid.type} ellipsoid")
            maximum = max(np.max(dist_star_0), np.max(dist_star_r))
            plt.ylim([0, maximum])
            plt.xlim([0, maximum])
            plt.tight_layout()
            plt.savefig(f"thesis_figures/multivariate_ls/d{d}_distances_{ellipsoid.type}.png")
            plt.show()


if __name__ == '__main__':
    seed = 0
    # experiment1(seed)
    # experiment2(seed)
    experiment3(seed)
    # experiment4(seed)
    # experiment5(seed)
