import warnings
import numpy as np
import matplotlib.pyplot as plt
from ellipsoids import Ellipsoid
from robust_optimization import RobustOptimization
from multiple_dimension_cadro import LeastSquaresCadro
from utils.data_generator import MultivariateDataGenerator as MDG
import pandas as pd
from time import time


def experiment1(seed):
    """
    Experiment 1: Plot a test setup for a 3D CADRO model
    """
    plt.rcParams.update({'font.size': 15})

    generator = np.random.default_rng(seed)
    n = 20
    slope = 2 * np.ones((2,))
    # sample uniformly from the unit square
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
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
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
    plt.tight_layout()
    plt.show()


def experiment2(seed):
    """
    Experiment 2: For 5d linear regression, plot the realized values of alpha and lambda for illustrative purposes.
    """
    generator = np.random.default_rng(seed)
    m = 50
    sigma = 2
    rico = 3
    nb_tries = 50
    dimensions = [20, 25, 30, 40]

    # outcome: consists of 3-tuples [close to theta_r, close to theta_0, other]
    outcome_array = np.zeros((len(dimensions), 2, 3))
    test_loss_0_array = np.zeros((len(dimensions), 2, nb_tries))
    test_loss_star_array = np.zeros((len(dimensions), 2, nb_tries))
    lambda_array = np.zeros((len(dimensions), 2, nb_tries))
    timings = np.zeros((len(dimensions), 2, nb_tries))

    for k, d in enumerate(dimensions):
        slope = rico * np.ones((d - 1,))
        slope /= np.linalg.norm(slope)

        # support generating data
        x = MDG.uniform_unit_square(generator, d - 1, 2*m)
        y = (np.array([np.dot(x[:, i], slope) for i in range(2*m)]) +
             MDG.normal_disturbance(generator, sigma, 2*m, False))
        data = np.vstack([x, y])
        lj_ellipsoid = Ellipsoid.lj_ellipsoid(data)
        sphere = Ellipsoid.smallest_enclosing_sphere(data)

        ellipsoids = [lj_ellipsoid, sphere]

        # test data
        m_test = 1000
        x_test = MDG.uniform_unit_square(generator, d - 1, m_test)
        y_test = np.array([np.dot(x_test[:, i], slope) for i in range(m_test)]) + \
                 MDG.normal_disturbance(generator, sigma, m_test, False)
        test_data = np.vstack([x_test, y_test])

        for i, ellipsoid in enumerate(ellipsoids):
            for j in range(nb_tries):
                # training data
                x = MDG.uniform_unit_square(generator, d - 1, m)
                y = np.array([np.dot(x[:, i], slope) for i in range(m)]) + MDG.normal_disturbance(generator, sigma, m,
                                                                                                  False)
                training_data = np.vstack([x, y])
                MDG.contain_in_ellipsoid(generator, training_data, ellipsoid, slope)
                problem = LeastSquaresCadro(training_data, ellipsoid)
                t1 = time()
                problem.solve()
                t2 = time()
                problem.set_theta_r()
                theta_r = problem.theta_r
                theta_star = problem.theta
                theta_0 = problem.theta_0

                # fill in outcome array
                if np.linalg.norm(theta_star - theta_r) < 1e-3:
                    outcome_array[k, i, 0] += 1
                elif np.linalg.norm(theta_star - theta_0) < 1e-3:
                    outcome_array[k, i, 1] += 1
                else:
                    outcome_array[k, i, 2] += 1

                # fill in loss array
                test_loss_0 = problem.test_loss(test_data, 'theta_0')
                test_loss_star = problem.test_loss(test_data, 'theta')

                test_loss_0_array[k, i, j] = test_loss_0
                test_loss_star_array[k, i, j] = test_loss_star

                # fill in timing array
                timings[k, i, j] = t2 - t1

                # fill in lambda array
                lambda_array[k, i, j] = problem.results["lambda"]

            # plot a histogram for the losses for each dimension
            fig, ax = plt.subplots()
            hist_range = (min(np.min(test_loss_0_array[k, i, :]), np.min(test_loss_star_array[k, i, :])),
                          max(np.max(test_loss_0_array[k, i, :]), np.max(test_loss_star_array[k, i, :])))
            ax.hist(test_loss_0_array[k, i, :], bins=20, alpha=0.5, label=r"$\theta_0$", range=hist_range)
            ax.hist(test_loss_star_array[k, i, :], bins=20, alpha=0.5, label=r"$\theta$", range=hist_range)
            # add a vertical line for the robust cost
            ax.axvline(problem.test_loss(test_data, 'theta_r'), color='r', linestyle='dashed', linewidth=2)
            plt.title(f"Loss histogram for d = {d} ({ellipsoid.type})")
            plt.legend()
            plt.show()
            print(f"For d = {d} and ellipsoid type {ellipsoid.type}:")
            print(f"Robust cost: {problem.test_loss(test_data, 'theta_r')}")
            print(f"Final cost: {np.mean(test_loss_star_array[k, i, :])}")
            print("")

            # create a boxplot for the lambda values
            plt.figure()
            plt.boxplot(lambda_array[k, i, :])
            # plot all the lambda values
            plt.scatter(np.ones(nb_tries), lambda_array[k, i, :], marker='.')
            plt.title(f"Lambda values for d = {d} ({ellipsoid.type})")
            plt.show()

    for i, ellipsoid in enumerate(ellipsoids):
        # create a bar chart for the outcomes
        outcomes = outcome_array[:, i, :]
        categories = ("collapse", "data", "other")
        data = {categories[j]: outcomes[:, j] for j in range(outcomes.shape[1])}
        width = 1
        plt.figure()
        bottom = np.zeros(outcomes.shape[0])
        for label, weight_count in data.items():
            p = plt.bar(dimensions, weight_count, width, bottom=bottom, label=label)
            bottom += weight_count

        plt.legend()
        plt.title(f"Outcome for {ellipsoid.type} ellipsoid")
        plt.show()

    # create timing plot (for LJ)
    timings = timings[:, 0, :]
    means = np.mean(timings, axis=1)
    stds = np.std(timings, axis=1)
    plt.plot(dimensions, means, label="Mean", linestyle='-', marker='o')
    plt.fill_between(dimensions, means - stds, means + stds, alpha=0.2)
    plt.title("Mean time for solving the CADRO problem")
    plt.xlabel("Dimension")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    seed = 0
    # experiment1(seed)
    experiment2(seed)
