import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from datetime import datetime

import pandas as pd

from ellipsoids import Ellipsoid
from multiple_dimension_cadro import LeastSquaresCadro
from stochastic_dominance_cadro import StochasticDominanceCADRO
from utils.data_generator import MultivariateDataGenerator as MDG
import utils.multivariate_experiments as aux
from robust_optimization import RobustOptimization


def experiment1(seed):
    """
    Plot the difference between theta_star and theta_0/theta_r. Also plot separate figures for the loss histograms
    and alpha values.
    """
    plt.rcParams.update({'font.size': 15})
    dimensions = [5, 15]
    a, b = 0, 10
    assert b > a
    generator = np.random.default_rng(seed)
    nb_tries = 200

    data_size = lambda d: [2 * d, 5 * d, 8 * d, 10 * d, 15 * d]
    sigmas = [1, 2]

    for n_d, d in enumerate(dimensions):
        ms = data_size(d)
        slope = np.ones((d - 1,))

        with open('progress.txt', 'a') as f:
            f.write(f"{datetime.now()} - d = {d}\n")
        emp_slope = slope + np.clip(generator.normal(scale=0.5, size=(d - 1,)), -0.5, 0.5)  # random disturbance
        lj = Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)), -4, 4, theta=emp_slope,
                                            scaling_factor=1.05)
        lj.type = "LJ"

        delta_w = (b - a) / 2
        ses = Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)), -delta_w, delta_w,
                                             theta=emp_slope, scaling_factor=1.05)
        ses.type = "SES"

        ellipsoids = [lj, ses]

        for ellipsoid in ellipsoids:
            test_loss_0 = np.zeros((len(ms), len(sigmas), nb_tries))
            test_loss_star = np.zeros((len(ms), len(sigmas), nb_tries))
            test_loss_stoch_dom = np.zeros((len(ms), len(sigmas), nb_tries))
            test_loss_r = np.zeros((len(ms), len(sigmas)))

            distance_array = np.zeros((len(ms), len(sigmas), nb_tries))
            # 0 if theta_sd is close to theta_0, 1 if theta_sd is close to theta_r, 2 if neither

            # get robust cost
            robust_opt = RobustOptimization(ellipsoid)
            robust_opt.solve_least_squares()
            theta_r = robust_opt.theta

            for i, m in enumerate(ms):
                with open('progress.txt', 'a') as f:
                    f.write(f"{datetime.now()} - m = {m}\n")
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

                        # solve the CADRO problem
                        problem = LeastSquaresCadro(data, ellipsoid, solver=cp.MOSEK)
                        problem.solve()
                        theta_star = problem.theta

                        # solve the moment DRO problem
                        nodes = np.linspace(0, 0.25, 50)
                        nodes = np.append(nodes, 1)
                        stoch_dominance = StochasticDominanceCADRO(data, ellipsoid, threshold_type=nodes, solver=cp.MOSEK)
                        stoch_dominance.solve()
                        theta_sd = stoch_dominance.theta

                        # fill in the distance array
                        if np.linalg.norm(theta_sd - theta_star) < 1e-4 * np.linalg.norm(theta_star):
                            distance_array[i, j, k] = 0
                        elif np.linalg.norm(theta_sd - theta_r) < 1e-4 * np.linalg.norm(theta_r):
                            distance_array[i, j, k] = 1
                        else:
                            distance_array[i, j, k] = 2

                        # fill in the loss arrays
                        test_loss_0[i, j, k] = problem.test_loss(test_data, 'theta_0')
                        test_loss_star[i, j, k] = problem.test_loss(test_data, 'theta')
                        test_loss_stoch_dom[i, j, k] = stoch_dominance.test_loss(test_data, 'theta')
                        if k == 0:
                            test_loss_r[i, j] = problem.test_loss(test_data, 'theta_r')

                    # plot the loss histograms
                    # remove outliers
                    valid_indices = np.where(test_loss_0[i, j, :] < 10 * np.median(test_loss_0[i, j, :]))[0]
                    loss_0_plot = test_loss_0[i, j, valid_indices]
                    loss_stoch_dom_plot = test_loss_stoch_dom[i, j, valid_indices]

                    plt.figure()
                    aux.plot_loss_histograms(plt.gca(), loss_0_plot, loss_stoch_dom_plot, test_loss_r[i, j],
                                             bins=20)
                    plt.tight_layout()
                    plt.savefig(
                        f"thesis_figures/stoch_dom/loss_hist_d{d}_{ellipsoid.type}_m{m}_sigma{sigma}.png")
                    plt.close()

                    # save the losses to file
                    with open('results.txt', 'a') as f:
                        f.write(f"{datetime.now()} - d = {d}, ellipsoid = {ellipsoid.type}, sigma = {sigma}, m = {m}\n")
                        f.write(f"Loss 0: {np.median(test_loss_0[i, j, :])} ("
                                f"{np.percentile(test_loss_0[i, j, :], 25)}, "
                                f"{np.percentile(test_loss_0[i, j, :], 75)})\n")
                        f.write(f"Loss star: {np.median(test_loss_star[i, j, :])} ("
                                f"{np.percentile(test_loss_star[i, j, :], 25)}, "
                                f"{np.percentile(test_loss_star[i, j, :], 75)})\n")
                        f.write(f"Loss stoch dom: {np.median(test_loss_stoch_dom[i, j, :])} ("
                                f"{np.percentile(test_loss_stoch_dom[i, j, :], 25)}, "
                                f"{np.percentile(test_loss_stoch_dom[i, j, :], 75)})\n")
                        f.write(f"Loss r: {test_loss_r[:, j]}\n")
                        f.write("---------------------------------------------------------------\n")

            # plot the average loss in function of m for every sigma
            for j, sigma in enumerate(sigmas):
                fig, ax = plt.subplots()
                aux.plot_loss_m(ax, np.median(test_loss_0[:, j, :], axis=1),
                                np.percentile(test_loss_0[:, j, :], 75, axis=1),
                                np.percentile(test_loss_0[:, j, :], 25, axis=1),
                                np.median(test_loss_star[:, j, :], axis=1),
                                np.percentile(test_loss_star[:, j, :], 75, axis=1),
                                np.percentile(test_loss_star[:, j, :], 25, axis=1),
                                np.median(test_loss_stoch_dom[:, j, :], axis=1),
                                np.percentile(test_loss_stoch_dom[:, j, :], 75, axis=1),
                                np.percentile(test_loss_stoch_dom[:, j, :], 25, axis=1),
                                ms, title=None, label_dro="stoch. dom.", scale='linear')

                plt.tight_layout()
                plt.savefig(
                    f"thesis_figures/stoch_dom/loss_m_d{d}_{ellipsoid.type}_sigma{sigma}.png")
                plt.close()


            # gather the distance array in a table
            distance_df = pd.DataFrame(columns=sigmas, index=ms)
            for i, m in enumerate(ms):
                for j, sigma in enumerate(sigmas):
                    distance_df.loc[m, sigma] = (
                        np.sum(distance_array[i, j, :] == 0),
                        np.sum(distance_array[i, j, :] == 1),
                        np.sum(distance_array[i, j, :] == 2)
                    )

            # save to excel
            distance_df.to_csc(f"thesis_figures/multivariate_ls/thesis_figures/stoch_dom/distance_array_d{d}_{ellipsoid.type}.csv")

    plt.rcParams.update({'font.size': 10})


def experiment2(seed):
    generator = np.random.default_rng(seed)
    d = 15
    m = 120
    sigma = 2
    a, b = 0, 10

    slope = np.ones((d - 1,))
    emp_slope = slope + np.clip(generator.normal(scale=0.5, size=(d - 1,)), -0.5, 0.5)  # random disturbance
    ellipsoid = Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)), -4, 4, theta=emp_slope,
                                               scaling_factor=1.05)

    # generate 100 Chebyshev nodes on [0, 1]
    # nodes = 0.5 * np.cos((2 * np.arange(1, 101) - 1) * np.pi / 200) + 0.5

    # generate equally spaced nodes
    nodes = np.linspace(0, 0.1, 50)
    # add 1 to nodes
    nodes = np.append(nodes, 1)

    while True:
        x = (b - a) * MDG.uniform_unit_hypercube(generator, d - 1, m) + a
        y = np.array([np.dot(x[:, i], slope) for i in range(m)]) + \
            MDG.normal_disturbance(generator, sigma, m, True)
        data = np.vstack((x, y))

        problem = StochasticDominanceCADRO(data, ellipsoid, nb_thresholds=20, threshold_type=nodes)
        problem.set_theta_r()
        results = problem.solve()

        # use this code to generate a non-binary decision ------------------------------------
        # loss_r, loss, loss_0 = results["loss_r"], results["loss"], results["loss_0"]
        # if np.abs(loss_r - loss) > 0.1 * np.abs(loss_r) and np.abs(loss_0 - loss) > 0.1 * np.abs(loss_0):
        #     break
        # else:
        #     print("Retrying")
        # ---------------------------------------------------

        if True:
            break

    lambdas, alphas = results["lambda"], results["alpha"]
    nodes = problem.thresholds

    plt.scatter(nodes, alphas / np.max(alphas), color='b', marker='o', label='alpha (normalized)')
    plt.scatter(nodes, lambdas, marker='.', color='r', label='lambda')
    plt.xlabel('Threshold')
    plt.legend()
    plt.show()

    loss_0, loss_r, loss = results["loss_0"], results["loss_r"], results["loss"]
    dist_0 = np.linalg.norm(loss_0 - loss) / np.linalg.norm(loss)
    dist_r = np.linalg.norm(loss_r - loss) / np.linalg.norm(loss)
    print(f"Distance between l(theta_0) and l(theta_star): {dist_0}")
    print(f"Distance between l(theta_r) and l(theta_star): {dist_r}")
    print(f"l(theta_0) = {results['loss_0']} - l(theta_r) = {results['loss_r']} - l(theta_star) = {results['loss']}")


if __name__ == "__main__":
    seed = 0
    # experiment1(seed)
    experiment2(seed)
