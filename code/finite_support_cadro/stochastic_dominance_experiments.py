import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from datetime import datetime
from ellipsoids import Ellipsoid
from multiple_dimension_cadro import LeastSquaresCadro
from stochastic_dominance_cadro import StochasticDominanceCADRO
from moment_dro import MomentDRO
from utils.data_generator import MultivariateDataGenerator as MDG
import utils.multivariate_experiments as aux


def experiment1(seed):
    """
    Plot the difference between theta_star and theta_0/theta_r. Also plot separate figures for the loss histograms
    and alpha values.
    """
    with open('progress_sd.txt', 'a') as f:
        f.write(f"{datetime.now()} - Starting experiment 1\n")

    # plt.rcParams.update({'font.size': 15})
    dimensions = [50]
    a, b = -5, 5
    assert b > a
    generator = np.random.default_rng(seed)
    nb_tries = 300

    data_size = lambda d: (np.logspace(np.log10(0.25), np.log10(4), 8, base=10) * d).astype(int)
    sigmas = [1]

    for n_d, d in enumerate(dimensions):
        ms = data_size(d)
        slope = np.ones((d - 1,))

        with open('progress_sd.txt', 'a') as f:
            f.write(f"{datetime.now()} - d = {d}\n")
        emp_slope = slope + np.clip(generator.normal(scale=1, size=(d - 1,)), -1, 1)  # random disturbance
        lj = Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)), -4, 4, theta=emp_slope,
                                            scaling_factor=1.05)
        lj.type = "LJ"

        delta_w = (b - a) / 2
        ses = Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)), -delta_w, delta_w,
                                             theta=emp_slope, scaling_factor=1.05)
        ses.type = "SCC"

        ellipsoids = [lj, ses]

        # fig, ax = plt.subplots()

        for ellipsoid in ellipsoids:
            test_loss_0 = np.zeros((len(ms), len(sigmas), nb_tries))
            test_loss_star = np.zeros((len(ms), len(sigmas), nb_tries))
            test_loss_stoch_dom = np.zeros((len(ms), len(sigmas), nb_tries))
            test_loss_r = np.zeros((len(ms), len(sigmas)))

            for i, m in enumerate(ms):
                with open('progress_sd.txt', 'a') as f:
                    f.write(f"{datetime.now()} - m = {m}\n")
                for j, sigma in enumerate(sigmas):
                    test_x = (b - a) * MDG.uniform_unit_hypercube(generator, d - 1, 10000) + a
                    test_y = np.array([np.dot(test_x[:, k], slope) for k in range(10000)]) + \
                             MDG.normal_disturbance(generator, sigma, 10000, True)
                    test_data = np.vstack([test_x, test_y])
                    MDG.contain_in_ellipsoid(generator, test_data, ellipsoid, slope)
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
                        nodes = np.linspace(0, 1, 40)
                        stoch_dominance = StochasticDominanceCADRO(data, ellipsoid, threshold_type=nodes,
                                                                   solver=cp.MOSEK)
                        stoch_dominance.solve()

                        # fill in the loss arrays
                        test_loss_0[i, j, k] = problem.test_loss(test_data, 'theta_0')
                        test_loss_star[i, j, k] = problem.test_loss(test_data, 'theta')
                        test_loss_stoch_dom[i, j, k] = stoch_dominance.test_loss(test_data, 'theta')
                        if k == 0:
                            test_loss_r[i, j] = problem.test_loss(test_data, 'theta_r')

                    # plot the loss histograms
                    # remove outliers
                    # valid_indices = np.where(test_loss_0[i, j, :] < 10 * np.median(test_loss_0[i, j, :]))[0]
                    # loss_0_plot = test_loss_0[i, j, valid_indices]
                    # loss_stoch_dom_plot = test_loss_stoch_dom[i, j, valid_indices]
                    #
                    # plt.figure()
                    # aux.plot_loss_histograms(plt.gca(), loss_0_plot, loss_stoch_dom_plot, test_loss_r[i, j],
                    #                          bins=20)
                    # plt.tight_layout()
                    # plt.savefig(
                    #     f"loss_hist_d{d}_{ellipsoid.type}_m{m}_sigma{sigma}.png")
                    # plt.close()
                    #
                    # # save the losses to file
                    # with open('results.txt', 'a') as f:
                    #     f.write(f"{datetime.now()} - d = {d}, ellipsoid = {ellipsoid.type}, sigma = {sigma}, m = {m}\n")
                    #     f.write(f"Loss 0: {np.median(test_loss_0[i, j, :])} ("
                    #             f"{np.percentile(test_loss_0[i, j, :], 25)}, "
                    #             f"{np.percentile(test_loss_0[i, j, :], 75)})\n")
                    #     f.write(f"Loss star: {np.median(test_loss_star[i, j, :])} ("
                    #             f"{np.percentile(test_loss_star[i, j, :], 25)}, "
                    #             f"{np.percentile(test_loss_star[i, j, :], 75)})\n")
                    #     f.write(f"Loss stoch dom: {np.median(test_loss_stoch_dom[i, j, :])} ("
                    #             f"{np.percentile(test_loss_stoch_dom[i, j, :], 25)}, "
                    #             f"{np.percentile(test_loss_stoch_dom[i, j, :], 75)})\n")
                    #     f.write(f"Loss r: {test_loss_r[:, j]}\n")
                    #     f.write("---------------------------------------------------------------\n")

            # save the losses to file
            np.save(f"results_d{d}_{ellipsoid.type}_loss_0.npy", test_loss_0)
            np.save(f"results_d{d}_{ellipsoid.type}_loss_star.npy", test_loss_star)
            np.save(f"results_d{d}_{ellipsoid.type}_loss_stoch_dom.npy", test_loss_stoch_dom)
            np.save(f"results_d{d}_{ellipsoid.type}_loss_r.npy", test_loss_r)

            # plot the average loss in function of m for every sigma
    #         ellipsoid_type = ellipsoid.type
    #         colors = ['orange', 'b', 'g', 'black'] if ellipsoid_type == "LJ" else ['r', 'purple', 'brown', 'grey']
    #         for j, sigma in enumerate(sigmas):
    #             aux.plot_loss_m(ax, np.median(test_loss_0[:, j, :], axis=1),
    #                             np.percentile(test_loss_0[:, j, :], 75, axis=1),
    #                             np.percentile(test_loss_0[:, j, :], 25, axis=1),
    #                             np.median(test_loss_star[:, j, :], axis=1),
    #                             np.percentile(test_loss_star[:, j, :], 75, axis=1),
    #                             np.percentile(test_loss_star[:, j, :], 25, axis=1),
    #                             np.median(test_loss_stoch_dom[:, j, :], axis=1),
    #                             np.percentile(test_loss_stoch_dom[:, j, :], 75, axis=1),
    #                             np.percentile(test_loss_stoch_dom[:, j, :], 25, axis=1),
    #                             ms, title=None, scale='linear', label_star=f"CADRO ({ellipsoid_type})",
    #                             label_0=f"SAA ({ellipsoid_type})", label_dro=f"Stoch. Dom. ({ellipsoid_type})",
    #                             colors=colors)
    #
    #         # draw horizontal line for the robust loss
    #         ax.axhline(np.median(test_loss_r[:, 0]), color=colors[3], linestyle='dashed', linewidth=1,
    #                    label=f"Robust loss ({ellipsoid_type})")
    #
    #     # set ylim to the largest value for the dro losses
    #     bottom = min(np.min(np.percentile(test_loss_stoch_dom, 25, axis=2)),
    #                     np.min(np.percentile(test_loss_star, 25, axis=2)))
    #     ax.set_ylim(bottom=bottom)
    #     top = max(np.max(np.percentile(test_loss_stoch_dom, 75, axis=2)),
    #               np.max(np.percentile(test_loss_star, 75, axis=2)))
    #     ax.set_ylim(top=top)
    #
    #     plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    #     plt.grid()
    #     plt.xlabel('m')
    #     plt.ylabel('Loss')
    #     plt.savefig(
    #         f"loss_m_all_d{d}.png")
    #     plt.close()
    #
    # plt.rcParams.update({'font.size': 10})


def experiment2(seed):
    generator = np.random.default_rng(seed)
    d = 5
    m = 50
    sigma = 1
    a, b = 0, 10

    slope = np.ones((d - 1,))
    emp_slope = slope + np.clip(generator.normal(scale=0.5, size=(d - 1,)), -1, 1)  # random disturbance
    ellipsoid = Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)), -4, 4, theta=emp_slope,
                                               scaling_factor=1.05)

    # generate 100 Chebyshev nodes on [0, 1]
    # nodes = 0.5 * np.cos((2 * np.arange(1, 101) - 1) * np.pi / 200) + 0.5

    # generate equally spaced nodes
    nodes = np.linspace(0, 1, 75)
    # add 1 to nodes
    # nodes = np.append(nodes, 1)

    test_x = (b - a) * MDG.uniform_unit_hypercube(generator, d - 1, 10000) + a
    test_y = np.array([np.dot(test_x[:, k], slope) for k in range(10000)]) + \
             MDG.normal_disturbance(generator, sigma, 10000, True)
    test_data = np.vstack([test_x, test_y])
    MDG.contain_in_ellipsoid(generator, test_data, ellipsoid, slope)

    x = (b - a) * MDG.uniform_unit_hypercube(generator, d - 1, m) + a
    y = np.array([np.dot(x[:, i], slope) for i in range(m)]) + \
        MDG.normal_disturbance(generator, sigma, m, True)
    data = np.vstack((x, y))
    MDG.contain_in_ellipsoid(generator, data, ellipsoid, slope)
    problem = StochasticDominanceCADRO(data, ellipsoid, threshold_type=nodes)
    problem.set_theta_r()
    results = problem.solve()

    lambdas, alphas = results["lambda"], results["alpha"]
    nodes = problem.thresholds

    # 1 x 2 subplot
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    # enable grid
    axs[0].grid()
    axs[1].grid()
    axs[0].scatter(nodes, alphas, color='b', marker='.', label='alpha')
    axs[1].scatter(nodes, lambdas, marker='.', color='r', label='lambda')
    axs[1].set_xlabel('Threshold')
    axs[0].legend()
    axs[1].legend()
    plt.show()

    # get the loss of every training data point
    losses = problem.loss_array(test_data, 'theta_0')

    # plot the empirical CDF
    x = np.append(problem.thresholds, problem.eta_bar)
    y = np.append(results['alpha'], 0)
    plt.plot(x, 1 - y, label='Empirical CDF')

    # plot a cdf histogram of the losses
    x = np.sort(losses)
    if problem.eta_bar > x[-1]:
        x = np.append(x, problem.eta_bar)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, label=r'CDF of $J(\xi, \theta_0)$')

    plt.xlabel('Loss')
    plt.ylabel('CDF')
    plt.title(f"d = {d}")
    plt.grid()
    plt.legend()
    plt.savefig("thesis_figures/stoch_dom/empirical_cdf.png")
    plt.show()


def experiment3(seed):
    with open('progress_sd.txt', 'a') as f:
        f.write(f"{datetime.now()} - Starting experiment 3\n")
    generator = np.random.default_rng(seed)
    d = [25, 50]
    # ms is a logspace between 0.25 and 5
    ms = lambda d: (np.logspace(np.log10(0.25), np.log10(4), 8, base=10) * d).astype(int)
    sigma = 1

    for i, dim in enumerate(d):
        a, b = 0, 10
        slope = np.ones((dim - 1,))
        emp_slope = slope + np.clip(generator.normal(scale=0.5, size=(dim - 1,)), -1, 1)
        lj = Ellipsoid.ellipse_from_corners(a * np.ones((dim - 1,)), b * np.ones((dim - 1,)), -3, 3, theta=emp_slope,
                                            scaling_factor=1.05)
        lj.type = "LJ"

        delta_w = (b - a) / 2
        scc = Ellipsoid.ellipse_from_corners(a * np.ones((dim - 1,)), b * np.ones((dim - 1,)), -delta_w, delta_w,
                                             theta=emp_slope, scaling_factor=1.05)
        scc.type = "SCC"

        ellipsoids = [lj, scc]

        nodes = np.linspace(0, 1, 75)
        m = ms(dim)

        test_x = (b - a) * MDG.uniform_unit_hypercube(generator, dim - 1, 10000) + a
        test_y = np.array([np.dot(test_x[:, k], slope) for k in range(10000)]) + \
                 MDG.normal_disturbance(generator, sigma, 10000, True)
        test_data = np.vstack([test_x, test_y])
        nb_tries = 300

        sigmaG = aux.subgaussian_parameter(dim, a, b, -4, 4, emp_slope)

        for j, ellipsoid in enumerate(ellipsoids):

            test_loss_star_sd = np.zeros((len(m), nb_tries))
            cost_star_sd = np.zeros((len(m), nb_tries))

            test_loss_star_classical = np.zeros((len(m), nb_tries))
            cost_star_classical = np.zeros((len(m), nb_tries))

            cost_star_dro = np.zeros((len(m), nb_tries))
            loss_star_dro = np.zeros((len(m), nb_tries))

            for k, m_val in enumerate(m):
                with open('progress_sd.txt', 'a') as f:
                    f.write(f"{datetime.now()} - d = {dim}, ellipsoid = {ellipsoid.type}, m = {m_val}\n")

                for l in range(nb_tries):
                    x = (b - a) * MDG.uniform_unit_hypercube(generator, dim - 1, m_val) + a
                    y = np.array([np.dot(x[:, i], slope) for i in range(m_val)]) + \
                        MDG.normal_disturbance(generator, sigma, m_val, True)
                    data = np.vstack([x, y])
                    MDG.contain_in_ellipsoid(generator, data, ellipsoid, slope)

                    problem_classical = LeastSquaresCadro(data, ellipsoid, solver=cp.MOSEK)
                    problem_classical.solve()

                    test_loss_star_classical[k, l] = problem_classical.test_loss(test_data, 'theta')
                    cost_star_classical[k, l] = problem_classical.objective

                    problem_sd = StochasticDominanceCADRO(data, ellipsoid, threshold_type=nodes)
                    problem_sd.solve()
                    test_loss_star_sd[k, l] = problem_sd.test_loss(test_data, 'theta')
                    cost_star_sd[k, l] = problem_sd.objective

                    if m_val >= 2 * dim:
                        # using less data results in a non-psd covariance matrix
                        problem_dro = MomentDRO(ellipsoid, data, 0.05, sigmaG)
                        problem_dro.solve(check_data=False)

                        cost_star_dro[k, l] = problem_dro.cost
                        loss_star_dro[k, l] = problem_dro.test_loss(test_data)

            np.save(f"results_estimation_d{dim}_{ellipsoid.type}_loss_star_sd.npy", test_loss_star_sd)
            np.save(f"results_estimation_d{dim}_{ellipsoid.type}_cost_star_sd.npy", cost_star_sd)

            np.save(f"results_estimation_d{dim}_{ellipsoid.type}_loss_star_classical.npy", test_loss_star_classical)
            np.save(f"results_estimation_d{dim}_{ellipsoid.type}_cost_star_classical.npy", cost_star_classical)

            np.save(f"results_estimation_d{dim}_{ellipsoid.type}_loss_star_dro.npy", loss_star_dro)
            np.save(f"results_estimation_d{dim}_{ellipsoid.type}_cost_star_dro.npy", cost_star_dro)


def experiment_3b():
    plt.rcParams.update({'font.size': 15})
    # load the results
    d = [25, 50]
    ellipsoids = ["LJ"]  # ["LJ", "SCC"]

    plt.figure()
    md = [0.33, 0.5, 1, 1.5, 2, 3, 4, 5]
    # generate a list of colors from the standard color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ellipsoid in ellipsoids:
        k = 0
        for dim in d:
            loss_star_sd = np.load(f"results_estimation_d{dim}_{ellipsoid}_loss_star_sd.npy")
            cost_star_sd = np.load(f"results_estimation_d{dim}_{ellipsoid}_cost_star_sd.npy")
            loss_star_classical = np.load(f"results_estimation_d{dim}_{ellipsoid}_loss_star_classical.npy")
            cost_star_classical = np.load(f"results_estimation_d{dim}_{ellipsoid}_cost_star_classical.npy")
            loss_star_dro = np.load(f"results_estimation_d{dim}_{ellipsoid}_loss_star_dro.npy")
            cost_star_dro = np.load(f"results_estimation_d{dim}_{ellipsoid}_cost_star_dro.npy")

            # exclude the first two values of m for dro
            loss_star_dro = loss_star_dro[4:, :]
            cost_star_dro = cost_star_dro[4:, :]

            overestimation_sd = (cost_star_sd - loss_star_sd) / loss_star_sd
            overestimation_classical = (cost_star_classical - loss_star_classical) / loss_star_classical
            overestimation_dro = (cost_star_dro - loss_star_dro) / loss_star_dro

            est_sd_p50 = np.percentile(overestimation_sd, 50, axis=1)
            est_sd_p25 = np.percentile(overestimation_sd, 25, axis=1)
            est_sd_p75 = np.percentile(overestimation_sd, 75, axis=1)
            est_sd_max = np.max(overestimation_sd, axis=1)
            est_sd_min = np.min(overestimation_sd, axis=1)

            est_classical_p50 = np.percentile(overestimation_classical, 50, axis=1)
            est_classical_p25 = np.percentile(overestimation_classical, 25, axis=1)
            est_classical_p75 = np.percentile(overestimation_classical, 75, axis=1)
            est_classical_max = np.max(overestimation_classical, axis=1)
            est_classical_min = np.min(overestimation_classical, axis=1)

            est_dro_p50 = np.percentile(overestimation_dro, 50, axis=1)
            est_dro_p25 = np.percentile(overestimation_dro, 25, axis=1)
            est_dro_p75 = np.percentile(overestimation_dro, 75, axis=1)
            est_dro_max = np.max(overestimation_dro, axis=1)
            est_dro_min = np.min(overestimation_dro, axis=1)

            # plot the overestimation
            plt.errorbar(md, est_sd_p50, yerr=[est_sd_p50 - est_sd_p25, est_sd_p75 - est_sd_p50],
                         label=f"Stoch. Dom. (d={dim})", fmt='o-', color=colors[k])
            plt.scatter(md, est_sd_min, color=colors[k], marker='o', facecolors='none')
            plt.scatter(md, est_sd_max, color=colors[k], marker='o', facecolors='none')
            k += 1

            plt.errorbar(md, est_classical_p50,
                         yerr=[est_classical_p50 - est_classical_p25, est_classical_p75 - est_classical_p50],
                         label=f"CADRO (d={dim})", fmt='o-', color=colors[k])
            plt.scatter(md, est_classical_min, color=colors[k], marker='o', facecolors='none')
            plt.scatter(md, est_classical_max, color=colors[k], marker='o', facecolors='none')
            k += 1

            plt.errorbar(md[4:], est_dro_p50, yerr=[est_dro_p50 - est_dro_p25, est_dro_p75 - est_dro_p50],
                         label=f"Mom. DRO* (d={dim})", fmt='o-', alpha=0.5, color=colors[k])
            plt.scatter(md[4:], est_dro_min, marker='o', facecolors='none', alpha=0.5, color=colors[k])
            plt.scatter(md[4:], est_dro_max, marker='o', facecolors='none', alpha=0.5, color=colors[k])
            k += 1

        plt.xlabel("m/d")
        plt.ylabel("Overestimation")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.yscale('log')
        plt.grid()
        plt.title(f"{ellipsoid} ellipsoid")
        plt.ylim([0.5, 50])
        plt.yticks([0.5, 1, 5, 1e1, 50], [0.5, 1, 5, 10, 50])
        plt.gcf().set_size_inches(12, 5)
        plt.tight_layout()
        plt.savefig(f"thesis_figures/stoch_dom/overestimation_{ellipsoid}.pdf")
        plt.show()
        plt.close()


if __name__ == "__main__":
    seed = 0
    # experiment1(seed)
    # experiment2(seed)
    # experiment3(seed)
    experiment_3b()
