from typing import List, Callable

from multiple_dimension_cadro import LeastSquaresCadro
from ellipsoids import Ellipsoid
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def f(x):
    # return np.sin(2 * np.pi * x)
    # return np.sin(2 * np.pi * x) + 0.3 * np.sin(4 * np.pi * x)
    # polynomial
    # return 1 - 2 * x + 3 * x ** 2 - 4 * x ** 3 + 5 * x ** 4 - 6 * x ** 5 + 7 * x ** 6
    # return 1 - x**2
    # very oscillatory polynomial
    # return 1 - 2 * x + 3 * x ** 2 - 4 * x ** 3 + 5 * x ** 4 - 6 * x ** 5 + 7 * x ** 6 - \
    # 8 * x ** 7 + 9 * x ** 8 - 10 * x ** 9
    # rational function
    return 1 / (1 + 25 * x ** 2)
    # exponentials
    # return 5 * np.exp(-x)
    # return np.exp(-x ** 2) + 0.5 * np.exp(-x ** 4)


def ls_matrix(x, y, basis_functions: Callable[[np.ndarray, int], float], n: int):
    # d x m matrix
    data = np.zeros((n + 1, len(x)))
    for i in range(n):
        # fill the i-th row
        data[i, :] = basis_functions(x, i)
    # fill the last row
    data[-1, :] = y

    return data


def evaluate(theta: np.ndarray, basis_functions: Callable[[np.ndarray, int], float], points: np.ndarray, n: int):
    values = np.zeros(len(points))
    for i, x in enumerate(points):
        values[i] = np.dot(theta, [basis_functions(x, j) for j in range(n)])

    return values


def fit_curve(x, y, x_test, y_test,
              basis_functions: Callable[[np.ndarray, int], float], n: int,
              phi_min: np.ndarray, phi_max: np.ndarray, f_min: float, f_max: float):
    training_data = ls_matrix(x, y, basis_functions, n)
    ellipsoid = Ellipsoid.ellipse_from_corners(phi_min, phi_max, f_min, f_max, theta=np.zeros((n,)))
    problem = LeastSquaresCadro(training_data, ellipsoid)
    problem.solve()
    problem.set_theta_r()

    test_data = ls_matrix(x_test, y_test, basis_functions, n)
    test_loss = problem.test_loss(test_data)
    test_loss_0 = problem.test_loss(test_data, type='theta_0')
    test_loss_r = problem.test_loss(test_data, type='theta_r')

    results = problem.results
    results.update({"test_loss": test_loss})
    results.update({"test_loss_0": test_loss_0})
    results.update({"test_loss_r": test_loss_r})

    return results


def plot_basis_functions(basis_functions: Callable[[np.ndarray, int], float], n: int, a: float, b: float):
    points = np.linspace(a, b, 300)
    for i in range(n):
        plt.plot(points, basis_functions(points, i), label=f"i = {i + 1}")
    plt.legend()
    plt.show()


def experiment1(seed, basis_functions, m, sigma, a, b, f, f_min, f_max, phi_min, phi_max, title, verbose=True,
                nodes="equidistant"):
    """
    For the given basis functions, plot the fitted curve for different values of n.
    """
    generator = np.random.default_rng(seed)
    if nodes == "equidistant":
        x = np.linspace(a, b, m)
        x_test = np.linspace(a, b, 3000)
    elif nodes == "random":
        x = generator.uniform(a, b, m)
        x_test = generator.uniform(a, b, 3000)
    elif nodes == "chebyshev":
        # chebyshev nodes on [0, 1]
        x = np.cos((2 * np.arange(1, m + 1) - 1) * np.pi / (2 * m))
        x = (x + 1) / 2
        x_test = np.cos((2 * np.arange(1, 3000 + 1) - 1) * np.pi / (2 * 3000))
        x_test = (x_test + 1) / 2
    else:
        raise ValueError("nodes should be either equidistant, random or chebyshev")

    generator.shuffle(x)
    y = f(x) + sigma * generator.standard_normal(len(x))
    y_test = f(x_test) + sigma * generator.standard_normal(len(x_test))
    y = np.clip(y, f_min, f_max)
    y_test = np.clip(y_test, f_min, f_max)

    n_list = list(range(1, 10))
    plot_points = np.linspace(a, b, 300)
    function_values_star = np.zeros((len(n_list), len(plot_points)))
    function_values_r = np.zeros((len(n_list), len(plot_points)))
    function_values_0 = np.zeros((len(n_list), len(plot_points)))

    for i, n in enumerate(n_list):
        phi_min_array = np.full((n,), phi_min)
        phi_max_array = np.full((n,), phi_max)
        if verbose:
            print("n = ", n)
        results = fit_curve(x, y, x_test, y_test, basis_functions, n, phi_min_array, phi_max_array, f_min, f_max)

        if verbose:
            print("test_loss_0: ", results["test_loss_0"])
            print("test loss: ", results["test_loss"])
            print()

        theta_star = np.reshape(results["theta"], (n,))
        theta_r = np.reshape(results["theta_r"], (n,))
        theta_0 = np.reshape(results["theta_0"][0], (n,))

        function_values_star[i, :] = evaluate(theta_star, basis_functions, plot_points, n)
        function_values_r[i, :] = evaluate(theta_r, basis_functions, plot_points, n)
        function_values_0[i, :] = evaluate(theta_0, basis_functions, plot_points, n)

    actual_curve = f(plot_points)

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    for i, n in enumerate(n_list):
        ax[i // 3, i % 3].plot(plot_points, actual_curve, label="actual", color='grey')
        # add points
        ax[i // 3, i % 3].scatter(x, y, color='grey', marker='.')
        ax[i // 3, i % 3].plot(plot_points, function_values_r[i, :], label=r"$\theta_r$", color='red')
        ax[i // 3, i % 3].plot(plot_points, function_values_0[i, :], label=r"$\theta_0$", color='orange')
        ax[i // 3, i % 3].plot(plot_points, function_values_star[i, :], label=r"$\theta^\star$",
                               linestyle='-.', color='black')

        ax[i // 3, i % 3].legend()
        ax[i // 3, i % 3].set_title(f"n = {n}")

    plt.savefig(f"experiment1_{title}.png")
    plt.show()


def experiment2(seed, basis_functions, m, sigma, a, b, f, f_min, f_max, phi_min, phi_max,
                title, results_df, nb_tries=100, verbose=False, nodes="equidistant"):
    """
    For the given basis functions, run the algorithm with increasing degree of the basis functions
    until the robust solution is found. Then return the degree, and the results.
    """
    generator = np.random.default_rng(seed)
    if nodes == "equidistant":
        x_ord = np.linspace(a, b, m)
        x_test = np.linspace(a, b, 3000)
    elif nodes == "random":
        x_ord = generator.uniform(a, b, m)
        x_test = generator.uniform(a, b, 3000)
    elif nodes == "chebyshev":
        # chebyshev nodes on [0, 1]
        x_ord = np.cos((2 * np.arange(1, m + 1) - 1) * np.pi / (2 * m))
        x_ord = (x_ord + 1) / 2
        x_test = np.cos((2 * np.arange(1, 3000 + 1) - 1) * np.pi / (2 * 3000))
        x_test = (x_test + 1) / 2
    else:
        raise ValueError("nodes should be either equidistant, random or chebyshev")

    collapse_dims = np.zeros(nb_tries)
    loss_0_array = np.zeros(nb_tries)
    loss_r_array = np.zeros(nb_tries)

    for j in range(nb_tries):
        x = np.copy(x_ord)
        generator.shuffle(x)
        y = f(x) + sigma * generator.standard_normal(len(x))
        y_test = f(x_test) + sigma * generator.standard_normal(len(x_test))
        y = np.clip(y, f_min, f_max)
        y_test = np.clip(y_test, f_min, f_max)

        n = 1

        while True:
            phi_min_array = np.full((n,), phi_min)
            phi_max_array = np.full((n,), phi_max)
            if verbose:
                print("n = ", n)
            results = fit_curve(x, y, x_test, y_test, basis_functions, n, phi_min_array, phi_max_array, f_min, f_max)

            if verbose:
                print("test_loss_0: ", results["test_loss_0"])
                print("test loss: ", results["test_loss"])
                print()

            if np.abs(results["test_loss_r"] - results["test_loss"]) < 1e-5:
                break
            elif n > 50:
                n = np.inf
                break
            n += 1

        collapse_dims[j] = n
        loss_0_array[j] = results["test_loss_0"]
        loss_r_array[j] = results["test_loss_r"]

    # collapse dims
    n_mean = np.mean(collapse_dims[collapse_dims != np.inf])
    n_std = np.std(collapse_dims[collapse_dims != np.inf])
    n_nb_inf = np.sum(collapse_dims == np.inf)
    valid_indices = np.where(collapse_dims != np.inf)
    # losses and theta differences
    loss_0_data = {"median": np.median(loss_0_array[valid_indices]),
                   "p75": np.percentile(loss_0_array[valid_indices], 75),
                   "p25": np.percentile(loss_0_array[valid_indices], 25)}
    loss_r_data = {"median": np.median(loss_r_array[valid_indices]),
                   "p75": np.percentile(loss_r_array[valid_indices], 75),
                   "p25": np.percentile(loss_r_array[valid_indices], 25)}
    results_df.loc[len(results_df)] = {"function": title, "collapse_dim_mean": n_mean, "collapse_dim_std": n_std,
                                       "collapse_dim_inf": n_nb_inf,
                                       "test_loss_0_p25": loss_0_data["p25"],
                                       "test_loss_0_median": loss_0_data["median"],
                                       "test_loss_0_p75": loss_0_data["p75"],
                                       "test_loss_r_p25": loss_r_data["p25"],
                                       "test_loss_r_median": loss_r_data["median"],
                                       "test_loss_r_p75": loss_r_data["p75"]}
    return results_df


def experiment3():
    """
    Plot the six test functions and corresponding basis functions.
    """
    plt.rcParams.update({'font.size': 15})

    basis_1 = lambda x, i: np.sin(i * np.pi * x) if i > 0 else np.ones_like(x)
    basis_2 = lambda x, i: x ** i
    basis_3 = lambda x, i: np.exp(-i * x)

    basis = [basis_1, basis_2, basis_3]
    basis_titles = ["trigonometric", "monomial", "exponential"]

    f1 = lambda x: np.sin(2 * np.pi * x) + 0.3 * np.sin(4 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)
    f2 = lambda x: np.maximum(0, 1 - np.abs(2 * x - 1))
    f3 = lambda x: 500 * (x - 0.25) * (x - 0.75) ** 2 * (x - 0.05) * (x - 1)
    f4 = lambda x: np.exp(- x ** 2)
    f5 = lambda x: 1 / (1 + 25 * x ** 2)
    f6 = lambda x: 5 * np.exp(-x) + np.exp(-2 * x) + 0.5 * np.exp(-5 * x)

    fs = [f1, f2, f3, f4, f5, f6]
    titles = ["trigonometric", "hat", "polynomial", "gaussian", "rational", "exponentials"]

    x = np.linspace(0, 1, 500)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    for i, f in enumerate(fs):
        ax[i // 3, i % 3].plot(x, f(x))
        ax[i // 3, i % 3].set_title(titles[i])
    plt.savefig("experiment3_functions.png")
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i, b in enumerate(basis):
        for j in range(5):
            ax[i].plot(x, b(x, j), label=r"$\phi_{}$".format(j))
            ax[i].set_title(basis_titles[i])
        ax[i].legend()
    plt.savefig("experiment3_basis_functions.png")
    plt.show()


def experiment4(seed):
    """
    For a two-dimensional basis, plot the points in the feature space, together with the bounding box.
    """
    basis_functions = lambda x, i: np.sin(i * np.pi * x) if i > 0 else np.ones_like(x)
    # basis_functions = lambda x, i: x ** (i+1)
    m = 1000
    a, b = -1, 1
    sigma = 0.25
    x_long = np.linspace(a, b, 1000)
    f = lambda x: np.sin(2 * np.pi * x) + 0.3 * np.sin(4 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)
    f_min, f_max = np.min(f(x_long)) - 2 * sigma, np.max(f(x_long)) + 2 * sigma
    phi_min, phi_max = -1, 1

    x = np.linspace(a, b, m)
    generator = np.random.default_rng(seed)
    y = f(x) + sigma * generator.standard_normal(len(x))
    y = np.clip(y, f_min, f_max)

    # create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot the points
    A = ls_matrix(x, y, basis_functions, 3)
    ax.scatter(A[1, :], A[2, :], A[3, :], color='blue', marker='.')
    # plot the bounding box
    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                ax.plot([phi_min, phi_min, phi_max, phi_max, phi_min],
                           [phi_min, phi_max, phi_max, phi_min, phi_min],
                           [f_min, f_min, f_min, f_min, f_min], color='red')
                ax.plot([phi_min, phi_min, phi_max, phi_max, phi_min],
                           [phi_min, phi_max, phi_max, phi_min, phi_min],
                           [f_max, f_max, f_max, f_max, f_max], color='red')
                ax.plot([phi_min, phi_min, phi_min, phi_min, phi_min],
                           [phi_min, phi_max, phi_max, phi_min, phi_min],
                           [f_min, f_min, f_max, f_max, f_min], color='red')
                ax.plot([phi_max, phi_max, phi_max, phi_max, phi_max],
                           [phi_min, phi_max, phi_max, phi_min, phi_min],
                           [f_min, f_min, f_max, f_max, f_min], color='red')
                ax.plot([phi_min, phi_min, phi_max, phi_max, phi_min],
                           [phi_min, phi_min, phi_min, phi_min, phi_min],
                           [f_min, f_max, f_max, f_min, f_min], color='red')
                ax.plot([phi_min, phi_min, phi_max, phi_max, phi_min],
                           [phi_max, phi_max, phi_max, phi_max, phi_max],
                           [f_min, f_max, f_max, f_min, f_min], color='red')

    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")
    ax.set_zlabel(r"$f$")
    plt.savefig("thesis_figures/curve_fitting/experiment4.png")
    plt.show()


if __name__ == "__main__":
    # seed = 0
    # m = 40
    # a, b = 0, 1
    # nb_tries = 100
    # plt.rcParams.update({'font.size': 15})
    # interpolation = "chebyshev"
    # # bookkeeping for experiment 2
    # results = pd.DataFrame(columns=["function", "collapse_dim_mean", "collapse_dim_std", "collapse_dim_inf",
    #                                 "test_loss_0_p25", "test_loss_0_median", "test_loss_0_p75",
    #                                 "test_loss_r_p25", "test_loss_r_median", "test_loss_r_p75"])
    #
    # n = 4  # number of basis functions
    # d = n + 1
    # # basis functions
    # # sin(n * pi * x), n=1,..,d-1
    # basis_functions = lambda x, i: np.sin(i * np.pi * x) if i > 0 else np.ones_like(x)
    # # trig function
    # f = lambda x: np.sin(2 * np.pi * x) + 0.3 * np.sin(4 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)
    # f_min, f_max = -1.5, 1.5
    # phi_min, phi_max = -1, 1
    # sigma = 0.25
    #
    # experiment1(seed, basis_functions, m, sigma, a, b, f, f_min, f_max, phi_min, phi_max,
    #             "trigonometric", nodes=interpolation)
    #
    # results = experiment2(seed, basis_functions, m, sigma, a, b, f, f_min, f_max, phi_min, phi_max,
    #                       "trigonometric", results, nb_tries, nodes=interpolation)
    #
    # # hat function
    # f = lambda x: np.maximum(0, 1 - np.abs(2 * x - 1))
    # f_min, f_max = -0.5, 1.5
    #
    # results = experiment2(seed, basis_functions, m, sigma, a, b, f, f_min, f_max, phi_min, phi_max, "hat",
    #                       results, nb_tries, nodes=interpolation)
    #
    # # monomial basis functions
    # basis_functions = lambda x, i: x ** i
    # f = lambda x: 500 * (x - 0.25) * (x - 0.75) ** 2 * (x - 0.05) * (x - 1)
    # f_min, f_max = -3.5, 1.5
    # phi_min, phi_max = 0, 1
    # sigma = 1
    # experiment1(seed, basis_functions, m, sigma, a, b, f, f_min, f_max, phi_min, phi_max,
    #             "polynomial", nodes=interpolation)
    #
    # results = experiment2(seed, basis_functions, m, sigma, a, b, f, f_min, f_max, phi_min, phi_max, "polynomial",
    #                       results, nb_tries, nodes=interpolation)
    #
    # f = lambda x: np.exp(- x ** 2)
    # f_min, f_max = 0, 1.5
    #
    # results = experiment2(seed, basis_functions, m, sigma, a, b, f, f_min, f_max, phi_min, phi_max, "gaussian",
    #                       results, nb_tries, nodes=interpolation)
    #
    # # negative exponential basis functions
    # basis_functions = lambda x, i: np.exp(-i * x)
    # f = lambda x: 1 / (1 + 25 * x ** 2)
    # f_min, f_max = -0.5, 1.5
    # phi_min, phi_max = 0, 1
    # sigma = 0.1
    # experiment1(seed, basis_functions, m, sigma, a, b, f, f_min, f_max, phi_min, phi_max,
    #             "exponential", nodes=interpolation)
    #
    # results = experiment2(seed, basis_functions, m, sigma, a, b, f, f_min, f_max, phi_min, phi_max, "rational",
    #                       results, nb_tries, nodes=interpolation)
    #
    # # exponentials
    # f = lambda x: 5 * np.exp(-x) + np.exp(-2 * x) + 0.5 * np.exp(-5 * x)
    # f_min, f_max = 1.5, 7
    # phi_min, phi_max = 0, 1
    # sigma = 0.1
    #
    # results = experiment2(seed, basis_functions, m, sigma, a, b, f, f_min, f_max, phi_min, phi_max, "exponentials",
    #                       results, nb_tries, nodes=interpolation)
    #
    # # round all numbers to two decimal places
    # print(results)
    # results.to_latex("results.txt", float_format="%.2f")
    #
    # plt.rcParams.update({'font.size': 10})
    #
    # # experiment3()

    experiment4(0)
