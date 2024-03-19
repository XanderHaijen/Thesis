from typing import List, Callable

from multiple_dimension_cadro import LeastSquaresCadro
from ellipsoids import Ellipsoid
from utils.data_generator import ScalarDataGenerator
import numpy as np
import matplotlib.pyplot as plt


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
    # return 1 / (1 + 25 * x ** 2)
    # exponentials
    return 5 * np.exp(-x)
    # return np.exp(-x ** 2) + 0.5 * np.exp(-x ** 4)


def ls_matrix(x, y, basis_functions: List[Callable[[float], float]]):
    # d x m matrix
    data = np.zeros((len(basis_functions) + 1, len(x)))
    for i in range(len(basis_functions)):
        phi = basis_functions[i]
        # fill the i-th row
        data[i, :] = phi(x)
    # fill the last row
    data[-1, :] = y

    return data


def create_lj_from_box(lb, ub):
    # create the set of corner points of the hyperractangle
    corners = np.zeros((2 ** len(lb), len(lb)))
    for i in range(2 ** len(lb)):
        for j in range(len(lb)):
            if i & (1 << j):
                corners[i, j] = ub[j]
            else:
                corners[i, j] = lb[j]

    # create the ellipsoid
    ellipsoid = Ellipsoid.lj_ellipsoid(corners.T)
    return ellipsoid


def evaluate(theta, basis_functions, points):
    values = np.zeros(len(points))
    for i, x in enumerate(points):
        values[i] = np.dot(theta, [phi(x) for phi in basis_functions])

    return values


if __name__ == "__main__":
    seed = 0
    m = 50
    sigma = 0.1
    a, b = 0, 1
    f_min, f_max = 0, 5
    assert f_min < f_max

    # training_data
    generator = np.random.default_rng(seed)
    x = np.linspace(a, b, m)
    y = f(x) + sigma * generator.standard_normal(len(x))
    y = np.clip(y, f_min, f_max)

    # test_data
    x_test = np.linspace(a, b, 1000)
    y_test = f(x_test) + sigma * generator.standard_normal(len(x_test))
    y_test = np.clip(y_test, f_min, f_max)

    # basis functions
    # sin(n * pi * x), n=1,..,d-1
    # basis_functions = (lambda x: np.ones_like(x),
    #                    lambda x: np.sin(np.pi * x),
    #                    lambda x: np.sin(2 * np.pi * x),
    #                    lambda x: np.sin(3 * np.pi * x),
    #                    lambda x: np.sin(4 * np.pi * x),
    #                    lambda x: np.sin(5 * np.pi * x),
    #                    lambda x: np.sin(6 * np.pi * x))
    # d = len(basis_functions) + 1
    # phi_min = [-1] * (d - 1)
    # phi_max = [1] * (d - 1)

    # monomial
    basis_functions = (lambda x: np.ones_like(x),
                       lambda x: x,
                       lambda x: x ** 2,
                       lambda x: x ** 3,
                       lambda x: x ** 4,
                       lambda x: x ** 5,
                       lambda x: x ** 6,
                       lambda x: x ** 7,
                       lambda x: x ** 8,
                       lambda x: x ** 9,
                       lambda x: x ** 10)
    d = len(basis_functions) + 1
    phi_min = [0] * (d - 1)
    phi_max = [1] * (d - 1)

    # plot the basis functions
    # points = np.linspace(a, b, 300)
    # for i, phi in enumerate(basis_functions):
    #     plt.figure()
    #     plt.plot(points, phi(points))
    #     plt.title(f"basis function {i + 1}")
    # plt.show()

    # create data matrix
    training_data = ls_matrix(x, y, basis_functions)
    test_data = ls_matrix(x_test, y_test, basis_functions)

    # create the ellipsoid based on the bounding box
    lb = phi_min + [f_min]
    ub = phi_max + [f_max]
    ellipsoid = create_lj_from_box(lb, ub)

    # ellipsoid = Ellipsoid.lj_ellipsoid(training_data)

    # create the problem
    # shuffling the data
    generator.shuffle(training_data.T)
    problem = LeastSquaresCadro(training_data, ellipsoid)
    problem.solve()
    problem.set_theta_r()

    # plot results
    points = np.linspace(a, b, 300)

    theta_star = problem.results["theta"]
    theta_r = problem.results["theta_r"]
    theta_0 = problem.results["theta_0"][0]

    curve_star = evaluate(theta_star, basis_functions, points)
    curve_r = evaluate(theta_r.reshape(-1), basis_functions, points)
    curve_0 = evaluate(theta_0.reshape(-1), basis_functions, points)
    actual_curve = f(points)

    plt.figure()
    plt.scatter(x, y, label="data", marker=".")
    plt.plot(points, curve_star, label=r"$\theta^*$")
    plt.plot(points, curve_r, label=r"$\theta_r$", linestyle="--")
    plt.plot(points, curve_0, label=r"$\theta_0$", linestyle="--")
    plt.plot(points, actual_curve, label="actual", linestyle="--")
    plt.legend()
    plt.show()

    print("theta_0: ", theta_0)
    print("theta_star: ", theta_star)
