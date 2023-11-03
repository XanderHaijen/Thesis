# different ways to compute an ellipsoid containing the given set of points
import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cvxpy as cp
from scipy.optimize import brentq


def ellipse_from_pca_2d(data, scaling_factor: float = 1, plot: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the ellipse that contains the given data using Principal Component Analysis (PCA).
    :param data: (2, n) array of points. The ellipse is defined as
    {x | x^T A x + 2 a^T x + c => 0}.
    :param data: (2, n) array of points
    :param scaling_factor: scaling factor for the ellipse
    :param plot: whether to plot the ellipse along with the data points
    :return: the matrix A, vector a and scalar c
    """
    assert data.shape[0] == 2
    pca = PCA(n_components=2)
    pca.fit(data.T)
    # get the eigenvalues and eigenvectors
    s1, s2 = pca.singular_values_
    # get the principal component
    v1 = pca.components_[0]

    # construct the ellipse
    center = np.mean(data, axis=1)
    cos_angle = np.abs(np.dot(v1, np.array([1, 0]))) / np.linalg.norm(v1)
    sin_angle = np.sqrt(1 - cos_angle ** 2)
    R = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]]) # rotation matrix
    A = scaling_factor * np.diag([- 1 / s1 ** 2, - 1 / s2 ** 2])
    A = R.T @ A @ R
    a = center.T @ A
    c = center.T @ A @ center + 1

    if plot:
        fig = plt.figure(0)
        ax = fig.add_subplot(111, aspect='equal')
        ax.scatter(data[0, :], data[1, :])
        ellipse = Ellipse(xy=(center[0], center[1]), width=2 * np.sqrt(- 1 / A[0, 0]), height=2 * np.sqrt(- 1 / A[1, 1]),
                          edgecolor='r', fc='None')
        ax.add_artist(ellipse)
        plt.xlim(center[0] - 1.25 * np.sqrt(- 1 / A[0, 0]), center[0] + 1.25 * np.sqrt(- 1 / A[0, 0]))
        plt.ylim(center[1] - 1.25 * np.sqrt(- 1 / A[1, 1]), center[1] + 1.25 * np.sqrt(- 1 / A[1, 1]))
        plt.title(f"Ellipse from PCA (scaling factor {scaling_factor})")
        plt.grid()
        plt.show()

    return A, a, c


def minimum_volume_ellipsoid(data, theta0: float, scaling_factor: float = 1, plot: bool = False) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the minimum volume enclosing ellipse of the given data, also known as the Löwner-John ellipsoid.
    The Löwner-John ellipsoid is parametrized by the set
    {x | ||Ax+b||_2 <= 1}.
    :param data: (d, n) array of points
    :param scaling_factor: scaling factor for the ellipse
    :param plot: whether to plot the ellipse along with the data points (only works for 2D data)
    """
    if scaling_factor < 1:
        raise ValueError("Scaling factor must be greater than or equal to 1")
    d = data.shape[0]
    n = data.shape[1]
    A = cp.Variable((d, d))
    b = cp.Variable((d, 1))
    constraints = []
    for i in range(n):
        data_point = data[:, i]
        data_point = np.reshape(data_point, (d, 1))
        constraints = constraints + [cp.norm(A @ data_point + b) <= 1]
    constraints = constraints + [A >> 0]
    objective = cp.Minimize(- cp.log_det(A))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status == cp.INFEASIBLE:
        raise ValueError("Problem is infeasible")
    if problem.status == cp.OPTIMAL_INACCURATE:
        print("WARNING: Problem is optimal but inaccurate")

    A = A.value
    b = b.value
    A_bar = - A.T @ A
    b_bar = - A.T @ b
    c_bar = scaling_factor - b.T @ b

    if plot and d == 2:
        n = 200
        padding = 4
        x_min = np.min(data[0, :])
        x_max = np.max(data[0, :])
        y_max = np.max(data[1, :])
        y_min = np.min(data[1, :])
        plt.figure()
        plot_ellipse_from_matrices(A_bar, b_bar, c_bar, theta0,
                                   x_max, x_min, y_max, y_min, n, padding)
        plt.scatter(data[0, :], data[1, :])
        plt.title(f"Löwner-John ellipsoid (scaled by {scaling_factor})")
        plt.grid()
        plt.show()

    return A_bar, b_bar, c_bar


def smallest_enclosing_sphere(data, scaling_factor: float = 1, plot: bool = False):
    """
    Compute the smallest enclosing sphere of the given data. This is done by using the Löwner-John ellipsoid, but
    fixing the matrix A to be diagonal with equal diagonal entries.
    """
    if scaling_factor < 1:
        raise ValueError("Scaling factor must be greater than or equal to 1")

    d = data.shape[0]
    n = data.shape[1]
    radius = cp.Variable(1)
    center = cp.Variable((d, 1))
    constraints = []
    for i in range(n):
        data_point = data[:, i]
        data_point = np.reshape(data_point, (d, 1))
        constraints = constraints + [cp.norm(data_point - center) <= radius]
    constraints = constraints + [radius >= 0]
    objective = cp.Minimize(radius)
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status == cp.INFEASIBLE:
        raise ValueError("Problem is infeasible")
    if problem.status == cp.OPTIMAL_INACCURATE:
        print("WARNING: Problem is optimal but inaccurate")

    radius = radius.value * scaling_factor

    # ellipsoidal parameters
    A_bar = - np.eye(d)
    b_bar = center.value
    c_bar = radius ** 2 - b_bar.T @ b_bar

    if plot and d == 2:
        # plot the circle
        circle = plt.Circle((center[0].value, center[1].value), radius, color='r', fill=False)
        fig, ax = plt.subplots()
        ax.add_artist(circle)
        ax.set_aspect('equal')
        ax.scatter(data[0, :], data[1, :])
        plt.xlim(center[0].value - 1.25 * radius, center[0].value + 1.25 * radius)
        plt.ylim(center[1].value - 1.25 * radius, center[1].value + 1.25 * radius)
        plt.grid()
        plt.title(f"Smallest enclosing circle (scaling {scaling_factor})")
        plt.show()

    return A_bar, b_bar, c_bar


def plot_ellipse_from_matrices(A, a, c, theta0: float, x_max, x_min, y_max, y_min, n, padding, equal_axis: bool = True):
    """
    Plot the ellipse defined by the matrices A, a, c. The ellipse is defined as
    {x | x^T A x + 2 a^T x + c => 0}. The ellipse is implicitly constructed by iteratively extending the ellipse
    in the left and right direction along the x-axis by solving the ellipse equation for the y-coordinate.
    This function does not invoke the plt.show() or plt.figure() command.
    :param A: (2, 2) matrix
    :param a: (2, 1) vector
    :param c: (1, 1) scalar
    :param theta0: the approximate slope of the data points defining the ellipse
    :param n: The range (x_min, x_max) is divided equidistantly into n points
    :param padding: padding for the y-axis. This is the maximum distance between the data points and the ellipse that
    is plotted.
    :param title: title of the plot
    :param equal_axis: whether to use equal axis
    """

    def ellipse(A, a, c, x, y):
        xy = np.array([[x], [y]])
        return xy.T @ A @ xy + 2 * a.T @ xy + c

    x_range = np.linspace(x_min, x_max, n)

    # first compute the upper half of the ellipse between x_min and x_max
    xs_upper = []
    ys_upper = []
    for x in x_range:
        try:
            y_window = (theta0 * x, y_max + padding)
            y = brentq(lambda y: ellipse(A, a, c, x, y), *y_window, xtol=1e-6)
            xs_upper.append(x)
            ys_upper.append(y)
        except ValueError:
            pass

    # then compute the lower half of the ellipse
    xs_lower = []
    ys_lower = []
    for x in x_range:
        try:
            y_window = (y_min - padding, theta0 * x)
            y = brentq(lambda y: ellipse(A, a, c, x, y), *y_window, xtol=1e-6)
            xs_lower.append(x)
            ys_lower.append(y)
        except ValueError:
            pass

    step = (x_max - x_min) / n
    while True:
        # keep extending the ellipse in the left direction until we reach a ValueError
        x_min = x_min - step
        x_max = x_max + step
        fail = 0
        try:
            # for x_min: both halves
            y_window_up = (theta0 * x_min, y_max + padding)
            y_window_low = (y_min - padding, theta0 * x_min)
            y_up = brentq(lambda y: ellipse(A, a, c, x_min, y), *y_window_up, xtol=1e-6)
            y_low = brentq(lambda y: ellipse(A, a, c, x_min, y), *y_window_low, xtol=1e-6)
            xs_upper.insert(0, x_min)
            ys_upper.insert(0, y_up)
            xs_lower.insert(0, x_min)
            ys_lower.insert(0, y_low)
        except ValueError:
            fail += 1

        try:
            # for x_max: both halves
            y_window_up = (theta0 * x_max, y_max + padding)
            y_window_low = (y_min - padding, theta0 * x_max)
            y_up = brentq(lambda y: ellipse(A, a, c, x_max, y), *y_window_up, xtol=1e-6)
            y_low = brentq(lambda y: ellipse(A, a, c, x_max, y), *y_window_low, xtol=1e-6)
            xs_upper.append(x_max)
            ys_upper.append(y_up)
            xs_lower.append(x_max)
            ys_lower.append(y_low)
        except ValueError:
            fail += 1

        if fail == 2:
            break

    xs_upper.reverse()
    ys_upper.reverse()
    xs = xs_upper + xs_lower
    ys = ys_upper + ys_lower
    xs.append(xs[0])
    ys.append(ys[0])

    if equal_axis:
        plt.axis('equal')
    plt.plot(xs, ys, color='r')


if __name__ == '__main__':
    x = np.linspace(0, 1, 100)
    theta0 = 2
    sigma = 0.5
    y = theta0 * x + sigma * np.random.randn(100)
    data = np.vstack((x, y))
    # A, a, c = ellipse_from_pca_2d(data, scaling_factor=1, plot=True)

    A, a, c = minimum_volume_ellipsoid(data, theta0, plot=True)

    # A, a, c = smallest_enclosing_sphere(data, plot=True)