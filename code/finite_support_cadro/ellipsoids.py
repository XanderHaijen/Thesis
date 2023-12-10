# different ways to compute an ellipsoid containing the given set of points
import numpy as np
from typing import Tuple, Union
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import brentq


class Ellipsoid:
    def __init__(self, A, a, c, shape=None, center=None):
        n, m = A.shape
        assert n == m
        assert a.shape == (n, 1)
        assert c.shape == (1, 1)
        assert self.is_nsd(A)

        self.A = A
        self.a = a
        self.c = c
        self.circle = (n == 2 and np.array_equal(A, - np.eye(2)))

        # shape and center for plotting (optional)
        if self.dim == 2 or self.dim == 3:
            self.shape = shape
            self.center = center
        else:
            self.shape = None
            self.center = None

    @staticmethod
    def is_nsd(A):
        """
        Check whether the given matrix is negative semidefinite.
        :param A: (n, n) matrix
        :return: True if A is negative semidefinite, False otherwise
        """
        return np.all(np.linalg.eigvals(A) <= 0)

    def __contains__(self, x):
        return x.T @ self.A @ x + 2 * self.a.T @ x + self.c >= 0

    def contains(self, x):
        return self.__contains__(x)


    @property
    def dim(self):
        return self.A.shape[0]

    @staticmethod
    def lj_ellipsoid(data: np.ndarray, theta0: float, scaling_factor: float = 1, plot: bool = False):
        """
        Compute the Löwner-John ellipsoid of the given data. The ellipse is defined as
        {x | x^T A x + 2 a^T x + c => 0}.
        :param data: (d, n) array of points
        :param theta0: the approximate slope of the data points defining the ellipse
        :param scaling_factor: scaling factor for the ellipse
        :param plot: whether to plot the ellipse along with the data points (only works for 2D data)
        :return: an Ellipsoid object
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
        c_bar = scaling_factor ** 2 - b.T @ b

        ellipsoid = Ellipsoid(A_bar, b_bar, c_bar, shape=-A_bar, center=np.linalg.solve(A, -b))

        if plot and d == 2:
            ellipsoid.plot()

        return ellipsoid

    @staticmethod
    def smallest_enclosing_sphere(data: np.ndarray, scaling_factor: float = 1, plot: bool = False):
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

        ellipsoid = Ellipsoid(A_bar, b_bar, c_bar, shape = np.eye(d) / radius**2, center = center.value)

        if plot and d == 2:
            ellipsoid.plot()

        return ellipsoid

    @staticmethod
    def from_principal_axes(R: np.ndarray, data: np.ndarray, theta0: float, scaling_factor: Union[float, None] = 1,
                            plot: bool = False, lengths=None):
        """
        Compute the smallest possible ellipsoid that contains the given data, given that the principal axes of the
        ellipse are known. The principal axes are given by the matrix R. The ellipse is defined as
        {x | x^T A x + 2 a^T x + c => 0}.
        :param R: (d, d) matrix. The columns of R are the principal axes of the ellipse.
        :param data: (d, n) array of points
        :param scaling_factor: scaling factor for the ellipse
        :param theta0: the approximate slope of the data points defining the ellipse
        :param plot: whether to plot the ellipse along with the data points (only works for 2D data)
        :param lengths: the lengths of the principal axes. If None, the lengths are optimization variables. If lengths
        are given, the lengths are scaled by a factor of scaling_factor, which is the same for all axes.
        :return: an Ellipsoid object
        """
        if scaling_factor is not None and scaling_factor < 1:
            raise ValueError("Scaling factor must be greater than or equal to 1")
        elif scaling_factor is None:
            scaling_factor = cp.Variable(1)
        elif lengths is not None and scaling_factor is not None:
            raise ValueError("Either lengths or scaling_factor must be None, otherwise the problem is overconstrained.")
        elif lengths is None and scaling_factor is None:
            raise ValueError(
                "Either lengths or scaling_factor must be given, otherwise the problem is non-convex.")

        # the free variables are the lengths of the principal axes if not given, and the center of the ellipse
        d = data.shape[0]
        n = data.shape[1]
        if lengths is None:
            lengths = scaling_factor * cp.Variable((d, 1))
        else:
            lengths = scaling_factor * cp.reshape(lengths, (d, 1))
        center = cp.Variable((d, 1))
        # construct A as each column of R multiplied by the corresponding length
        A = cp.diag(lengths) @ R.T
        constraints = [A >> 0]
        for i in range(n):
            data_point = data[:, i]
            data_point = np.reshape(data_point, (d, 1))
            constraints = constraints + [cp.norm(A @ data_point + center) <= 1]
        objective = cp.Minimize(- cp.log_det(A))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        if problem.status == cp.INFEASIBLE:
            raise ValueError("Problem is infeasible")
        if problem.status == cp.OPTIMAL_INACCURATE:
            print("WARNING: Problem is optimal but inaccurate")

        A = A.value
        center = center.value

        A_bar = - A.T @ A
        b_bar = - A.T @ center
        c_bar = 1 - center.T @ center


        ellipsoid = Ellipsoid(A_bar, b_bar, c_bar, shape=-A_bar, center=np.linalg.solve(A, -center))

        if plot and d == 2:
            ellipsoid.plot()

        return ellipsoid

    # def plot(self, theta_0:float = 0, ax=None, color='r'):
    #     assert self.A.shape == (2, 2)
    #     if ax is None:
    #         ax = plt.gca()
    #     if self.circle:
    #         center = self.a
    #         radius = np.sqrt(self.c + self.a.T @ self.a)
    #         circle = plt.Circle((center[0].value, center[1].value), radius, color=color, fill=False)
    #         ax.add_patch(circle)
    #         ax.set_aspect('equal')
    #         plt.xlim(center[0].value - 1.25 * radius, center[0].value + 1.25 * radius)
    #         plt.ylim(center[1].value - 1.25 * radius, center[1].value + 1.25 * radius)
    #     else:
    #         self._plot_ellipse_from_matrices(self.A, self.a, self.c, theta0=1 / 4,
    #                                    x_max=10, x_min=-10,
    #                                    y_max=10, y_min=-10,
    #                                    n=200, padding=4, color=color)

    def plot(self, ax=None, **style):
        """Plot an Ellipsoid set in 2D or 3D."""
        if self.dim == 2:
            return self._plot_ellipse2d(ax, **style)
        if self.dim == 3:
            return self._plot_ellipse3d(ax, **style)

    def _plot_ellipse2d(self, ax=None, **style):
        from matplotlib.patches import Ellipse
        ax = plt.gca() if ax is None else ax
        eigval, eigvec = np.linalg.eigh(self.shape)
        angle = np.arctan2(*np.flip(eigvec[0, :])) / np.pi * 180.
        lenx = 2. / np.sqrt(eigval[0])
        leny = 2. / np.sqrt(eigval[1])
        ellipse_patch = Ellipse(xy=(self.center[0], self.center[1]), width=lenx, height=leny, angle=angle,
                                fill=False, **style)
        ax.add_patch(ellipse_patch)
        ax.autoscale_view()
        return ax

    def _plot_ellipse3d(self, ax = None, **style):
        raise NotImplementedError("3D plotting not implemented yet.")

    @staticmethod
    def _plot_ellipse_from_matrices(A, a, c, theta0: float, x_max, x_min, y_max, y_min,
                                    n, padding, equal_axis: bool = True, color='r'):
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
        :param equal_axis: whether to use equal axis
        """
        # TODO replace this with a more efficient algorithm (rcracers.utils.geometry.plot_ellipsoid)
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
                y_window_up = (theta0 * x_max, y_max + padding**2)
                y_window_low = (y_min - padding**2, theta0 * x_max)
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



        if equal_axis:
            plt.axis('equal')
        plt.plot(xs_lower, ys_lower, color=color)
        plt.plot(xs_upper, ys_upper, color=color)
        # connect the two halves
        plt.plot([xs_lower[-1], xs_upper[-1]], [ys_lower[-1], ys_upper[-1]], color='r', linestyle='--')
        plt.plot([xs_lower[0], xs_upper[0]], [ys_lower[0], ys_upper[0]], color='r', linestyle='--')


class EllipsoidTests:

    def __init__(self, theta0, sigma, angle, lengths=None):
        self.theta0 = theta0
        self.sigma = sigma
        x = np.linspace(-1, 1, 20)
        y = theta0 * x + sigma * np.random.randn(20)
        self.data = np.vstack((x, y))
        self.angle = angle

    def test_lj(self):
        plt.figure()
        Ellipsoid.lj_ellipsoid(self.data, self.theta0, plot=True)
        plt.scatter(self.data[0, :], self.data[1, :])
        plt.axis('equal')
        plt.show()

    def test_smallest_enclosing_sphere(self):
        plt.figure()
        Ellipsoid.smallest_enclosing_sphere(self.data, plot=True)
        plt.scatter(self.data[0, :], self.data[1, :])
        plt.axis('equal')
        plt.show()

    def test_from_principal_axes(self):
        plt.figure()
        R = np.array([[np.cos(self.angle), -np.sin(self.angle)], [np.sin(self.angle), np.cos(self.angle)]])
        Ellipsoid.from_principal_axes(R, self.data, self.theta0, plot=True, lengths=None)
        plt.scatter(self.data[0, :], self.data[1, :])
        plt.axis('equal')
        plt.show()

    def test_from_principal_axes_with_lengths(self, lengths=None):
        plt.figure()
        R = np.array([[np.cos(self.angle), -np.sin(self.angle)], [np.sin(self.angle), np.cos(self.angle)]])
        lengths = np.array([[3], [2]]) if lengths is None else lengths
        Ellipsoid.from_principal_axes(R, self.data, self.theta0, plot=True, lengths=lengths, scaling_factor=None)
        plt.scatter(self.data[0, :], self.data[1, :])
        plt.axis('equal')
        plt.show()

    def run(self):
        self.test_lj()
        self.test_smallest_enclosing_sphere()
        self.test_from_principal_axes()
        self.test_from_principal_axes_with_lengths()


if __name__ == '__main__':
    theta_0 = 3
    sigma = 0.5
    angle = np.pi / 4
    tests = EllipsoidTests(theta_0, sigma, angle)
    tests.run()