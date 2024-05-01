# different ways to compute an ellipsoid containing the given set of points
import numpy as np
from typing import Tuple, Union
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import brentq


class Ellipsoid:
    def __init__(self, A, a, c, shape=None, center=None, kind: str = None):
        n, m = A.shape
        assert n == m
        assert a.shape == (n, 1)
        assert c.shape == (1, 1)
        assert self.is_nsd(A)

        self.A = A
        self.a = a
        self.c = c
        self.circle = (n == 2 and np.array_equal(A, - np.eye(2)))
        self.type = kind

        # geometric parametrization
        self.shape = shape
        self.center = center

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
    def lj_ellipsoid(data: np.ndarray, theta0: float = None, scaling_factor: float = 1, plot: bool = False):
        """
        Compute the Löwner-John ellipsoid of the given data. The ellipse is defined as
        {x | x^T A x + 2 a^T x + c => 0}.
        :param data: (d, n) array of points
        :param theta0: the approximate slope of the data points defining the ellipse
        :param scaling_factor: scaling factor for the ellipse
        :param plot: whether to plot the ellipse along with the data points (only works for 2D data)
        :return: an Ellipsoid object
        """
        if theta0 is not None:
            print("Use of theta0 is no longer needed in Ellipsoid.lj_ellipsoid. It will be ignored.")
        if scaling_factor < 1:
            raise ValueError("Scaling factor must be greater than or equal to 1")
        d = data.shape[0]
        n = data.shape[1]
        A = cp.Variable((d, d), PSD=True)
        b = cp.Variable((d, 1))
        constraints = []
        for i in range(n):
            data_point = data[:, i]
            data_point = np.reshape(data_point, (d, 1))
            constraints = constraints + [cp.norm(A @ data_point + b) <= 1]
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

        ellipsoid = Ellipsoid(A_bar, b_bar, c_bar, shape=-A_bar / scaling_factor ** 2, center=np.linalg.solve(A, -b), kind="LJ")

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

        ellipsoid = Ellipsoid(A_bar, b_bar, c_bar, shape=np.eye(d) / radius ** 2, center=center.value, kind="SES")

        if plot and d == 2:
            ellipsoid.plot()

        return ellipsoid

    @staticmethod
    def from_principal_axes(R: np.ndarray, data: np.ndarray,
                            scaling_factor: Union[float, None] = 1,
                            plot: bool = False,
                            max_length: float = None,
                            solver=cp.MOSEK, verbose=False):
        """
        Compute the smallest possible ellipsoid that contains the given data, given that the principal axes of the
        ellipse are known. The principal axes are given by the matrix R. The ellipse is defined as
        {x | x^T A x + 2 a^T x + c => 0}.
        :param R: (d, d) matrix. The columns of R are the principal axes of the ellipse. The principal axes are assumed
        to be orthonormal. Alternatively, a dict can be given with the keys 'd' and values 'v', where 'v' is the d-th
        principal axis.
        :param data: (d, n) array of points
        :param scaling_factor: scaling factor for the ellipse
        :param plot: whether to plot the ellipse along with the data points (only works for 2D data)
        :param max_length: maximum length of the principal axes. If given, the lengths of the principal axes are
        constrained to be less than or equal to max_length.
        :param solver: the solver to use for the optimization problem (default is MOSEK)
        :param verbose: whether to print the solver output
        :return: an Ellipsoid object
        """

        # the free variables are the lengths of the principal axes if not given, and the center of the ellipse
        d = data.shape[0]
        n = data.shape[1]

        assert R.shape == (d, d)

        lengths = cp.Variable((d, ))
        center = cp.Variable((d, 1))

        # construct A as each column of R multiplied by the corresponding length
        A = R @ cp.diag(cp.reshape(lengths, (d,))) @ R.T
        constraints = [A >> 0]
        constraints += [lengths[i] >= 0 for i in range(d)]
        if max_length is not None:
            constraints += [lengths[i] <= 1/(max_length**2) for i in range(d)]

        for i in range(n):
            data_point = data[:, i]
            data_point = np.reshape(data_point, (d, 1))
            constraints = constraints + [cp.norm(A @ data_point + center) <= 1]
        objective = cp.Minimize(- cp.log_det(A))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver, verbose=verbose)

        if problem.status == cp.INFEASIBLE:
            raise ValueError("Problem is infeasible")
        if problem.status == cp.OPTIMAL_INACCURATE:
            print("WARNING: Problem is optimal but inaccurate")

        A = R @ np.diag(lengths.value / scaling_factor) @ R.T
        center = center.value

        A_bar = - A.T @ A
        b_bar = - A.T @ center
        c_bar = 1 - center.T @ center

        ellipsoid = Ellipsoid(A_bar, b_bar, c_bar, shape=-A_bar, center=np.linalg.solve(A, -center))

        if plot and d == 2:
            ellipsoid.plot()

        print("Lengths")
        print(lengths.value)

        return ellipsoid

    @staticmethod
    def lj_from_corners(corners: np.ndarray, scaling_factor: float = 1):

        d = corners.shape[0]
        m = corners.shape[1]

        if d > 20:
            print("WARNING: The number of dimensions is greater than 20. This may take a long time. Consider"
                  "using the 'ellipsoid_from_corners' function instead.")

        assert m == 2 ** d
        # get the center of the data hypercube
        center = np.mean(corners, axis=1)
        center = np.reshape(center, (d, 1))

        A = cp.Variable((d, d), PSD=True)

        # construct the constraints
        constraints = []
        for i in range(m):
            point = corners[:, i]
            point = np.reshape(point, (d, 1))
            constraints = constraints + [cp.norm(A @ point - center) <= 1]

        # construct the objective
        objective = cp.Minimize(- cp.log_det(A))

        # solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status == cp.INFEASIBLE:
            raise ValueError("Problem is infeasible")
        if problem.status == cp.OPTIMAL_INACCURATE:
            print("WARNING: Problem is optimal but inaccurate")

        A = A.value
        A_bar = - A.T @ A
        b_bar = A.T @ center
        # b_bar = np.reshape(b_bar, (d, 1))
        c_bar = scaling_factor**2 - center.T @ center
        # c_bar = np.reshape(c_bar, (1, 1))

        ellipsoid = Ellipsoid(A_bar, b_bar, c_bar, shape=-A_bar / scaling_factor**2, center=center, kind="Circumcircle")

        return ellipsoid

    @staticmethod
    def ellipse_from_corners(lbx: np.ndarray, ubx: np.ndarray,
                             lbw: float, ubw: float,
                             theta: np.ndarray, scaling_factor: float = 1):
        """
        Analytically compute the ellipsoid that contains the given hyperrectangle. We assume the following problem
        - x is a (d-1)-dimensional vector bounded by lbx and ubx
        - y is a scalar variable that can be written as x^T theta + w, where w is bounded by lbw and ubw
        - theta is a (d-1)-dimensional vector

        The problem is to find the ellipsoid that contains the set of points (x, y) that satisfy the given constraints.

        :param lbx: lower bounds for x
        :param ubx: upper bounds for x
        :param lbw: lower bounds for w
        :param ubw: upper bounds for w
        :param theta: the vector theta (i.e. the slope of the line or hyperplane)

        :return: an Ellipsoid object
        """
        d = len(lbx) + 1
        assert np.shape(ubx) == (d-1, )
        assert np.shape(lbx) == (d-1, )
        assert isinstance(lbw, (int, float))
        assert isinstance(ubw, (int, float))
        assert np.shape(theta) == (d-1, )
        assert scaling_factor >= 1, "Scaling factor must be greater than or equal to 1"

        # construct the inverse matrix A
        Delta_x = np.diag(ubx - lbx) / 2.0
        inv_Delta_x = np.diag(2.0 / (ubx - lbx))
        Delta_w = np.reshape(ubw - lbw, (1, )) / 2.0
        mu_x = np.reshape((ubx + lbx) / 2, (d-1, 1))
        mu_w = np.reshape((ubw + lbw) / 2, (1, ))
        # element-wise multiplication of theta and delta_x
        A12 = np.reshape(theta.T, (1, d-1))
        A = np.bmat([[Delta_x, np.zeros((d-1, 1))],
                     [A12, np.reshape(Delta_w, (1, 1))]])
        inv_A = np.bmat([[inv_Delta_x, np.zeros((d-1, 1))],
                         [-(1 / Delta_w) * A12, np.reshape(1 / Delta_w, (1, 1))]])
        inv_A = np.array(inv_A)  # convert matrix object to numpy array
        b2 = np.dot(theta, mu_x) + mu_w
        b = np.vstack((mu_x, b2))

        shape = inv_A.T @ inv_A / d
        center = b

        A_ellipse = - shape
        a_ellipse = np.reshape(shape @ b, (d, 1))
        c_ellipse = np.reshape(scaling_factor ** 2 - center.T @ shape @ center, (1, 1))

        ellipsoid = Ellipsoid(A_ellipse, a_ellipse, c_ellipse, shape=shape / scaling_factor**2,
                              center=center, kind="LJ")

        return ellipsoid

    @staticmethod
    def sphere_from_parameters(center, radius, scaling_factor=1):
        """
        Compute the sphere with center and radius.
        :param center: center of the sphere
        :param radius: radius of the sphere
        :param scaling_factor: scaling factor for the sphere
        :return: an Ellipsoid object
        """
        d = len(center)

        A = - np.eye(d)
        a = np.reshape(center, (d, 1))
        c = - center.T @ center + d * radius ** 2 * scaling_factor ** 2

        shape = np.eye(d) / (d * radius ** 2 * scaling_factor ** 2)
        center = center

        ellipsoid = Ellipsoid(A, a, c, shape=shape, center=center, kind="SES")

        return ellipsoid

    def normalize(self) -> None:
        """
        Normalize A, a, and c by dividing by c
        """
        self.A = self.A / self.c
        self.a = self.a / self.c
        self.c = 1

    def plot(self, ax=None, **style):
        """Plot an Ellipsoid set in 2D or 3D."""
        if self.dim == 2:
            return self._plot_ellipse2d(ax, **style)
        if self.dim == 3:
            return self._plot_ellipse3d(ax, **style)

    def _plot_ellipse2d(self, ax=None, **style):
        from matplotlib.patches import Ellipse
        ax = plt.gca() if ax is None else ax
        eigval, eigvec = np.linalg.eig(self.shape)
        # sort the eigenvalues and eigenvectors in ascending order
        idx = eigval.argsort()
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        angle = np.arctan2(*np.flip(eigvec[0, :])) / np.pi * 180.
        lenx = 2. / np.sqrt(eigval[0])
        leny = 2. / np.sqrt(eigval[1])
        ellipse_patch = Ellipse(xy=(self.center[0], self.center[1]), width=lenx, height=leny, angle=angle,
                                fill=False, **style)
        ax.add_patch(ellipse_patch)
        ax.autoscale_view()
        return ax

    def _plot_ellipse3d(self, ax=None, **style):
        raise NotImplementedError("3D plotting not implemented yet.")


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
