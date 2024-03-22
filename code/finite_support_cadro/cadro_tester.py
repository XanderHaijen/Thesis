import numpy as np
from ellipsoids import Ellipsoid
from one_dimension_cadro import CADRO1DLinearRegression
from multiple_dimension_cadro import LeastSquaresCadro
import matplotlib.pyplot as plt
from utils.data_generator import MultivariateDataGenerator as MDG
from time import time
from matplotlib.patches import Rectangle


class ContinuousCADROTester:
    def __init__(self, data: np.ndarray, ellipse: Ellipsoid):
        self.data = data
        self.ellipsoid = ellipse


class CADRO1DLinearRegressionTester(ContinuousCADROTester):
    def __init__(self, data: np.ndarray, ellipse: Ellipsoid, theta0: np.ndarray = None,
                 theta: float = None, confidence_level: float = 0.05, nb_theta0: int = 1):
        super().__init__(data, ellipse)
        self.problem = CADRO1DLinearRegression(data, ellipse)
        self.theta0 = theta0
        self.theta = theta
        self.confidence_level = confidence_level

    def run(self):
        self.problem.solve(theta0=self.theta0, theta=self.theta, confidence_level=self.confidence_level,
                           nb_theta_0=nb_theta0)

        return self.problem.print_results(include_robust=True)


class LeastSquaresTester(ContinuousCADROTester):
    def __init__(self, data: np.ndarray, ellipse: Ellipsoid, theta0: list = None,
                 theta: np.ndarray = None, confidence_level: float = 0.05, nb_theta0: int = 1):
        super().__init__(data, ellipse)
        self.problem = LeastSquaresCadro(data, ellipse)
        self.theta0 = theta0
        self.theta = theta
        self.confidence_level = confidence_level
        self.nb_theta0 = nb_theta0

    def run(self):
        self.problem.solve(theta0=self.theta0, theta=self.theta, confidence_level=self.confidence_level,
                           nb_theta_0=self.nb_theta0)
        self.problem.set_theta_r()
        self.theta = self.problem.theta
        self.problem.print_results(include_robust=True)
        return self.problem.results


def ellipse_from_corners(corners_x: np.ndarray, theta: np.ndarray,
                         ub: float, lb: float, scaling_factor: int = 1,
                         return_corners: bool = False, plot: bool = False):
    """
    Create the d-dimensional circumcircle based on the x-corners and the data hyperplane.
    :param corners_x: the corners of the data hypercube
    :param theta: the data hyperplane slope
    :param ub: the upper bound of for the data deviation
    :param lb: the lower bound for the data deviation
    :param scaling_factor: the scaling factor for the ellipse
    :param return_corners: whether to return the corners of the bounding box
    :param plot: whether to plot the corners of the bounding box (only for d = 2 or d = 3)
    """
    d = corners_x.shape[0] + 1
    m = corners_x.shape[1]
    # for each corner, get the hyperplane value
    corners_y = np.array([np.dot(corners_x[:, i], theta) for i in range(corners_x.shape[1])])
    corners_y_plus = corners_y + ub
    corners_y_min = corners_y - lb
    corners = np.zeros((d, 2 * m))
    corners[:d-1, :m] = corners_x
    corners[d-1, :m] = corners_y_plus
    corners[:d-1, m:] = corners_x
    corners[d-1, m:] = corners_y_min


    if plot and d == 2:
        plt.scatter(corners[:d-1, :m], corners[d-1, :m], label="upper bound")
        plt.scatter(corners[:d-1, m:], corners[d-1, m:], label="lower bound")
        plt.legend()
        plt.show()
    elif plot and d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(corners[0, :m], corners[1, :m], corners[2, :m], label="upper bound")
        ax.scatter(corners[0, m:], corners[1, m:], corners[2, m:], label="lower bound")
        plt.legend()
        plt.show()

    # select the 2d^2 corners that are the furthest from the center
    center = np.max(corners, axis=1) / 2 + np.min(corners, axis=1) / 2
    distances = np.linalg.norm(corners - center[:, None], axis=0)
    n = min(2**d, d**2)
    corners = corners[:, np.argsort(distances)[-n:]]
    ellipsoid = Ellipsoid.lj_ellipsoid(corners, scaling_factor=scaling_factor)

    # assert that the ellipsoid contains the corners
    assert np.all(ellipsoid.contains(corners))

    if return_corners:
        return ellipsoid, corners
    else:
        return ellipsoid

def main(d, generator):
    n = 50
    sigma = 1
    a = 0
    b = 10
    direction = np.ones(d, )
    # direction /= np.linalg.norm(direction)
    # sample uniformly on the unit square
    data = (b - a) * generator.uniform(size=(d, n)) + a
    # values for y are <x, direction> + noise
    y = (np.array([np.dot(data[:, i], direction) for i in range(n)]) +
         np.clip(generator.normal(scale=sigma, size=(1, n)), -3*sigma, 3 * sigma))
    data = np.vstack((data, y))

    # create an array with all the corners of the d - dimensional hypercube
    M = min(2**d, 1e6)  # maximum number of corners
    corners_x = np.zeros((int(M), d))
    k = 0
    for i in range(2 ** d):
        new_corner = np.zeros((d, ))
        for j in range(d):
            if i & (1 << j):
                new_corner[j] = b
            else:
                new_corner[j] = a
        # add the corner with probability 1/2^d
        if M < 2**d and generator.uniform() <= M / 2**d:
            corners_x[k, :] = new_corner
            k += 1
        elif M >= 2**d:
            corners_x[k, :] = new_corner
            k += 1
    corners_x = corners_x[:k, :]
    #
    ellipsoid, corners = ellipse_from_corners(corners_x.T, direction, ub=3 * sigma, lb=5*sigma,
                                              return_corners=True)

    ellipsoid2 = Ellipsoid.ellipse_from_corners(a * np.ones((d, )), b * np.ones((d, )),
                                                -5 * sigma, 3 * sigma, direction,
                                                scaling_factor=1.01)

    ellipsoid.normalize()
    ellipsoid2.normalize()

    # calculate the condition number of the ellipsoid's A matrix
    print(f"Condition number of the ellipsoid: {np.linalg.cond(ellipsoid.A)}")
    print(f"Condition number of the ellipsoid2: {np.linalg.cond(ellipsoid2.A)}")

    print(f"Difference between the two ellipsoids: {np.linalg.norm(ellipsoid.A - ellipsoid2.A)}")

    # check for all corners if they are contained in the ellipsoid
    assert np.all(ellipsoid.contains(corners))
    assert np.all(ellipsoid2.contains(corners))


    # MDG.contain_in_ellipsoid(generator, data, ellipsoid, direction, (0, b))
    if d == 1:
        plt.scatter(data[0, :], data[1, :])
        # # plot the corners of the hypercube and connect them
        # plt.scatter(corners[0, :], corners[1, :])
        # plot the line theta*x
        x = np.linspace(a, b, 3)
        plt.plot(x, direction * x, label="true hyperplane", color="red")
        # for i in range(corners.shape[1]):
        #     plt.plot([corners[0, i], corners[0, i]], [corners[1, i], corners[1, i]])
        # # plot the ellipse
        # ellipsoid.plot()
        ellipsoid2.plot(color="green")
        plt.axis("equal")
        plt.show()

    # test LeastSquaresCadro
    # tester = LeastSquaresTester(data, ellipsoid, nb_theta0=1)
    # results = tester.run()


if __name__ == "__main__":
    generator = np.random.default_rng(0)
    timings = []
    dimensions = [1, 5, 10, 15, 20]
    for d in dimensions:
        t1 = time()
        main(d, generator)
        t2 = time()
        timings.append(t2 - t1)
        print(f"dimension {d} took {round(t2 - t1, 3)} seconds")
    plt.plot(dimensions, timings)
    # log-log plot
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
