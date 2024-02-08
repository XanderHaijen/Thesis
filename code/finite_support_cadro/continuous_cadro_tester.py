import numpy as np
from ellipsoids import Ellipsoid
from continuous_cadro import CADRO1DLinearRegression
from multiple_dimension_cadro import LeastSquaresCadro
import matplotlib.pyplot as plt
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
        return self.problem.print_results(include_robust=True)

if __name__ == "__main__":
    generator = np.random.default_rng(0)
    d = 2
    n = 50
    direction = np.array([1., 1.])
    direction /= np.linalg.norm(direction)
    # sample uniformly on the unit square
    data = generator.uniform(size=(d, n))
    # values for y are <x, direction> + noise
    y = np.array([np.dot(data[:, i], direction) + generator.normal(0, 0.1) for i in range(n)])
    data = np.vstack((data, y))
    ellipse = Ellipsoid.lj_ellipsoid(data)
    # make 3d plot for the data
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[0, :], data[1, :], data[2, :])
    # draw the direction plane
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Data")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = np.dot(np.array([X[i, j], Y[i, j]]), direction)
    ax.plot_surface(X, Y, Z, alpha=0.2, color='r')
    plt.show()

    # test LeastSquaresCadro
    tester = LeastSquaresTester(data, ellipse, nb_theta0=2)
    tester.run()


