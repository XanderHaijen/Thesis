import numpy as np
from ellipsoids import Ellipsoid
from one_dimension_cadro import CADRO1DLinearRegression
from multiple_dimension_cadro import LeastSquaresCadro
from stochastic_dominance_cadro import StochasticDominanceCADRO
import matplotlib.pyplot as plt
from utils.data_generator import MultivariateDataGenerator as MDG

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
        self.nb_theta0 = nb_theta0

    def run(self):
        self.problem.solve(theta0=self.theta0, theta=self.theta, confidence_level=self.confidence_level,
                           nb_theta_0=self.nb_theta0)

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


class StochasticDominanceTester(ContinuousCADROTester):
    def __init__(self, data: np.ndarray, ellipse: Ellipsoid, theta0: np.ndarray = None,
                 theta: np.ndarray = None, confidence_level: float = 0.05,
                 nb_thresholds: int = 100, threshold_mode: str = 'equidistant'):
        super().__init__(data, ellipse)
        self.problem = StochasticDominanceCADRO(data, ellipse,
                                                nb_thresholds=nb_thresholds, threshold_type=threshold_mode)
        self.theta0 = theta0
        self.theta = theta
        self.confidence_level = confidence_level

    def run(self):
        self.problem.solve(theta0=self.theta0, theta=self.theta, confidence_level=self.confidence_level)
        self.problem.set_theta_r()
        self.theta = self.problem.theta
        self.problem.print_results(include_robust=True)
        return self.problem.results


if __name__ == "__main__":

    generator = np.random.default_rng(0)
    d = 5
    slope = 2 * np.ones((d - 1, ))
    m = 100
    x = 10 * MDG.uniform_unit_hypercube(generator, d - 1, m)
    y = np.array([np.dot(x[:, i], np.ones((d - 1,))) for i in range(m)]) + \
        MDG.normal_disturbance(generator, 1, m, True)
    data = np.vstack((x, y))
    ellipsoid = Ellipsoid.ellipse_from_corners(np.array([0] * (d - 1)), np.array([10] * (d - 1)), -5, 5,
                                               slope - 0.5, 1.05)
    # tester = CADRO1DLinearRegressionTester(data, ellipsoid, nb_theta0=2)
    # generate 100 Chebyshev nodes on [0, 1]
    nodes = 0.5 * np.cos((2 * np.arange(1, 101) - 1) * np.pi / 200) + 0.5
    tester = StochasticDominanceTester(data, ellipsoid)
    results = tester.run()
    print("theta ", results["theta"])
    print("theta_r ", results["theta_r"])
    print("theta_0 ", results["theta_0"])

    nodes = tester.problem.thresholds
    lambdas = results["lambda"]
    alphas = results["alpha"]
    plt.scatter(nodes, alphas / np.max(alphas), marker='o', color='b', label='alpha (normalized)')
    plt.scatter(nodes, lambdas, marker='.', color='r', label='lambda')
    plt.legend()
    plt.show()