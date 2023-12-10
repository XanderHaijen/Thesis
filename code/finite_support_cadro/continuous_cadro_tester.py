import numpy as np
from ellipsoids import Ellipsoid
from continuous_cadro import CADRO1DLinearRegression
class ContinuousCADROTester:
    def __init__(self, data: np.ndarray, ellipse: Ellipsoid):
        self.data = data
        self.ellipsoid = ellipse



class CADRO1DLinearRegressionTester(ContinuousCADROTester):
    def __init__(self, data: np.ndarray, ellipse: Ellipsoid, theta0: np.ndarray = None,
                 theta: float = None, confidence_level: float = 0.05):
        super().__init__(data, ellipse)
        self.problem = CADRO1DLinearRegression(data, ellipse)
        self.theta0 = theta0
        self.theta = theta
        self.confidence_level = confidence_level

    def run(self):
        self.problem.solve(theta0=self.theta0, theta=self.theta, confidence_level=self.confidence_level)
        return self.problem.print_results(include_robust=True)


if __name__ == "__main__":
    generator = np.random.default_rng(0)
    x = np.linspace(-1, 1, 20)
    y = 2 * x + generator.beta(2, 2, size=len(x))  # noise is beta distributed
    data = np.vstack((x, y))
    ellipsoid = Ellipsoid.smallest_enclosing_sphere(data)
    tester = CADRO1DLinearRegressionTester(data, ellipsoid)
    tester.run()