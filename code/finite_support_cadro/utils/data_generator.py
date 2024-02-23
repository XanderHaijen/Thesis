import numpy as np
import sys
sys.path.append("..")
# ignore error below
from finite_support_cadro.ellipsoids import Ellipsoid


class ScalarDataGenerator:
    def __init__(self, x: np.ndarray, seed: int = 0):
        """
        :param x: x values (x is a d x m matrix)
        :param seed: seed for random number generator
        """
        self.generator = np.random.default_rng(seed)
        self.x = x
        self.__y = np.zeros(len(x))

    @property
    def y(self):
        # return a copy of y
        return self.__y.copy()

    def generate_linear_norm_disturbance(self, mu: float, sigma: float, theta_0: float,
                                         outliers: bool = False) -> np.ndarray:
        y = theta_0 * self.x + self.generator.normal(mu, sigma, size=len(self.x))
        if outliers:
            self._generate_outliers(y, theta_0, sigma)
        self.__y = y
        return y

    def generate_linear_beta_disturbance(self, a: float, b: float, theta_0: float,
                                         outliers: bool = False) -> np.ndarray:
        y = theta_0 * self.x + self.generator.beta(a, b, size=len(self.x))

        # standard deviation of beta distribution is sqrt(ab / (a + b)^2(a + b + 1))
        sigma = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        if outliers:
            self._generate_outliers(y, theta_0, sigma)
        return y

    def _generate_outliers(self, y: np.ndarray, theta_0: float, sigma: float) -> None:
        indices = self.generator.choice(len(self.x), size=int(len(self.x) / 7), replace=False)
        y[indices] = (theta_0 * self.x[indices] +
                      self.generator.choice([-1, 1], size=int(len(self.x) / 7)) * 5 * sigma)

    def contain_within_ellipse(self, ellipse: Ellipsoid) -> None:
        """
        Checks if all data is contained within the ellipse. If not, it re-samples the violating data points until
        all data is contained within the ellipse.
        :param ellipse: Ellipsoid object
        :return: None. y is modified in place.
        """
        for i in range(len(self.x)):
            while not ellipse.contains(np.array([self.x[i], self.__y[i]])):
                self.__y[i] = self.generator.normal(0, 1)


class MultivariateDataGenerator(np.random.RandomState):
    """
    MultivariateDataGenerator is a wrapper around np.random.default_rng and provides some
    useful methods for experimentation with multivariate CADRO
    """
    def __init__(self):
        pass

    @staticmethod
    def uniform_unit_square(generator: np.random.default_rng, dimension: int, n: int) -> np.ndarray:
        """
        Generates n d-dimensional vectors uniformly distributed on the unit square
        """
        return generator.uniform(size=(dimension, n))

    @staticmethod
    def normal_disturbance(generator: np.random.default_rng,
                           stdev: float, n: int, outliers: bool = False,
                           outlier_rate: int = 7, outlier_size: float = 10) -> np.ndarray:
        """
        Generates n scalars distributed according to the normal distribution with mean 0 and standard deviation stdev.
        This function generates a row vector
        """
        y = generator.normal(scale=stdev, size=n, loc=0)

        if outliers:
            p = [0.5 / outlier_rate, 0.5 / outlier_rate, 1 - 1 / outlier_rate]
            y += generator.choice([-1, 1, 0], p=p, size=n, replace=True) * stdev * outlier_size

        return y

    @staticmethod
    def contain_in_ellipsoid(generator: np.random.default_rng, data: np.ndarray, ellipsoid: Ellipsoid, slope: np.ndarray,
                             x_range: tuple = (0, 1)) -> None:
        """
        Re-samples points which are not in the provided Ellipsoid. Re-sampling is done according to the standard
        normal distribution.
        """
        d, n = data.shape
        for i in range(n):
            while not ellipsoid.contains(data[:, i]):
                # re-sample the point from the range of x (range is a hypercube with lengths x_range)
                data[:-1, i] = np.array([generator.uniform(x_range[0], x_range[1]) for _ in range(d - 1)])
                data[-1, i] = np.dot(data[:-1, i], slope) + generator.normal(scale=1)



