import numpy as np


class ScalarDataGenerator:
    def __init__(self, x: np.ndarray, seed: int = 0):
        """
        :param x: x values (x is a d x m matrix)
        :param seed: seed for random number generator
        """
        self.generator = np.random.default_rng(seed)
        self.x = x

    def generate_linear_norm_disturbance(self, mu: float, sigma: float, theta_0: float,
                                         outliers: bool = False) -> np.ndarray:
        y = theta_0 * self.x + self.generator.normal(mu, sigma, size=len(self.x))
        if outliers:
            self._generate_outliers(y, theta_0, sigma)
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
