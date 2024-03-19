import numpy as np
import cvxpy as cp
from ellipsoids import Ellipsoid
from utils.multivariate_experiments import hypercube_corners, ellipse_from_corners


class MomentDRO:
    """
    Implements the Moment DRO algorithm as in Delage and Ye (2010) for the specific use case of multivariate linear
    regression. The algorithm is used to find the optimal distribution that minimizes the worst-case expected loss
    under the first and second moments of the distribution. The algorithm is based on the concept of support ellipsoid
    and the R_hat value. The algorithm is based on the following paper:
    - Delage, E., & Ye, Y. (2010). Distributionally robust optimization under moment uncertainty with application to
      data-driven problems. Operations Research, 58(3), 595-612.
    """
    def __init__(self, ellipsoid: Ellipsoid, data: np.ndarray, confidence: float,
                 solver: str = "MOSEK"):
        """
        :param ellipsoid: the support set of the distribution
        :param data: the data points as an m x d matrix where m is the number of points and d is the dimension
        :param confidence: the confidence level for which to solve
        """
        self.ellipsoid = ellipsoid
        self.data = data
        self.confidence = confidence
        self.m, self.d = data.shape
        self.solver = solver
        # setter for mu0 and sigma0
        self._get_moments()

        self.Rhat = None  # set by calling R_hat


    @property
    def mu0(self):
        return self.__mu0

    @property
    def sigma0(self):
        return self.__sigma0

    def _get_moments(self) -> None:
        """
        Calculate the empirical mean and covariance of the data

        :return: None. The empirical mean and covariance are stored in the object.
        """
        self.__mu0 = np.mean(self.data, axis=0)
        self.__sigma0 = np.cov(self.data, rowvar=False)

    def R_hat(self) -> None:
        """
        Calculates the R_hat value as defined in Delage and Ye (2010), Cor. 3. It gives the maximum weighted distance
        from xi to the empirical mean.

        :return: None. The R_hat value is stored in the object.
        """
        x = cp.Variable(self.d)
        # weighted norm
        objective = cp.Maximize(cp.quad_form(x - self.mu0, np.linalg.inv(self.sigma0)))
        # constraint
        A, a, c = self.ellipsoid.A, self.ellipsoid.a, self.ellipsoid.c
        constraints = [cp.quad_form(x, A) + 2 * a @ x + c >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.solver)

        if prob.status != "optimal":
            raise ValueError(f"Problem status: {prob.status}")

        self.Rhat = prob.value

    def assert_m(self) -> bool:
        """
        Assert that the data size is large enough. This checks the condition given by Delage and Ye (2010), Eq. (13)
        """
        assert self.Rhat is not None, "R_hat is not set. Call R_hat() first."
        delta = 1 - np.sqrt(1 - self.confidence)
        m1 = (self.Rhat ** 2 + 2)**2 * (2 + np.sqrt(2 + 2 * np.log(4 / delta))) ** 2
        m2 = ((8 + np.sqrt(32 * np.log(4 / self.confidence))) ** 2) / ((np.sqrt(self.Rhat + 4) - self.Rhat) ** 4)

        return self.m > max(m1, m2)

    def get_constants(self) -> (float, float):
        """
        Get the constants for the optimization problem. These are
        - gamma1: the confidence on the first moment
        - gamma2: the confidence on the second moment
        The formulas are as in Delage and Ye (2010), Cor. 3 and Eq. (15)

        :return: gamma1, gamma2
        """
        delta2 = 1 - np.sqrt(1 - self.confidence)
        # Delage and Ye, Cor. 3
        # R_bar
        temp = (2 + np.sqrt(2*np.log(4/self.confidence))) / np.sqrt(self.m)
        R_bar = self.Rhat / np.sqrt(1 - (self.Rhat**2+2) * temp)
        # alpha_bar and beta_bar
        alpha_bar = (R_bar ** 2 / np.sqrt(self.m)) * (np.sqrt(1 - self.m / R_bar ** 4) + np.sqrt(np.log(4 / delta2)))
        beta_bar = (R_bar ** 2 / self.m) * (2 + np.log(2 * np.log(2 / delta2)))

        # Delage and Ye, Eq. (15)
        gamma1 = beta_bar / (1 - alpha_bar - beta_bar)
        gamma2 = (1 + beta_bar) / (1 - alpha_bar - beta_bar)

        return gamma1, gamma2

    @staticmethod
    def _loss_matrices(theta, cvxpy=False):
        """
        Return the loss matrix used in the optimization problem. This is the loss matrix
        B = [theta; -1] [theta; -1]^T

        :param theta: the parameter for which the loss matrices are calculated
        :param cvxpy: a boolean value. If True, the function will return cvxpy expressions. Default is False.

        :return: the loss matrix B
        """
        if cvxpy:
            theta_ext = cp.vstack([theta, cp.reshape(-1.0, (1, 1))])
            return theta_ext @ theta_ext.T
        else:
            theta_ext = np.vstack([theta, -1.0])
            return theta_ext @ theta_ext.T

    def solve(self, check_data: bool = True):
        """
        Solve the moment DRO problem. This function returns the optimal value for theta. It implements the
        optimization problem as in Delage and Ye (2010), Eq. (6)
        """
        # step 1: calculate R_hat
        self.R_hat()

        # step 2: check if the data size is large enough
        if check_data:
            if not self.assert_m():
                raise ValueError("The data size is not large enough.")

        # step 3: get the constants
        gamma1, gamma2 = self.get_constants()


def moment_dro_tester(seed):
    generator = np.random.default_rng(seed)
    # generate data
    n = 100
    d = 2
    a, b = 0, 5
    assert a < b
    sigma = 2
    slope = np.ones((d-1, ))
    train_x = generator.uniform(a, b, (n, d-1))
    train_y = np.array([np.dot(slope, x) for x in train_x]) + sigma * generator.standard_normal(n)
    data = np.hstack((train_x, train_y.reshape(-1, 1)))
    corners = hypercube_corners(a, b, d, d_max=1e6, generator=generator)
    ellipsoid = ellipse_from_corners(corners.T, slope, ub=3*sigma, lb=3*sigma)

    # test the MomentDRO class
    dro = MomentDRO(ellipsoid, data, confidence=0.95)
    dro.solve()


if __name__ == "__main__":
    moment_dro_tester(0)
