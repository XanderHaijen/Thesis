import numpy as np
import cvxpy as cp
from ellipsoids import Ellipsoid
import utils.multivariate_experiments as aux

class MomentDRO:
    """
    Implements the Moment DRO algorithm as in Delage and Ye (2010) for the specific use case of multivariate linear
    regression. The algorithm is used to find the optimal distribution that minimizes the worst-case expected loss
    under the first and second moments of the distribution. The algorithm is based on the concept of support ellipsoid
    and the R_hat value. The algorithm is based on the following papers:

    Delage, E., & Ye, Y. (2010). Distributionally Robust Optimization Under Moment Uncertainty with Application to
    Data-Driven Problems. Operations Research, 58(3), 595-612. https://doi.org/10.1287/opre.1090.0741

    Coppens, P., Schuurmans, M., & Patrinos, P. (2020). Data-driven distributionally robust LQR with multiplicative
    noise. Proceedings of the 2nd Conference on Learning for Dynamics and Control, 521–530.
    https://proceedings.mlr.press/v120/coppens20a.html

    """

    def __init__(self, ellipsoid: Ellipsoid, data: np.ndarray, confidence: float, sigmaG: float = 1.0,
                 solver: str = "MOSEK"):
        """
        :param ellipsoid: the support set of the distribution
        :param data: the data points as an m x d matrix where m is the number of points and d is the dimension
        :param confidence: the confidence level for which to solve
        """
        self.ellipsoid = ellipsoid
        self.data = data
        self.confidence = confidence
        self.d, self.m = data.shape
        self.solver = solver
        # setter for mu0 and sigma0
        self._get_moments()
        self.sigmaG = sigmaG

        self.theta = None  # set by calling solve
        self._objective = None  # set by calling solve

    @property
    def mu0(self):
        return self._get_moments()[0]

    @property
    def sigma0(self):
        return self._get_moments()[1]

    @property
    def cost(self):
        return self._objective

    def set_sigma(self):
        """
        Calculate the empirical covariance of the data
        """
        A = self.ellipsoid.shape
        # largest eigenvalue of A
        eigval, _ = np.linalg.eigh(A)
        lambda_max = np.max(eigval)
        return lambda_max

    def _get_moments(self) -> (np.ndarray, np.ndarray):
        """
        Calculate the empirical mean and covariance of the data

        :return: mu0, sigma0 as np arrays
        """
        mu0 = np.mean(self.data, axis=1)
        sigma0 = np.cov(self.data, rowvar=True, bias=True)  # + 1e-8 * np.eye(self.data.shape[0])

        return mu0, sigma0

    # def R_hat(self) -> None:
    #     """
    #     Calculates the R_hat value as defined in Delage and Ye (2010), Cor. 3. It gives the maximum weighted distance
    #     from xi to the empirical mean.
    #
    #     :return: None. The R_hat value is stored in the object.
    #     """
    #     mu0, sigma0 = self._get_moments()
    #     tau = cp.Variable()
    #     _lambda = cp.Variable()
    #     inv_sigma0 = np.linalg.solve(sigma0, np.eye(self.d))
    #     inv_sigma0 = 0.5 * (inv_sigma0 + inv_sigma0.T)  # make sure it is symmetric
    #     B = - inv_sigma0
    #     b = cp.reshape(inv_sigma0 @ mu0, (self.d, 1))
    #     beta = - mu0.T @ inv_sigma0 @ mu0 + tau
    #     A, a, c = self.ellipsoid.A, self.ellipsoid.a, self.ellipsoid.c
    #
    #     M = cp.bmat([[B - _lambda * A, b - _lambda * a], [b.T - _lambda * a.T, beta - _lambda * c]])
    #     constraints = [M >> 0, _lambda >= 0]
    #
    #     objective = cp.Minimize(tau)
    #     prob = cp.Problem(objective, constraints)
    #     prob.solve(solver=self.solver)
    #
    #     if prob.status != "optimal":
    #         raise ValueError(f"Problem status: {prob.status}")
    #
    #     self.Rhat = np.sqrt(prob.value)
    #
    # def assert_m(self) -> int:
    #     """
    #     Assert that the data size is large enough. This checks the condition given by Delage and Ye (2010)
    #     """
    #     assert self.Rhat is not None, "R_hat is not set. Call R_hat() first."
    #     delta_bar = 1 - np.sqrt(1 - self.confidence)
    #     m1 = (self.Rhat ** 2 + 2) ** 2 * (2 + np.sqrt(2 * np.log(4 / delta_bar))) ** 2
    #     m2 = ((8 + np.sqrt(32 * np.log(4 / delta_bar))) ** 2) / ((np.sqrt(self.Rhat + 4) - self.Rhat) ** 4)
    #
    #     return int(np.ceil(max(m1, m2)))

    def assert_m(self, p, q, epsilon=1 / 30) -> int:
        """
        Returns the theoretically minimal data size for which the problem is solvable. This is based on the
        theoretical results from Coppens, Schuurmans and Patrinos (2020) (Eq 11)
        """
        sigma = self.sigmaG
        M1 = sigma ** 2 * np.sqrt(32 * q)
        M2 = np.sqrt(32 * sigma ** 4 * q +
                     8 * sigma ** 2 * (1 - 2 * epsilon) * q +
                     4 * sigma ** 2 * (1 - 2 * epsilon) ** 2 * p)

        M = ((M1 + M2) / (2 * (1 - 2 * epsilon))) ** 2

        return int(np.ceil(M))

    def get_parameters(self, confidence_level, epsilon=1 / 30):
        """
        Get the parameters for the optimization problem. These are the confidence levels for the first and second moments
        based on Coppens, Schuurmans and Patrinos (2020)
        :param confidence_level: the confidence level for the optimization problem
        :param sigma: the standard deviation of the noise (see Assumption 1)
        :param epsilon: the epsilon value for the confidence level (see Thm 6)
        :return: confidence radius for the first and second moments
        """
        assert 0 < confidence_level < 1, "Beta should be between 0 and 1"
        assert 0 < epsilon < 1 / 2, "Epsilon should be between 0 and 1/2"

        q = self.d * np.log(1 + 1 / epsilon) + np.log(2 / confidence_level)  # Thm 6
        p = self.d + 2 * np.sqrt(self.d * np.log(1 / confidence_level)) + 2 * np.log(1 / confidence_level)  # Thm 7

        return p, q

    # def get_constants(self) -> (float, float):
    #     """
    #     Get the constants for the optimization problem. These are
    #     - gamma1: the confidence on the first moment
    #     - gamma2: the confidence on the second moment
    #     The formulas are as in Delage and Ye (2010), Cor. 3
    #
    #     :return: gamma1, gamma2
    #     """
    #     delta_bar = 1 - np.sqrt(1 - self.confidence)
    #     # Delage and Ye, Cor. 3
    #     # R_bar
    #     M = max(self.m, self.assert_m())
    #     temp = (2 + np.sqrt(2 * np.log(4 / delta_bar))) / np.sqrt(M)
    #     R_bar = self.Rhat / np.sqrt(1 - (self.Rhat ** 2 + 2) * temp)
    #     # alpha_bar and beta_bar
    #     alpha_bar = (R_bar ** 2 / np.sqrt(M)) * (np.sqrt(1 - self.d / R_bar ** 4) + np.sqrt(np.log(4 / delta_bar)))
    #     beta_bar = (R_bar ** 2 / M) * (2 + np.sqrt(2 * np.log(2 / delta_bar))) ** 2
    #
    #     # Delage and Ye, Eq. (15)
    #     gamma1 = beta_bar / (1 - alpha_bar - beta_bar)
    #     gamma2 = (1 + beta_bar) / (1 - alpha_bar - beta_bar)
    #
    #     return gamma1, gamma2

    def get_constants(self, p, q, epsilon=1 / 30, confidence_level=0.05):
        """
        Get the constants for the optimization problem. These are the confidence levels for the first and second moments
        based on Coppens, Schuurmans and Patrinos (2020)
        :param p: (see Thm 7)
        :param q: (see Thm 6)
        :param epsilon: the epsilon value for the confidence level (see Thm 6)
        :param confidence_level: the confidence level for the optimization problem
        :param sigma: the standard deviation (see Assumption 1)
        :return: confidence radius for the first and second moments
        """
        assert 0 < epsilon < 1 / 2, "Epsilon should be between 0 and 1/2"
        assert 0 < confidence_level < 1, "Confidence level should be between 0 and 1"

        C = 1.1  # safety factor
        M = max(self.m, int(self.assert_m(p, q) * C))
        sigma = self.sigmaG

        t_sigma = (sigma ** 2) / (1 - 2 * epsilon) * (np.sqrt(32 * q / M) + 2 * q / M)  # Eq 10
        t_mu = sigma ** 2 / M * p  # Thm 7

        r_sigma = 1 / (1 - t_mu - t_sigma)
        r_mu = t_mu / (1 - t_mu - t_sigma)

        return r_mu, r_sigma

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

    def solve(self, check_data: bool = True, verbose: bool = False):
        """
        Solve the moment DRO problem. This function returns the optimal value for theta. It implements the
        optimization problem as in Delage and Ye (2010), Eq. (6)
        """
        # step 1: calculate p and q
        p, q = self.get_parameters(self.confidence / 2)
        # Note: Thms 6-7 give formulas for p(beta) and q(beta). We calculate b(beta/2) and a(beta/2) instead
        # because these are the values we need later on (see Thm 8)

        # step 2: check if the data size is large enough
        if check_data:
            if not self.m >= self.assert_m(p, q):
                print("Data size is not large enough. The problem may not be solvable.")
                print(f"m: {self.m}, m_min: {self.assert_m(p, q)}")

        # step 3: get the constants
        gamma1, gamma2 = self.get_constants(p, q, self.confidence)
        mu0, sigma0 = self._get_moments()

        # step 4: define variables
        theta = cp.Variable(self.d - 1)
        _lambda = cp.Variable((), 'lambda')
        Q = cp.Variable((self.d, self.d), 'Q', symmetric=True)
        q = cp.Variable(self.d, 'q')
        r = cp.Variable((), 'r')
        t = cp.Variable((), 't')

        # objective function
        objective = t + r

        # constraints
        constraints = [Q >> 0 * np.eye(Q.shape[0])]  # constraint (8d)
        constraints += self._constraint_8c(t, Q, q, gamma1, gamma2, sigma0, mu0)  # constraint (8b)
        constraints += self._constraint_8b(_lambda, Q, q, theta, r)  # constraint (8c)

        # solve the problem
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=self.solver, verbose=verbose)

        if prob.status == cp.INFEASIBLE:
            raise ValueError("Problem is infeasible")
        elif prob.status == cp.OPTIMAL_INACCURATE:
            print("Problem is solved but the solution is inaccurate")

        self.theta = theta.value
        self._objective = prob.value
        if verbose:
            print(f"Optimal theta: {self.theta}")
            # print values of the variables
            print(f"Q: {Q.value} \n")
            print(f"q: {q.value} \n")
            print(f"r: {r.value} \n")
            print(f"t: {t.value} \n")

    def _constraint_8c(self, t, Q, q, gamma1, gamma2, sigma0, mu0):
        """
        Constraint (8b) in Delage and Ye (2010)
        """
        eigval, eigvec = np.linalg.eigh(sigma0)
        assert np.min(eigval) > 1e-6, "Σ is not pd"
        sigma0_sqrt = eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T
        # print(mu0)
        return [t >= self._frob_prod(gamma2 * sigma0 + np.outer(mu0, mu0), Q) +
                mu0 @ q +
                np.sqrt(gamma1) * cp.norm2(sigma0_sqrt @ (q + Q @ (2 * mu0)))
                ]

    def _constraint_8b(self, _lambda, Q, q, theta, r):
        """
        Constraint (6e) in Delage and Ye (2010)
        """
        A, a, c = self.ellipsoid.A, self.ellipsoid.a, self.ellipsoid.c
        A12 = 0.5 * cp.reshape(q, (self.d, 1)) - _lambda * a
        A_bar = cp.bmat([[Q - _lambda * A, A12], [A12.T, r - _lambda * c]])
        ext = cp.vstack([-1.0, 0.0])
        Theta_ext = cp.vstack([cp.reshape(theta, (self.d - 1, 1)), ext])
        M = cp.bmat([[A_bar, Theta_ext], [Theta_ext.T, cp.reshape(1.0, (1, 1))]])
        return [M >> 0, _lambda >= 0]

    @staticmethod
    def _frob_prod(A, B, cvxpy=True):
        """
        Calculate the Frobenius product of two matrices A and B. This is the sum of the element-wise product of the
        matrices.

        :param A: the first matrix
        :param B: the second matrix
        :param cvxpy: a boolean value. If True, the function will return a cvxpy expression. Default is True.

        :return: the Frobenius product of A and B
        """
        if cvxpy:
            return cp.sum(cp.multiply(A, B))
        else:
            return np.sum(np.multiply(A, B))

    def loss(self, theta=None):
        if theta is None:
            theta = self.theta

        return self._loss_function(theta, self.data)

    def test_loss(self, test_data, theta=None):
        if theta is None:
            theta = self.theta

        return self._loss_function(theta, test_data)

    @staticmethod
    def _loss_function(theta, data, cvxpy=False):
        """
        This function calculates the loss function for the given theta and data. The loss function is defined as
        (x * theta^T - y)^2 summed over all data points.

        :param theta: A (d, 1) vector. This is the parameter for which the loss function is calculated.
        :param data: A (d, m) matrix. Each column in this matrix represents a data point.
        :param cvxpy: A boolean value. If True, the function will return a cvxpy expression. Default is False.

        :return: The loss function calculated for the given theta and data.
        """
        if len(data.shape) > 1:
            H = data[:-1, :]
            h = data[-1, :].reshape((1, -1))
            nb_samples = data.shape[1]
        else:
            H = data[:-1].reshape((-1, 1))
            h = data[-1].reshape((1, 1))
            nb_samples = 1

        # If cvxpy is True, return a cvxpy expression for the loss function
        if cvxpy:
            loss = cp.sum_squares(cp.matmul(cp.transpose(theta), H) - h) / nb_samples
        else:
            loss = np.sum((np.dot(np.transpose(theta), H) - h) ** 2) / nb_samples

        return loss


def moment_dro_tester(seed):
    generator = np.random.default_rng(seed)
    # generate data
    n = 150
    d = 5
    a, b = -2, 2
    assert a < b
    sigma = 1

    # set SigmaG
    SigmaG = aux.subgaussian_parameter(d, a, b, -3, 3, np.ones((d - 1, )))

    slope = np.ones((d - 1,))
    train_x = generator.uniform(a, b, (n, d - 1))
    train_y = np.array([np.dot(slope, x) for x in train_x]) + sigma * generator.standard_normal(n)
    data = np.vstack((train_x.T, train_y))
    ellipsoid = Ellipsoid.ellipse_from_corners(a * np.ones((d - 1,)), b * np.ones((d - 1,)), -3 * sigma, 3 * sigma,
                                               slope, scaling_factor=1.05)

    # test the MomentDRO class
    dro = MomentDRO(ellipsoid, data, confidence=0.05, sigmaG=SigmaG, solver="MOSEK")
    dro.solve(verbose=False, check_data=True)
    print(dro.theta)


if __name__ == "__main__":
    moment_dro_tester(0)
