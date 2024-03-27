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
        self.d, self.m = data.shape
        self.solver = solver
        # setter for mu0 and sigma0
        self._get_moments()

        self.Rhat = None  # set by calling R_hat
        self.theta = None # set by calling solve


    @property
    def mu0(self):
        return self._get_moments()[0]

    @property
    def sigma0(self):
        return self._get_moments()[1]

    def _get_moments(self) -> (np.ndarray, np.ndarray):
        """
        Calculate the empirical mean and covariance of the data

        :return: mu0, sigma0 as np arrays
        """
        mu0 = np.mean(self.data, axis=1)
        sigma0 = np.cov(self.data, rowvar=True)

        return mu0, sigma0

    def R_hat(self) -> None:
        """
        Calculates the R_hat value as defined in Delage and Ye (2010), Cor. 3. It gives the maximum weighted distance
        from xi to the empirical mean.

        :return: None. The R_hat value is stored in the object.
        """
        mu0, sigma0 = self._get_moments()
        tau = cp.Variable()
        _lambda = cp.Variable()
        inv_sigma0 = np.linalg.inv(sigma0)
        inv_sigma0 = 0.5 * (inv_sigma0 + inv_sigma0.T) # make sure it is symmetric
        B = - inv_sigma0
        b = cp.reshape(inv_sigma0 @ mu0, (self.d, 1))
        beta = - mu0.T @ inv_sigma0 @ mu0 + tau
        A, a, c = self.ellipsoid.A, self.ellipsoid.a, self.ellipsoid.c

        M = cp.bmat([[B - _lambda * A, b - _lambda * a], [b.T - _lambda * a.T, beta - _lambda * c]])
        constraints = [M >> 0, _lambda >= 0]

        objective = cp.Minimize(tau)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.solver)

        if prob.status != "optimal":
            raise ValueError(f"Problem status: {prob.status}")

        self.Rhat = np.sqrt(tau.value)

    def assert_m(self) -> float:
        """
        Assert that the data size is large enough. This checks the condition given by Delage and Ye (2010), Eq. (13)
        """
        assert self.Rhat is not None, "R_hat is not set. Call R_hat() first."
        delta_1 = 1 - np.sqrt(1 - self.confidence)
        m1 = (self.Rhat ** 2 + 2) ** 2 * (2 + np.sqrt(2 + 2 * np.log(4 / delta_1))) ** 2
        m2 = ((8 + np.sqrt(32 * np.log(4 / self.confidence))) ** 2) / ((np.sqrt(self.Rhat + 4) - self.Rhat) ** 4)

        return np.ceil(max(m1, m2))

    def get_constants(self) -> (float, float):
        """
        Get the constants for the optimization problem. These are
        - gamma1: the confidence on the first moment
        - gamma2: the confidence on the second moment
        The formulas are as in Delage and Ye (2010), Cor. 3 and Eq. (15)

        :return: gamma1, gamma2
        """
        delta_bar = 1 - np.sqrt(1 - self.confidence)
        # Delage and Ye, Cor. 3
        # R_bar
        M = max(self.m, self.assert_m())
        temp = (2 + np.sqrt(2*np.log(4/self.confidence))) / np.sqrt(M)
        R_bar = self.Rhat / np.sqrt(1 - (self.Rhat**2+2) * temp)
        # alpha_bar and beta_bar
        alpha_bar = (R_bar ** 2 / np.sqrt(M)) * (np.sqrt(1 - self.d / R_bar ** 4) + np.sqrt(np.log(4 / delta_bar)))
        beta_bar = (R_bar ** 2 / M) * (2 + np.sqrt(2 * np.log(2 / delta_bar))) ** 2

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

    def solve(self, check_data: bool = True, verbose: bool = False):
        """
        Solve the moment DRO problem. This function returns the optimal value for theta. It implements the
        optimization problem as in Delage and Ye (2010), Eq. (6)
        """
        # step 1: calculate R_hat
        self.R_hat()

        # step 2: check if the data size is large enough
        if check_data:
            if not self.m >= self.assert_m():
                print("Data size is not large enough. The problem may not be solvable.")
                print(f"m: {self.m}, m_min: {self.assert_m()}")

        # step 3: get the constants
        gamma1, gamma2 = self.get_constants()
        mu0, sigma0 = self._get_moments()

        # step 4: define variables
        theta = cp.Variable(self.d-1)
        _lambda = cp.Variable()
        Q = cp.Variable((self.d, self.d), symmetric=True)
        q = cp.Variable(self.d)
        r = cp.Variable()
        t = cp.Variable()

        # objective function
        objective = r + t

        # constraints
        constraints = [Q >> 0] # constraint (3.8d)
        constraints += self._constraint_38b(t, Q, q, gamma1, gamma2, sigma0, mu0)  # constraint (3.8b)
        constraints += self._constraint_38c(_lambda, Q, q, theta, r)  # constraint (3.8c)

        # solve the problem
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=self.solver, verbose=verbose)

        if prob.status == cp.INFEASIBLE:
            raise ValueError("Problem is infeasible")
        elif prob.status == cp.OPTIMAL_INACCURATE:
            print("Problem is solved but the solution is inaccurate")

        self.theta = theta.value
        if verbose:
            print(f"Optimal theta: {self.theta}")
            # print values of the variables
            print(f"Q: {Q.value} \n")
            print(f"q: {q.value} \n")
            print(f"r: {r.value} \n")
            print(f"t: {t.value} \n")

        # step 4: define variables
        theta = cp.Variable(self.d-1)
        _lambda = cp.Variable()
        Q = cp.Variable((self.d, self.d), symmetric=True)
        q = cp.Variable(self.d)
        r = cp.Variable()
        P = cp.Variable((self.d, self.d), symmetric=True)
        p = cp.Variable(self.d)
        s = cp.Variable()

        objective = gamma2 * self._frob_prod(sigma0, Q) - cp.quad_form(mu0, Q) + r + \
            self._frob_prod(sigma0, P) - 2 * mu0.T @ p + gamma1 * s

        # constraints (number corresponds to the equation in Delage and Ye (2010))
        constraints = [Q >> 0]  # constraint (6d)
        constraints += [q + 2 * Q @ mu0 + 2 * p == 0]  # constraint (6b)
        constraints += self._constraint_6c(P, p, s)  # constraint (6c)
        constraints += self._constraint_6e(_lambda, Q, q, theta, r)  # constraint (6e)

        # solve the problem
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=self.solver, verbose=verbose)

        if prob.status == cp.INFEASIBLE:
            raise ValueError("Problem is infeasible")
        elif prob.status == cp.OPTIMAL_INACCURATE:
            print("Problem is solved but the solution is inaccurate")

        self.theta = theta.value
        if verbose:
            print(f"Optimal theta: {self.theta}")
            # print values of the variables
            print(f"Q: {Q.value} \n")
            print(f"q: {q.value} \n")
            print(f"r: {r.value} \n")
            print(f"P: {P.value} \n")
            print(f"p: {p.value} \n")
            print(f"s: {s.value} \n")
        return self.theta


    def _constraint_6c(self, P, p, s):
        """
        Constraint (6c) in Delage and Ye (2010)
        """
        _p = cp.reshape(p, (self.d, 1))
        _s = cp.reshape(s, (1, 1))
        M = cp.bmat([[P, _p], [_p.T, _s]])
        return [M >> 0]

    def _constraint_6e(self, _lambda, Q, q, theta, r):
        """
        Constraint (6e) in Delage and Ye (2010)
        """
        A, a, c = self.ellipsoid.A, self.ellipsoid.a, self.ellipsoid.c
        A12 = 0.5 * cp.reshape(q, (self.d, 1)) - _lambda * a
        A_bar = cp.bmat([[Q - _lambda * A, A12], [A12.T, r - _lambda * c]])
        ext = cp.vstack([-1.0, 0.0])
        Theta_ext = cp.vstack([cp.reshape(theta, (self.d-1, 1)), ext])
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
    n = 100
    d = 2
    a, b = 0, 5
    assert a < b
    sigma = 2
    slope = np.ones((d-1, ))
    train_x = generator.uniform(a, b, (n, d-1))
    train_y = np.array([np.dot(slope, x) for x in train_x]) + sigma * generator.standard_normal(n)
    data = np.vstack((train_x.T, train_y))
    ellipsoid = Ellipsoid.ellipse_from_corners(a * np.ones((d-1, )), b * np.ones((d-1, )), -3 * sigma, 3 * sigma,
                                               slope, scaling_factor=1.05)

    # test the MomentDRO class
    dro = MomentDRO(ellipsoid, data, confidence=0.95)
    dro.solve(verbose=True)
    print(dro.theta)


if __name__ == "__main__":
    moment_dro_tester(0)
