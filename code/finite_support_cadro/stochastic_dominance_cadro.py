from typing import Union
from robust_optimization import RobustOptimization
from continuous_cadro import ContinuousCADRO
import numpy as np
from ellipsoids import Ellipsoid
import cvxpy as cp
import study_minimal


class StochasticDominanceCADRO(ContinuousCADRO):
    """
    Stochastic Dominance CADRO for least squares objective function:
    J(theta) = || A theta - b ||_2^2
    """

    def __init__(self, data: np.ndarray, ellipsoid: Ellipsoid,
                 nb_thresholds: int = 50, threshold_type: Union[str, np.ndarray] = 'equidistant',
                 solver=cp.MOSEK, split=None, seed=0):
        """
        :param data: (d, m) matrix containing the data
        :param ellipsoid: Ellipsoid object
        :param split: split the data into two parts. If None, the default split is used.
        """
        super().__init__(data, ellipsoid, solver, split, seed)
        self.threshold_type = threshold_type
        if isinstance(threshold_type, np.ndarray):
            self.nb_thresholds = len(threshold_type)
        elif isinstance(threshold_type, str):
            self.nb_thresholds = nb_thresholds
        self.thresholds = None  # thresholds for the stochastic dominance constraints


    @property
    def loss(self):
        """
        :return: the loss function at theta_star
        """
        if self.theta is None:
            raise ValueError("theta has not been set yet.")
        else:
            return self._loss_function(self.theta, self.data)

    @property
    def loss_r(self) -> float:
        """
        :return: the loss function at theta_r
        """
        if self.theta_r is None:
            self.set_theta_r()

        return self._loss_function(self.theta_r, self.data)

    @property
    def loss_0(self, index: Union[int, None] = 0):
        """
        Returns the loss function at theta_0.
        :param index: index of theta_0 to use. If None, the average loss over all initial theta_0 is returned.
        :return: the loss function at theta_0
        """
        if self.theta_0 is None:
            raise ValueError("theta_0 has not been set yet.")
        else:
            return self._loss_function(self.theta_0[0], self.data)

    @property
    def eta_bar(self):
        """
        Compute the average loss over the data.
        """
        return self._eta_bar()

    def loss_array(self, data: np.ndarray = None, theta: str = "theta") -> np.ndarray:
        """
        Get the loss function for all the data points.
        """
        if not isinstance(theta, str):
            raise ValueError("theta must be a string. Must be either 'theta', 'theta_r' or 'theta_0'.")

        if theta == "theta":
            theta = self.theta
        elif theta == "theta_r":
            theta = self.theta_r
        elif theta == "theta_0":
            theta = self.theta_0[0]

        if data is None:
            return np.array([self._loss_function(theta, self.data[:, i]) for i in range(self.data.shape[1])])
        else:
            return np.array([self._loss_function(theta, data[:, i]) for i in range(data.shape[1])])

    def set_theta_r(self):
        """
        Set theta_r to be the solution of the robust optimization problem.
        """
        robust_opt = RobustOptimization(self.ellipsoid, solver=self.solver)
        robust_opt.solve_least_squares()
        self.theta_r = np.reshape(robust_opt.theta, (-1, 1))

    def set_theta_0(self, theta_0: np.ndarray = None, nb_theta_0: int = 1):
        if nb_theta_0 != 1:
            raise ValueError("nb_theta_0 must be 1 for Stochastic Dominance CADRO.")

        if theta_0 is None:
            train_data_indices = np.arange(0, self.split)
            self._find_theta0(train_data_indices, 1)
        else:
            assert len(theta_0) == 1
            self.theta_0 = theta_0

    def _find_theta0_batch(self, data_batch_indices: np.ndarray):
        """
        Find theta_0 for the given batch of data indices. Use the sample average approximation (SAA) method.
        :param data_batch_indices: indices of the data to use for the SAA method
        :return: theta_0
        """
        theta0 = cp.Variable((self.data.shape[0] - 1, 1))
        train_data = self.data[:, data_batch_indices]
        loss = self._loss_function(theta0, train_data, cvxpy=True)
        problem = cp.Problem(cp.Minimize(loss))
        problem.solve(solver=self.solver)
        return theta0.value

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

    @staticmethod
    def _scalar_loss(theta, x, y, cvxpy=False) -> float:
        """
        This function calculates the loss function for the given theta and a single data point (x, y). The loss function
        is defined as (x * theta^T - y)^2.
        :param theta: A (d, 1) vector. This is the parameter for which the loss function is calculated.
        :param x: A (d, 1) vector. This is the input data point.
        :param y: A scalar. This is the label for the input data point.
        :param cvxpy: A boolean value. If True, the function will return a cvxpy expression. Default is False.
        :return: The loss function calculated for the given theta and data as a scalar.
        """
        if not cvxpy:
            loss = (theta.T @ x - y) ** 2
            return loss[0]  # reshape to scalar
        else:
            return cp.square(cp.matmul(cp.transpose(theta), x) - y)

    def calibrate(self, confidence_level):
        """
        Calibrate the ellipsoid for the given confidence level.
        :param confidence_level: the confidence level for the ellipsoid
        """
        if self.theta_0 is None:
            raise ValueError("theta_0 has not been set yet.")

        self.alpha = np.zeros(self.nb_thresholds)

        if self.thresholds is None:
            self.thresholds = self._compute_thresholds(self.nb_thresholds, self.threshold_type)

        for i, threshold in enumerate(self.thresholds):
            self.alpha[i] = self._set_alpha(threshold, confidence_level)

    def _set_alpha(self, threshold, confidence_level, asym_cutoff: int = 80):
        m_cal = self.data.shape[1] - self.split
        method = "asymptotic" if m_cal > asym_cutoff else 'brentq'
        calibration = study_minimal.calibrate(length=m_cal, method=method, level=confidence_level,
                                              full_output=True)
        gamma = calibration.info['radius']
        kappa = int(np.ceil(m_cal * gamma))

        # function phi_i(theta_0, x) = 1_{l(theta_0, x) > threshold}
        eta = np.array([1 if self._scalar_loss(self.theta_0[0], self.data[:-1, i], self.data[-1, i]) >= threshold
                        else 0 for i in range(self.split, self.data.shape[1])])

        eta.sort(axis=0)
        eta_bar = 1 if self._eta_bar() >= threshold else 0
        alpha = (kappa / m_cal - gamma) * eta[kappa - 1] + np.sum(eta[kappa:m_cal]) / m_cal + eta_bar * gamma
        return alpha

    def _find_theta_star(self, theta=None):
        """
        Solve the multivariate least squares problem using the CADRO method.

        :param theta: the value for theta. If None, theta is an optimization variable. Default is None.

        :return: None. All values are stored in the object.
        """
        self.theta = cp.Variable(shape=(self.data.shape[0] - 1, 1)) if theta is None else theta
        if theta is not None:
            assert theta.shape == (self.data.shape[0] - 1, 1)
        self.lambda_ = cp.Variable(self.nb_thresholds)
        self.tau = cp.Variable(1)
        self.gamma = cp.Variable(3 * self.nb_thresholds)

        self.thresholds = self._compute_thresholds(self.nb_thresholds, self.threshold_type)

        objective = cp.Minimize(self.tau + cp.sum(self.lambda_ * self.alpha))
        constraints = [lambda_ >= 0 for lambda_ in self.lambda_] + \
                      [gamma >= 0 for gamma in self.gamma]

        constraints += self.stochastic_dominance_constraints()

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)

        if problem.status == cp.INFEASIBLE:
            raise ValueError("Problem is infeasible")
        elif problem.status == cp.OPTIMAL_INACCURATE:
            print("Problem is solved but the solution is inaccurate")

        self.lambda_ = self.lambda_.value
        self.tau = self.tau.value[0]
        self.gamma = self.gamma.value
        if isinstance(self.theta, cp.Variable):
            self.theta = self.theta.value[:, 0]

    def stochastic_dominance_constraints(self):
        """
        Return the stochastic dominance constraints for the optimization problem.
        :return: a list containing the stochastic dominance constraints
        """
        constraints = []

        B0, b0, beta0 = self._loss_matrices(self.theta_0[0], cvxpy=True)
        A, a, c = self.ellipsoid.A, self.ellipsoid.a, self.ellipsoid.c

        ext = cp.vstack([-1, 0])
        theta_vector = cp.vstack([self.theta, ext])

        for i in range(0, self.nb_thresholds):
            # we have
            # A_bar - sum_j=1^3 gamma_ij * B_bar_j >= 0

            # 1. construct A_bar
            # A_bar = [[M_bar, Theta], [Theta^T, 1]]
            M_bar_11 = np.zeros((self.data.shape[0], self.data.shape[0]))
            M_bar_12 = np.zeros((self.data.shape[0], 1))
            M_bar_21 = cp.transpose(M_bar_12)
            M_bar_22 = cp.sum(self.lambda_[:i]) + self.tau if i > 0 \
                else self.tau
            M_bar = cp.bmat([[M_bar_11, M_bar_12], [M_bar_21, cp.reshape(M_bar_22, (1, 1))]])

            # 2. construct B_bars
            v_next = self.thresholds[i]
            v_prev = self.thresholds[i - 1] if i > 0 else 0
            B_bar_1 = cp.bmat([[A, a], [cp.transpose(a), c]])
            B_bar_2 = cp.bmat([[B0, b0], [cp.transpose(b0), cp.reshape(beta0 - v_prev, (1,1))]])
            B_bar_3 = cp.bmat([[-B0, -b0], [-cp.transpose(b0), cp.reshape(-beta0 + v_next, (1, 1))]])
            B_bar = - self.gamma[3 * i] * B_bar_1 - self.gamma[3 * i + 1] * B_bar_2 - self.gamma[3 * i + 2] * B_bar_3

            # 3. construct the constraint
            C = B_bar + M_bar
            X = cp.bmat([[C, theta_vector], [cp.transpose(theta_vector), cp.reshape(1.0, (1, 1))]])
            constraints += [X >> 0]

        return constraints

    def test_loss(self, test_data: np.ndarray, type: str = 'theta', index: int = 0) -> float:
        """
        Compute the loss on the test data.
        :param test_data: (d, m) matrix containing the test data
        :param type: the type of loss to compute. Must be "theta_0", "theta_r" or "theta".
        :param index: index of the theta_0 to use. Default is 0. This argument is only used when type is "theta_0".

        :return: the loss on the test data
        """
        if type not in ['theta_0', 'theta_r', 'theta']:
            raise ValueError('Invalid type argument. Must be "theta_0", "theta_r" or "theta".')

        if type == 'theta_0':
            return self._loss_function(self.theta_0[index], test_data)
        elif type == 'theta_r':
            return self._loss_function(self.theta_r, test_data)
        else:
            return self._loss_function(self.theta, test_data)

    def _loss_matrices(self, theta, cvxpy=False):
        """
        Return the loss matrices used in the optimization problem. These are the loss matrices
        when the loss is computed for one data point.

        :param theta: the parameter for which the loss matrices are calculated
        :param cvxpy: a boolean value. If True, the function will return cvxpy expressions. Default is False.

        :return: the loss matrices B, b, and beta as cvxpy expressions or numpy arrays
        """
        if cvxpy:
            theta_ext = cp.vstack([theta, cp.reshape(-1.0, (1, 1))])
            return theta_ext @ theta_ext.T, np.zeros((self.data.shape[0], 1)), 0
        else:
            theta_ext = np.vstack([theta, -1.0])
            return theta_ext @ theta_ext.T, np.zeros((self.data.shape[0], 1)), 0

    def _compute_thresholds(self, nb_thresholds, threshold_type):
        min_threshold = 0
        max_threshold = self._eta_bar()
        if isinstance(threshold_type, str):
            if threshold_type == 'equidistant':
                return np.linspace(min_threshold, max_threshold, nb_thresholds)
            else:
                raise ValueError("Invalid threshold_type. Must be 'equidistant'.")
        elif isinstance(threshold_type, np.ndarray):
            # assert that threshold_type is in [0, 1]
            assert np.all(0 <= threshold_type) and np.all(threshold_type <= 1)
            thresholds = min_threshold + (max_threshold - min_threshold) * threshold_type
            np.sort(thresholds)
            return thresholds
        else:
            raise ValueError("Invalid type for threshold_type. Must be 'equidistant' or a numpy array.")