from typing import Union
from robust_optimization import RobustOptimization
from continuous_cadro import ContinuousCADRO
import numpy as np
from ellipsoids import Ellipsoid
import cvxpy as cp
import study_minimal


class LeastSquaresCadro(ContinuousCADRO):
    """
    CADRO for least squares objective functions:
    J(theta) = || H theta - h ||_2^2
    """

    def __init__(self, data: np.ndarray, ellipsoid: Ellipsoid, split=None):
        """
        :param data: (d, m) matrix containing the data
        :param ellipsoid: Ellipsoid object
        :param split: split the data into two parts. If None, the default split is used.
        """
        super().__init__(data, ellipsoid, split)

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
            raise ValueError("theta_r has not been set yet.")
        else:
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
        elif index is None:
            losses = np.array([self._loss_function(theta, self.data) for theta in self.theta_0])
            return np.mean(losses)
        else:
            return self._loss_function(self.theta_0[index], self.data)

    def set_theta_r(self):
        """
        Set theta_r to be the solution of the robust optimization problem.
        """
        robust_opt = RobustOptimization(self.ellipsoid, solver=self.solver)
        robust_opt.solve_least_squares()
        self.theta_r = robust_opt.theta

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

    def _calibrate_index(self, index: int, asym_cutoff: int = 80, confidence_level: float = 0.05) -> None:
        """
        Calibrate the index-th alpha value to the given confidence level.
        :param index: index of theta_0 to calibrate
        :param confidence_level: confidence level
        :return: None. alpha is modified in place.
        """
        m_cal = self.data.shape[1] - self.split
        method = "asymptotic" if m_cal > asym_cutoff else "brentq"
        calibration = study_minimal.calibrate(length=m_cal, method=method, confidence_level=confidence_level,
                                              full_output=True)
        gamma = calibration.info["gamma"]
        kappa = int(np.ceil(m_cal * gamma))
        # TODO: check if this is correct: should be use the calibration set or the training set to compute eta?
        eta = np.array([self._loss_function(self.theta_0[index], self.data[:, self.split + i]) for i in range(m_cal)])
        eta.sort(axis=0)
        eta_bar = self._eta_bar(index=index)
        alpha = (kappa / m_cal - gamma) * eta[kappa - 1] + np.sum(eta[kappa:m_cal]) / m_cal + eta_bar * gamma
        self.alpha[index] = alpha

    @staticmethod
    def _loss_function(theta, data, cvxpy=False):
        """
        The loss function for the least squares problem, which uses kronecker products and the vec operator.
        :param theta: (d, 1) vector
        :param data: (d, m) matrix
        :param cvxpy: if True, return a cvxpy expression
        """
        if cvxpy:
            # TODO: check dimensions
            Theta = cp.kron(cp.hstack([theta.T, -np.ones([1, 1])]), np.eye(data.shape[0]))
            return cp.sum([cp.norm(Theta @ data[:, i]) ** 2 for i in range(data.shape[1])])
        else:
            Theta = np.kron(np.hstack([theta, -1]), np.eye(data.shape[1]))
            return np.sum([np.linalg.norm(Theta @ data[:, i]) ** 2 for i in range(data.shape[1])])

    def _eta_bar(self, index: int = 0):
        Theta_0 = self._loss_matrix(self.theta_0[index])
        Theta_0_ext = np.vstack([Theta_0, np.zeros((1, Theta_0.shape[1]))])
        A, a, c = self.ellipsoid.A, self.ellipsoid.a, self.ellipsoid.c
        gamma_ = cp.Variable(1)
        tau_ = cp.Variable(1)
        M1 = cp.bmat([[-gamma_ * A, -gamma_ * a], [-gamma_ * a.T, tau_ + -gamma_ * c]])
        M2 = Theta_0_ext.T @ Theta_0_ext
        constraints = [gamma_ >= 0, M2 - M1 >> 0]
        objective = cp.Minimize(tau_)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)
        if problem.status == cp.INFEASIBLE:
            raise ValueError("eta_bar is infeasible for index " + str(index))
        return tau_.value[0]

    def _find_theta_star(self, theta=None):
        self.theta = cp.Variable(self.data.shape[0]) if theta is None else theta
        if theta is not None:
            assert theta.shape == (self.data.shape[0], 1)
        self.lambda_ = cp.Variable(len(self.theta_0))
        self.tau = cp.Variable(1)
        self.gamma = cp.Variable(len(self.theta_0))

        objective = cp.Minimize(self.tau + cp.sum(self.lambda_ * self.alpha))
        constraints = [lambda_ >= 0 for lambda_ in self.lambda_] + \
                      [gamma >= 0 for gamma in self.gamma]

        for i in range(len(self.theta_0)):
            constraints += self._lmi_constraint(i)

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
            self.theta = self.theta.value

    def _lmi_constraint(self, index) -> list:
        gamma_i = self.gamma[index]
        lambda_i = self.lambda_[index]
        theta_i = self.theta_0[index]

        Theta_i = self._loss_matrix(theta_i, cvxpy=True)
        Theta = self._loss_matrix(self.theta, cvxpy=True)

        # construct A_bar
        A11 = lambda_i * Theta_i.T @ Theta_i - gamma_i * self.ellipsoid.A
        A12 = - gamma_i * self.ellipsoid.a
        A22 = self.tau - gamma_i * self.ellipsoid.c
        A_bar = cp.bmat([[A11, A12], [A12.T, A22]])

        # extend Theta_i with a zero row
        Theta_i_ext = cp.vstack([Theta_i, np.zeros((1, Theta_i.shape[1]))])

        # construct the LMI constraint
        M = cp.bmat([[A_bar, Theta_i_ext], [Theta_i_ext.T, np.eye(Theta_i.shape[1])]])
        return [M >> 0]

    def test_loss(self, test_data: np.ndarray) -> float:
        """
        Compute the loss on the test data.
        :param test_data: (d, m) matrix containing the test data
        :return: the loss on the test data
        """
        return self._loss_function(self.theta, test_data)

    def _loss_matrix(self,theta, cvxpy=False) -> Union[np.ndarray, cp.kron]:
        """
        Return the loss matrix Theta used in the optimization problem.
        """
        if cvxpy:
            Theta = cp.kron(cp.hstack([theta, -1]), np.eye(self.data.shape[1]))
            return Theta
        else:
            Theta = np.kron(np.hstack([theta, -1]), np.eye(self.data.shape[1]))
            return Theta
