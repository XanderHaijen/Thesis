from typing import Union
import numpy as np
import cvxpy as cp

from ellipsoids import Ellipsoid
from robust_optimization import RobustOptimization
from continuous_cadro import ContinuousCADRO



class CADRO1DLinearRegression(ContinuousCADRO):
    def __init__(self, data: np.ndarray, ellipse: Ellipsoid, split=None):
        super().__init__(data, ellipse, split)

    @property
    def loss(self):
        if self.theta is None:
            raise ValueError("theta is not set")

        return self._loss_function(self.theta, self.data)

    @property
    def loss_r(self):
        if self.theta_r is None:
            return None

        return self._loss_function(self.theta_r, self.data)

    @property
    def loss_0(self, index: Union[int, None] = 0):
        if self.theta_0 is None:
            raise ValueError("theta_0 is not set")

        if index is None:
            losses = np.array([self._loss_function(theta_0, self.data) for theta_0 in self.theta_0])
            return np.mean(losses)
        else:
            return self._loss_function(self.theta_0[index], self.data)

    def set_theta_r(self):
        """
        Set theta_r to be the solution of the robust optimization problem.
        """
        robust_opt = RobustOptimization(self.ellipsoid, solver=self.solver)
        robust_opt.solve_1d_linear_regression()
        self.theta_r = robust_opt.theta

    def _find_theta0_batch(self, data_batch_indices: np.ndarray):
        """
        Solve the least squares problem using the sample average approach.

        :param data_batch_indices: indices of the data to use for the SAA method

        :return: theta_0 as a float
        """
        # solve the least squares problem (i.e. the sample average approach)
        theta0 = cp.Variable(1)
        train_data = self.data[:, data_batch_indices]
        objective = cp.Minimize(self._loss_function(theta0, train_data, cvxpy=True))
        problem = cp.Problem(objective)
        problem.solve(solver=self.solver)
        return theta0.value[0]

    def _find_theta_star(self, theta=None):
        """
        Solve the least squares problem using the CADRO method.

        :param theta: value for theta. If None, theta is an optimization variable. Default is None.

        :return: None. Values for theta, lambda, tau, and gamma are stored in the object.
        """
        self.theta = cp.Variable(1) if theta is None else theta
        self.lambda_ = cp.Variable(len(self.theta_0))
        self.tau = cp.Variable(1)
        self.gamma = cp.Variable(1)

        objective = cp.Minimize(self.alpha * self.lambda_ + self.tau)
        constraints = [lambda_ >= 0 for lambda_ in self.lambda_] + [self.gamma >= 0]

        constraints += self._lmi_constraint()

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)

        if problem.status == cp.INFEASIBLE:
            raise ValueError("The problem is infeasible")
        elif problem.status == cp.OPTIMAL_INACCURATE:
            print("The problem is solved but the solution is inaccurate")

        self.lambda_ = self.lambda_.value
        self.tau = self.tau.value[0]
        self.gamma = self.gamma.value
        if isinstance(self.theta, cp.Variable):
            self.theta = self.theta.value[0]

    def _lmi_constraint(self):
        """
        Construct the linear matrix inequality (LMI) constraints for the CADRO problem in cvxpy format.
        :return: a list containing the LMI constraint
        """
        B_0 = lambda i: np.array([[self.theta_0[i] ** 2, -self.theta_0[i]], [-self.theta_0[i], 1]])

        A_11 = cp.sum([self.lambda_[i] * B_0(i) - self.gamma * self.ellipsoid.A for i in range(len(self.theta_0))])
        # constructing M_11
        A_11 = cp.reshape(A_11, (2, 2))
        A_12 = self.gamma * self.ellipsoid.a  # + lambda_i * b_0
        A_12 = cp.reshape(A_12, (2, 1))
        A_22 = self.tau - self.gamma * self.ellipsoid.c  # + lambda_i * beta_0
        A_22 = cp.reshape(A_22, (1, 1))
        A = cp.bmat([[A_11, A_12], [cp.transpose(A_12), A_22]])
        # constructing M_12 and M_21
        M_12 = cp.vstack([self.theta, -1, 0])
        # constructing M_22
        M_22 = 1
        M_22 = np.reshape(M_22, (1, 1))
        # combine into M
        M = cp.bmat([[A, M_12], [cp.transpose(M_12), M_22]])
        # construct the constraints
        return [M >> 0]

    def test_loss(self, test_data: np.ndarray, type: str = "theta", index: int = 0) -> float:
        """
        Compute the loss on the test data. The loss is normalized by the number of data points and is given by
        1/m * sum_{i=1}^m (y_i - theta * x_i)^2.

        :param test_data: (d, m) matrix containing the test data
        :param type: type of theta to use. Must be either "theta", "theta_r", or "theta_0". Default is "theta".
        :param index: index of theta_0 to use. Default is 0. This argument is ignored if type is not "theta_0".

        :return: the loss on the test data as a float
        """
        if self.theta is None:
            raise ValueError("theta is not set")
        if type not in ("theta", "theta_r", "theta_0"):
            raise ValueError("type must be either theta, theta_r or theta_0")

        if type == "theta":
            return self._loss_function(self.theta, test_data) / test_data.shape[1]
        elif type == "theta_r":
            if self.theta_r is None:
                raise ValueError("theta_r is not set")
            return self._loss_function(self.theta_r, test_data) / test_data.shape[1]
        else:
            if self.theta_0 is None:
                raise ValueError("theta_0 is not set")
            return self._loss_function(self.theta_0[index], test_data) / test_data.shape[1]

    @staticmethod
    def _loss_function(theta, data, cvxpy=False):
        """
        Compute the loss function for the given data and theta.

        :param theta: the value of theta
        :param data: the data matrix (d x m)
        :param cvxpy: if True, the function returns a cvxpy expression. Default is False.

        :return: the loss function as a cvxpy expression or a float
        """
        x = data[0, :]
        y = data[1, :]
        if cvxpy:
            return cp.sum_squares(y - cp.multiply(theta, x))
        else:
            return np.sum((y - theta * x) ** 2)

    @staticmethod
    def _scalar_loss(theta, x, y, cvxpy=False):
        """
        Compute the loss function for the given single data point (x, y) and theta.

        :param theta: the value of theta
        :param x: the x value (scalar)
        :param y: the y value (scalar)
        :param cvxpy: if True, the function returns a cvxpy expression. Default is False.

        :return: the loss function as a cvxpy expression or a float
        """
        if cvxpy:
            return cp.power(y - cp.multiply(theta, x), 2)
        else:
            return (y - theta * x) ** 2

    def _loss_matrices(self, theta, cvxpy=False):
        """
        Compute the matrices B, b, and beta for the loss function.

        :param theta: the value of theta
        :param cvxpy: if True, the function returns cvxpy expressions. Default is False.

        :return: B, b, beta as numpy arrays or cvxpy expressions
        """
        if not cvxpy and isinstance(theta, cp.Variable):
            raise ValueError("theta must be a float when cvxpy=False")

        if not cvxpy:
            B = np.array([[theta ** 2, -theta], [-theta, 1]])
            b = np.zeros((2, 1))
            beta = 0
            return B, b, beta
        else:
            B = cp.vstack([cp.hstack([theta ** 2, -theta]), cp.hstack([-theta, 1])])
            b = cp.vstack([0, 0])
            beta = 0
            return B, b, beta
