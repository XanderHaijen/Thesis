from typing import Union
from robust_optimization import RobustOptimization
from continuous_cadro import ContinuousCADRO
import numpy as np
from ellipsoids import Ellipsoid
import cvxpy as cp


class LeastSquaresCadro(ContinuousCADRO):
    """
    CADRO for least squares objective function:
    J(theta) = || A theta - b ||_2^2
    """

    def __init__(self, data: np.ndarray, ellipsoid: Ellipsoid, solver=cp.MOSEK, split=None, seed=0):
        """
        :param data: (d, m) matrix containing the data
        :param ellipsoid: Ellipsoid object
        :param split: split the data into two parts. If None, the default split is used.
        """
        super().__init__(data, ellipsoid, solver, split, seed)

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
        elif index is None:
            # TODO implement taking mean
            pass
        else:
            return self._loss_function(self.theta_0[index], self.data)

    def set_theta_r(self):
        """
        Set theta_r to be the solution of the robust optimization problem.
        """
        robust_opt = RobustOptimization(self.ellipsoid, solver=self.solver)
        robust_opt.solve_least_squares()
        self.theta_r = np.reshape(robust_opt.theta, (-1, 1))

    def _find_theta0_batch(self, data_batch_indices: np.ndarray, m_max: int = 5000):
        """
        Find theta_0 for the given batch of data indices. Use the sample average approximation (SAA) method.
        :param data_batch_indices: indices of the data to use for the SAA method
        :return: theta_0
        """
        theta0 = cp.Variable((self.data.shape[0] - 1, 1))
        train_data = self.data[:, data_batch_indices]
        if len(data_batch_indices) > m_max:  # to avoid solver issues
            train_data = train_data[:, :m_max]
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
            x = np.reshape(x, (x.shape[0], 1))
            loss = (np.dot(theta.T, x) - y) ** 2
            return loss[0, 0]  # reshape to scalar
        else:
            return cp.square(cp.matmul(cp.transpose(theta), x) - y)

    def _find_theta_star(self, theta=None):
        """
        Solve the multivariate least squares problem using the CADRO method.

        :param theta: the value for theta. If None, theta is an optimization variable. Default is None.

        :return: None. All values are stored in the object.
        """
        self.theta = cp.Variable(shape=(self.data.shape[0] - 1, 1)) if theta is None else theta
        if theta is not None:
            assert theta.shape == (self.data.shape[0] - 1, 1)
        self.lambda_ = cp.Variable(len(self.theta_0))
        self.tau = cp.Variable(1)
        self.gamma = cp.Variable(len(self.theta_0))

        objective = cp.Minimize(self.alpha * self.lambda_ + self.tau)
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
            self.theta = self.theta.value[:, 0]

    def _lmi_constraint(self, index: int = 0) -> list:
        """
        Returns the LMI constraint for the given index in cvxpy format.
        TODO update the constraint to use sum instead of many constraints
        :param index: index of the theta_0 to use
        :return: a list containing the LMI constraint
        """
        theta_i = self.theta_0[index]
        lambda_i = self.lambda_[index]
        gamma_i = self.gamma[index]
        B_i, _, _ = self._loss_matrices(theta_i, cvxpy=True)
        ext = cp.vstack([-1, 0])
        theta_vector = cp.vstack([self.theta, ext])
        A_11 = lambda_i * B_i - gamma_i * self.ellipsoid.A
        A_12 = gamma_i * self.ellipsoid.a
        A_22 = self.tau - gamma_i * self.ellipsoid.c
        A_22 = cp.reshape(A_22, (1, 1))
        A = cp.bmat([[A_11, A_12], [cp.transpose(A_12), A_22]])
        M = cp.bmat([[A, theta_vector], [theta_vector.T, cp.reshape(1, (1, 1))]])
        return [M >> 0]

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
