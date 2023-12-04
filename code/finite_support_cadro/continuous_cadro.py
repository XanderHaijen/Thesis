import numpy as np
from ellipsoids import Ellipsoid
from robust_optimization import RobustOptimization
import cvxpy as cp
import study_minimal


class ContinuousCADRO:
    def __init__(self, data: np.ndarray, ellipse: Ellipsoid, solver=cp.MOSEK, split=None, seed=0):
        self.data = data  # data is a (d, m) matrix, where d is the dimension and m is the number of samples
        if split is None:
            self._set_data_split(data.shape[1])
        else:
            self.split = split
        self.ellipsoid = ellipse
        self.solver = solver
        self.generator = np.random.default_rng(seed)

    def _set_data_split(self, m: int, mu=0.01, nu=0.8) -> int:
        split = int(np.floor(nu * mu * (m * (m + 1)) / (mu * m + nu)))
        self.split = 2 if split < 2 else split


class CADRO1DLinearRegression(ContinuousCADRO):
    def __init__(self, data: np.ndarray, ellipse: Ellipsoid, split=None):
        super().__init__(data, ellipse, split)

        # attributes to be defined later
        self.lambda_ = None
        self.theta_0 = None
        self.theta_r = None
        self.theta = None
        self.alpha = None
        self.tau = None
        self.gamma = None

    def set_theta_0(self, theta_0: np.ndarray = None, nb_theta_0: int = 1):
        if theta_0 is None:
            train_data_indices = np.arange(0, self.split)
            self.find_theta0(train_data_indices, nb_theta_0)
        else:
            assert len(theta_0) == nb_theta_0
            self.theta_0 = theta_0

    def find_theta0(self, data_batch_indices: np.ndarray, nb_theta_0: int = 1):
        # randomly divide the data into nb_theta_0 batches
        self.generator.shuffle(data_batch_indices)
        data_batches = np.array_split(data_batch_indices, nb_theta_0)
        theta_0s = np.zeros(nb_theta_0)
        for i, batch in enumerate(data_batches):
            theta_0s[i] = self._find_theta0_batch(batch)
        self.theta_0 = theta_0s

    def _find_theta0_batch(self, data_batch_indices: np.ndarray):
        # solve the least squares problem (i.e. the sample average approach)
        theta0 = cp.Variable(1)
        train_data = self.data[:, data_batch_indices]
        objective = cp.Minimize(self.least_squares_loss(theta0, train_data[0, :], train_data[1, :], cvxpy=True))
        problem = cp.Problem(objective)
        problem.solve(solver=self.solver)
        return theta0.value[0]

    def find_theta_r(self):
        robust_opt = RobustOptimization(self.ellipsoid, solver=self.solver)
        robust_opt.solve_1d_linear_regression()
        self.theta_r = robust_opt.theta


    def calibrate(self, confidence_level):
        self.alpha = np.zeros(len(self.theta_0))
        for i in range(len(self.theta_0)):
            self.calibrate_index(index=i, confidence_level=confidence_level / len(self.theta_0))

    def calibrate_index(self, index: int = 0, asym_cutoff: int = 80, confidence_level: float = 0.05):
        m_cal = self.data.shape[1] - self.split
        method = "asymptotic" if m_cal > asym_cutoff else "brentq"
        calibration = study_minimal.calibrate(length=m_cal, method=method, level=confidence_level,
                                              full_output=True)
        gamma = calibration.info['radius']
        kappa = int(np.ceil(m_cal * gamma))
        eta = np.array([self.scalar_loss(self.theta_0[index], self.data[0, self.split + i],
                                         self.data[1, self.split + i]) for i in range(m_cal)])
        eta.sort(axis=0)
        eta_bar = self._eta_bar()
        alpha = (kappa / m_cal - gamma) * eta[kappa - 1] + np.sum(eta[kappa:m_cal]) / m_cal + eta_bar * gamma
        self.alpha[index] = alpha

    def _eta_bar(self):
        B, b, beta = self.loss_matrices(cvxpy=True)
        _tau = cp.Variable(1)
        _lambda = cp.Variable(1)
        M11 = - B - _lambda * self.ellipsoid.A
        M12 = - b - _lambda * self.ellipsoid.a
        M22 = - beta - _lambda * self.ellipsoid.c + _tau
        M22 = cp.reshape(M22, (1, 1))
        M = cp.bmat([[M11, M12], [M12.T, M22]])
        constraints = [M >> 0, _lambda >= 0]
        objective = cp.Minimize(_tau)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)

        if problem.status == cp.INFEASIBLE:
            raise ValueError("The problem is infeasible")
        elif problem.status == cp.OPTIMAL_INACCURATE:
            print("The problem is solved but the solution is inaccurate")

        return _tau.value[0]

    def set_theta_star(self, theta=None):
        if theta is not None:
            self.theta = theta
        else:
            self.theta = self.find_theta_star()

    def find_theta_star(self):
        theta_star = cp.Variable(1)
        self.lambda_ = cp.Variable(1)
        self.tau = cp.Variable(1)
        self.gamma = cp.Variable(1)

        objective = cp.Minimize(self.alpha * self.lambda_ + self.tau)
        constraints = [self.lambda_ >= 0, self.lmi_constraint()]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)

        ...  # TODO continue here

    def lmi_constraint(self, index: int = 0):
        theta0 = self.theta_0[index]
        B_0 = np.array([[theta0 ** 2, -theta0], [-theta0, 1]])
        # b_0 is column vector of size 2
        b_0 = np.zeros((2, 1))
        beta_0 = 0
        # constructing M_11
        a = cp.reshape(self.ellipsoid.a, (2, 1))
        M_111 = self.lambda_ * B_0 - self.gamma * self.ellipsoid.A
        M_111 = cp.reshape(M_111, (2, 2))
        M_112 = self.lambda_ * b_0 - self.gamma * self.ellipsoid.a
        M_112 = cp.reshape(M_112, (2, 1))
        M_113 = self.tau + self.lambda_ * beta_0 - self.gamma * self.ellipsoid.c
        M_113 = cp.reshape(M_113, (1, 1))
        M_11 = cp.bmat([[M_111, M_112], [cp.transpose(M_112), M_113]])
        # constructing M_12 and M_21
        M_12 = cp.vstack([self.theta, -1, 0])
        # constructing M_22
        M_22 = 1
        M_22 = np.reshape(M_22, (1, 1))
        # combine into M
        M = cp.bmat([[M_11, M_12], [cp.transpose(M_12), M_22]])
        # construct the constraints
        return [M >> 0]
    @staticmethod
    def least_squares_loss(theta, x, y, cvxpy=False):
        if cvxpy:
            return cp.sum_squares(y - cp.multiply(theta, x))
        else:
            return np.sum((y - theta * x) ** 2)

    @staticmethod
    def scalar_loss(theta, x, y, cvxpy=False):
        if cvxpy:
            return cp.power(y - cp.multiply(theta, x), 2)
        else:
            return (y - theta * x) ** 2

    def loss_matrices(self, cvxpy=False):
        if not cvxpy and isinstance(self.theta, cp.Variable):
            raise ValueError("theta must be a float when cvxpy=False")

        if not cvxpy:
            B = np.array([[self.theta ** 2, -self.theta], [-self.theta, 1]])
            b = np.zeros((2, 1))
            beta = 0
            return B, b, beta
        else:
            B = cp.vstack([cp.hstack([self.theta ** 2, -self.theta]), cp.hstack([-self.theta, 1])])
            b = cp.vstack([0, 0])
            beta = 0
            return B, b, beta
