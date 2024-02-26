from abc import abstractmethod
import numpy as np
import study_minimal
from ellipsoids import Ellipsoid
import cvxpy as cp
import warnings

class ContinuousCADRO:
    def __init__(self, data: np.ndarray, ellipse: Ellipsoid, solver=cp.MOSEK, split=None,
                 seed=0, suppress_userwarning=True):
        self.data = data  # data is a (d, m) matrix, where d is the dimension and m is the number of samples
        if split is None:
            self._set_data_split(data.shape[1])
        else:
            self.split = split
        self.ellipsoid = ellipse
        self.solver = solver
        self.generator = np.random.default_rng(seed)
        if suppress_userwarning:
            warnings.filterwarnings("ignore", category=UserWarning)

        self.theta_0 = None
        self.alpha = None
        self.lambda_ = None
        self.tau = None
        self.gamma = None
        self.theta = None
        self.theta_r = None

    @property
    def results(self):
        return {"theta_0": self.theta_0, "alpha": self.alpha, "lambda": self.lambda_, "tau": self.tau,
                "gamma": self.gamma, "theta": self.theta, "theta_r": self.theta_r, "loss": self.loss,
                "loss_r": self.loss_r, "loss_0": self.loss_0, "objective": self.objective}

    @property
    @abstractmethod
    def loss(self) -> float:
        pass

    @property
    @abstractmethod
    def loss_r(self) -> float:
        pass

    @property
    @abstractmethod
    def loss_0(self, index: int = 0) -> float:
        pass

    @property
    def objective(self):
        return np.sum(self.alpha * self.lambda_) + self.tau

    def _set_data_split(self, m: int, mu=0.01, nu=0.8):
        split = int(np.floor(nu * mu * (m * (m + 1)) / (mu * m + nu)))
        self.split = 2 if split < 2 else split

    def print_results(self, include_robust=False):
        print("theta_0 = ", self.theta_0)
        print("alpha = ", self.alpha)
        print("lambda = ", self.lambda_)
        print("tau = ", self.tau)
        print("gamma = ", self.gamma)
        print("theta = ", self.theta)
        if include_robust:
            if self.theta_r is None:
                self.set_theta_r()
            print("theta_r = ", self.theta_r)
        print("loss = ", self.loss)
        if include_robust:
            print("loss_r = ", self.loss_r)
        print("loss_0 = ", self.loss_0)
        print("objective = ", self.objective)

    def solve(self, theta0: list = None, theta: float = None, confidence_level: float = 0.05,
              nb_theta_0: int = 1):
        # sanity checks
        if theta0 is not None and len(theta0) != nb_theta_0:
            raise ValueError("theta0 must be an array of length nb_theta_0")
        # step 1: find theta_0
        self.set_theta_0(theta_0=theta0, nb_theta_0=nb_theta_0)
        # Step 2: calibrate ambiguity set
        self.calibrate(confidence_level=confidence_level)
        # Step 3: solve the CADRO problem
        self._set_theta_star(theta=theta)
        return self.results

    def set_theta_0(self, theta_0: np.ndarray = None, nb_theta_0: int = 1):
        if theta_0 is None:
            train_data_indices = np.arange(0, self.split)
            self._find_theta0(train_data_indices, nb_theta_0)
        else:
            assert len(theta_0) == nb_theta_0
            self.theta_0 = theta_0

    def calibrate(self, confidence_level):
        if self.theta_0 is None:
            raise ValueError("theta_0 is not set")

        self.alpha = [0] * len(self.theta_0)
        for i in range(len(self.theta_0)):
            self._calibrate_index(index=i, confidence_level=confidence_level / len(self.theta_0))

    def _find_theta0(self, data_batch_indices: np.ndarray, nb_theta_0: int = 1):
        # randomly divide the data into nb_theta_0 batches
        self.generator.shuffle(data_batch_indices)
        data_batches = np.array_split(data_batch_indices, nb_theta_0)
        theta_0s = [0] * nb_theta_0
        for i, batch in enumerate(data_batches):
            theta_0s[i] = self._find_theta0_batch(batch)
        self.theta_0 = theta_0s

    def _set_theta_star(self, theta=None):
        self._find_theta_star(theta)

        return self.theta

    def _calibrate_index(self, index: int = 0, asym_cutoff: int = 80, confidence_level: float = 0.05):
        m_cal = self.data.shape[1] - self.split
        method = "asymptotic" if m_cal > asym_cutoff else "brentq"
        calibration = study_minimal.calibrate(length=m_cal, method=method, level=confidence_level,
                                              full_output=True)
        gamma = calibration.info['radius']
        kappa = int(np.ceil(m_cal * gamma))
        eta = np.array([self._scalar_loss(self.theta_0[index], self.data[0, self.split + i],
                                          self.data[1, self.split + i]) for i in range(m_cal)])
        eta.sort(axis=0)
        eta_bar = self._eta_bar(index=index)
        alpha = (kappa / m_cal - gamma) * eta[kappa - 1] + np.sum(eta[kappa:m_cal]) / m_cal + eta_bar * gamma
        self.alpha[index] = alpha

    def _eta_bar(self, index: int = 0):
        B, b, beta = self._loss_matrices(theta=self.theta_0[index], cvxpy=True)
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
            raise ValueError("eta_bar is infeasible for index " + str(index))
        return _tau.value[0]

    def reset(self):
        """
        Resets the CADRO problem to its initial state. Only the data, the ellipsoid and the robust solution are kept.
        The split is also kept.
        """
        self.theta_0 = None
        self.alpha = None
        self.lambda_ = None
        self.tau = None
        self.gamma = None
        self.theta = None

    @abstractmethod
    def test_loss(self, test_data: np.ndarray) -> float:
        pass

    @abstractmethod
    def set_theta_r(self):
        pass

    @abstractmethod
    def _find_theta0_batch(self, data_batch_indices: np.ndarray):
        pass

    @abstractmethod
    def _find_theta_star(self, theta=None):
        pass

    @abstractmethod
    def _lmi_constraint(self, index: int = 0) -> list:
        pass

    @staticmethod
    @abstractmethod
    def _loss_function(theta, data, cvxpy=False):
        pass

    @staticmethod
    @abstractmethod
    def _scalar_loss(theta, x, y, cvxpy=False) -> float:
        pass

    @abstractmethod
    def _loss_matrices(self, theta, cvxpy):
        pass

