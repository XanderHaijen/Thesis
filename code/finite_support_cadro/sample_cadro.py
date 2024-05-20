from abc import abstractmethod
from typing import Callable, Union
import numpy as np
import study_minimal
import cvxpy as cp
import warnings


class SampleCadro:
    """
    Continuous CADRO for non-quadratic loss functions and/or non-ellipsoidal support sets. The problem is approximated
    by a sample-based approach.
    """

    def __init__(self, data: np.ndarray, samples: np.ndarray,
                 loss_function: Callable[[Union[np.ndarray, cp.Variable], np.ndarray], Union[float, cp.Expression]],
                 variable_dimension: int,
                 regularization: Callable[[Union[np.ndarray, cp.Variable]], Union[float, cp.Expression]] = None,
                 solver=cp.MOSEK, split=None,
                 seed=0, suppress_userwarning=True):
        """
        :param data: (d, m) matrix containing the data (training data and calibration data)
        :param samples: (d, m) matrix containing the samples (used for the sample-based constraints)
        :param loss_function: loss function to use for the optimization problem. Takes two arguments: theta and data
        and returns a float. It should calculate the loss for one data point.
        :param variable_dimension: dimension of the variable theta. Theta should be a (dim x 1) vector.
        :param regularization: regularization function to use for the optimization problem. Takes one argument: theta
        :param solver: solver to use for the optimization problem (cp.solver type). Default is cp.MOSEK.
        :param split: split the data into two parts. If None, the default split is used.
        :param seed: seed for the random number generator
        :param suppress_userwarning: if True, suppress the UserWarning. Default is True.
        """
        # check if data or samples contain nan values
        if np.isnan(data).any() or np.isnan(samples).any():
            raise ValueError("The data or samples contain nan values.")

        self.data = data  # data is a (d x m) matrix, where d is the dimension and m the number of data points
        if data.shape[0] != samples.shape[0]:
            raise ValueError("The dimensions of the data and the samples must match.")
        self.samples = samples  # samples is a (d x n) matrix, where d is the dimension and n the number of samples
        self.solver = solver

        if split is None:
            self.split = self._set_data_split(data.shape[1])
        else:
            self.split = split

        self.generator = np.random.default_rng(seed)
        if suppress_userwarning:
            warnings.filterwarnings("ignore", category=UserWarning)

        self.loss_function = loss_function
        if regularization is None:
            self.regularization = lambda theta: 0
        else:
            self.regularization = regularization
        self.var_dim = variable_dimension

        self.theta = None
        self.theta_0 = None
        self.theta_r = None
        self.lambda_ = None
        self.alpha = None
        self.tau = None

        self.__robust_cost = None

    @property
    def results(self):
        """
        Return the results of the CADRO problem. If the problem has not yet been solved, the results are not available.

        :return: the results of the CADRO problem as a dictionary with the following keys:
        - theta_0: initial theta_0 values
        - alpha: alpha (calibrated ambiguity set) values
        - lambda: lambda values (Lagrangian multipliers)
        - tau: tau value (epigraph variable)
        - gamma: gamma value (S-lemma variable)
        - theta: solution of the CADRO problem
        - theta_r: robust solution
        - loss: loss function at theta
        - loss_r: loss function at theta_r
        - loss_0: loss function at theta_0
        - objective: value of the objective function
        """
        return {"theta_0": self.theta_0, "alpha": self.alpha, "lambda": self.lambda_, "tau": self.tau,
                "theta": self.theta, "theta_r": self.theta_r, "loss": self.loss,
                "loss_r": self.loss_r, "loss_0": self.loss_0, "objective": self.objective}

    @property
    def loss(self) -> float:
        """
        :return: the loss function at theta
        """
        if self.theta is None:
            raise ValueError("theta has not been set yet.")
        else:
            # return the mean over all data points
            return np.mean([self.loss_function(self.theta, data_point) + self.regularization(self.theta)
                            for data_point in self.data.T])

    @property
    def loss_r(self) -> float:
        """
        :return: the loss function at theta_r
        """
        if self.theta_r is None:
            self._set_theta_r()

        return np.mean([self.loss_function(self.theta_r, data_point) + self.regularization(self.theta_r)
                        for data_point in self.data.T])

    @property
    def loss_0(self) -> float:
        """
        :return: the loss function at theta_0
        """
        if self.theta_0 is None:
            raise ValueError("theta_0 has not been set yet.")
        else:
            return np.mean([self.loss_function(self.theta_0, data_point) + self.regularization(self.theta_0)
                            for data_point in self.data.T])

    @property
    def objective(self) -> float:
        """
        Return the value of the objective function. The objective function is defined as
        Sum(alpha * lambda) + tau.

        """
        return np.sum(self.alpha * self.lambda_) + self.tau + self.regularization(self.theta)

    @staticmethod
    def _set_data_split(m: int, mu=0.01, nu=0.8):
        """
        Set the split for the data. The split is calculated using the formula
        split = floor(nu * mu * m * (m + 1) / (mu * m + nu))
        where m is the number of samples, mu and nu are parameters.
        The training data is the part data[:, :split] and the calibration data is the part data[:, split:].

        :param m: number of samples
        :param mu: parameter for the split calculation. Default is 0.01.
        :param nu: parameter for the split calculation. Default is 0.8.

        :return: None. The split is stored in the object.
        """
        split = int(np.floor(nu * mu * (m * (m + 1)) / (mu * m + nu)))
        return 2 if split < 2 else split

    @property
    def robust_cost(self):
        if self.__robust_cost is None:
            self._set_theta_r()

        return self.__robust_cost

    def print_results(self, include_robust=False):
        """
        Print the resulting values for the CADRO problem. If the problem hasn't been solved yet, the results are not
        available.
        :param include_robust: if True, print the robust solution as well. Default is False.
        :return: None. The results are printed to the console
        """
        print("theta_0 = ", self.theta_0)
        print("alpha = ", self.alpha)
        print("lambda = ", self.lambda_)
        print("tau = ", self.tau)
        print("theta = ", self.theta)
        if include_robust:
            if self.theta_r is None:
                self._set_theta_r()
            print("theta_r = ", self.theta_r)
        print("loss = ", self.loss)
        if include_robust:
            print("loss_r = ", self.loss_r)
        print("loss_0 = ", self.loss_0)
        print("objective = ", self.objective)

    def solve(self, confidence_level: float = 0.05):
        """
        Solve the CADRO problem. The method first finds the initial theta_0 values, then calibrates the ambiguity set,
        and finally solves the actual CADRO problem. It uses a discretized version of the actual constraint.

        :param confidence_level: confidence level for the calibration. Default is 0.05.

        :return: the results of the CADRO algorithm as specified in the 'results' property. The results are also saved
        in the object itself.
        """
        # step 1: calculate the initial guess
        self._set_theta_0()
        # step 2: calibrate the ambiguity set
        self._calibrate(confidence_level=confidence_level)
        # step 3: solve the CADRO problem
        self._set_theta_star()

        # to return the results, we first have to determine theta_r
        self._set_theta_r()
        # return the results
        return self.results

    def _set_theta_0(self):
        """
        Set the initial theta_0 values. The initial values are calculated using the sample average approach.

        :return: None. The initial theta_0 values are stored in the object.
        """
        training_data = self.data[:, :self.split]
        theta0 = cp.Variable(self.var_dim, 'theta0')
        objective = cp.sum([self.loss_function(theta0, data_point) for data_point in training_data.T]) / self.split + \
                    self.regularization(theta0)
        objective = cp.Minimize(objective)
        problem = cp.Problem(objective)
        problem.solve(solver=self.solver)
        self.theta_0 = theta0.value

    def _calibrate(self, confidence_level, asym_cutoff=80) -> None:
        """
        Calibrate the ambiguity set. The method calculates the value for alpha using the calibration data.

        :param confidence_level: confidence level for the calibration
        :param asym_cutoff: number of samples for which the asymptotic method is used. Default is 80.

        :return: None. The value for alpha is stored in the object.
        """
        if self.theta_0 is None:
            raise ValueError("theta_0 has not been set yet.")

        # calibration
        m_cal = self.data.shape[1] - self.split
        method = "asymptotic" if m_cal > asym_cutoff else "brentq"
        calibration = study_minimal.calibrate(length=m_cal, method=method, level=confidence_level,
                                              full_output=True)
        gamma = calibration.info["radius"]

        # parameters
        kappa = int(np.ceil(m_cal * gamma))
        eta_bar = self._eta_bar()
        eta = np.array([self.loss_function(self.theta_0, data_point) for data_point in self.data[:, self.split:].T])
        eta.sort(axis=0)

        # alpha
        self.alpha = (kappa / m_cal - gamma) * eta[kappa - 1] + np.sum(eta[kappa:m_cal]) / m_cal + eta_bar * gamma

    def _set_theta_star(self):
        """
        Solve the CADRO optimization problem. The method uses the sample-based constraints to approximate the problem.

        :return: None. The solution is stored in the object.
        """
        d = self.data.shape[0]
        n = self.samples.shape[1]  # number of samples

        # variables
        self.theta = cp.Variable((self.var_dim, 1), 'theta')
        self.lambda_ = cp.Variable((), 'lambda')
        self.tau = cp.Variable((), 'tau')

        # objective function
        objective = cp.Minimize(self.alpha * self.lambda_ + self.tau + self.regularization(self.theta))

        # constraints
        constraints = [self.lambda_ >= 0]
        for i in range(n):
            constraints += self._sample_constraint(i)

        # construct and solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)

        if problem.status == cp.INFEASIBLE:
            raise ValueError("The problem is infeasible.")
        elif problem.status == cp.OPTIMAL_INACCURATE:
            print("The problem is solved but the solution is inaccurate.")

        self.lambda_ = self.lambda_.value
        self.tau = self.tau.value
        self.theta = self.theta.value[:, 0]

    def _set_theta_r(self):
        """
        Set the robust solution theta_r. The robust solution is calculated using the robust optimization problem.

        :return: None. The robust solution and cost are stored in the object.
        """
        theta_r = cp.Variable((self.var_dim, 1), 'theta_r')
        # construct cp array with all losses
        losses = [self.loss_function(theta_r, sample) + self.regularization(theta_r) for sample in self.samples.T]
        objective = cp.Minimize(cp.maximum(*losses))
        problem = cp.Problem(objective)
        problem.solve(solver=self.solver)
        self.theta_r = theta_r.value[:, 0]
        self.__robust_cost = problem.value

    def test_loss(self, test_data, theta: Union[str, np.ndarray] = "theta") -> float:
        """
        Compute the loss on the test data.

        :param test_data: (d, m) matrix containing the test data
        :param theta: the value for theta. If 'theta', the value of theta is used. Default is 'theta'.

        :return: the loss on the test data
        """
        if isinstance(theta, str):
            if theta == 'theta':
                theta = self.theta
            elif theta == "theta_0":
                theta = self.theta_0
            elif theta == "theta_r":
                theta = self.theta_r
            else:
                raise ValueError("The value for theta is incorrect. Should be np.ndarray or "
                                 "'theta', 'theta_0', 'theta_r'.")
        elif isinstance(theta, np.ndarray):
            if theta.shape != (self.var_dim, 1):
                raise ValueError("The shape of theta is incorrect.")

        return np.mean([self.loss_function(theta, data_point) for data_point in test_data.T])

    def _sample_constraint(self, index: int):
        """
        Construct the sample-based constraint for the CADRO problem.

        :param index: index of the sample to use

        :return: the constraint as a list
        """
        sample = self.samples[:, index]
        constraint = [self.loss_function(self.theta, sample) -
                      self.lambda_ * self.loss_function(self.theta_0, sample) <= self.tau]
        return constraint

    def _eta_bar(self) -> float:
        """
        Calculate the value for eta_bar. The value is calculated using the samples, and is equal to the
        sample maximum of the loss function.
        """
        return np.max([self.loss_function(self.theta_0, sample) for sample in self.samples.T])

    def reset(self):
        """
        Resets the CADRO problem to its initial state. Only the data, the ellipsoid and the robust solution are kept.
        The split is also kept.
        """
        self.theta_0 = None
        self.alpha = None
        self.lambda_ = None
        self.tau = None
        self.theta = None
