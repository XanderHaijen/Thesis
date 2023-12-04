import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from ellipsoids import Ellipsoid


class RobustOptimization:
    def __init__(self, ellipsoid: Ellipsoid, solver=cp.MOSEK):
        """
        :param A: (n, n) matrix defining the ellipsoid
        :param a: (n, 1) vector defining the ellipsoid
        :param c: (1, 1) scalar defining the ellipsoid
        :param solver: solver to use for the optimization problem (cp.solver type). Default is cp.MOSEK.
        """
        self.__ellipsoid = ellipsoid
        self.theta = None
        self.cost = None
        self.tau = None
        self.lambda_ = None
        self.solver = solver

    @property
    def A(self):
        return self.__ellipsoid.A

    @property
    def a(self):
        return self.__ellipsoid.a

    @property
    def c(self):
        return self.__ellipsoid.c

    @property
    def ellipsoid(self):
        return self.__ellipsoid

    def solve_1d_linear_regression(self):
        """
            Solve the robust case for a quadratic loss function and an ellipsoidal ambiguity set in the 2D case.
            The ellipsoid is parameterized by A, a, c and given by the set
            {x | x^T A x + 2 a^T x + c => 0}.
            The loss is assumed to be quadratic for a linear regression problem, and is equal to
            l(xi, theta) = (xi_2 - theta * xi_1)^2.
            :return: the optimal theta, the optimal value, the optimal tau and the optimal lambda as a dictionary
        """
        if self.theta is not None: # if the problem has already been solved, return the results
            return {"theta": self.theta, "cost": self.cost, "tau": self.tau, "lambda": self.lambda_}
        else:
            self._solve_robust_1d_linreg()
            return {"theta": self.theta, "cost": self.cost, "tau": self.tau, "lambda": self.lambda_}

    def _solve_robust_1d_linreg(self):
        A = self.__ellipsoid.A
        a = self.__ellipsoid.a
        c = self.__ellipsoid.c

        assert A.shape == (2, 2)
        assert a.shape == (2, 1)
        assert c.shape == (1, 1)
        self.theta = cp.Variable()
        self.tau = cp.Variable()
        self.lambda_ = cp.Variable()
        lambda_positive = [self.lambda_ >= 0]
        A_bar = cp.bmat([[- self.lambda_ * A, - self.lambda_ * a],
                         [- self.lambda_ * a.T, - self.lambda_ * c + self.tau]])
        theta_vector = cp.vstack([self.theta, -1, 0])
        M = cp.bmat([[A_bar, theta_vector], [theta_vector.T, cp.reshape(1, (1, 1))]])
        constraints = [M >> 0] + lambda_positive
        objective = cp.Minimize(self.tau)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver)

        if problem.status == cp.INFEASIBLE:
            raise ValueError("Problem is infeasible")
        elif problem.status == cp.OPTIMAL_INACCURATE:
            print("Problem is solved but the solution is inaccurate")

        self.cost = problem.value
        self.theta = self.theta.value
        self.tau = self.tau.value
        self.lambda_ = self.lambda_.value
        return self.theta, self.cost, self.tau, self.lambda_


class RobustOptimizationTester:
    def __init__(self, ellipsoid, type="1d_linreg", seed=0):
        self.problem = RobustOptimization(ellipsoid)
        self.type = type
        self.generator = np.random.default_rng(seed=seed)

    def run(self):
        if self.type == "1d_linreg":
            return self._test_1d_linreg()
        else:
            raise ValueError("Unknown problem type")

    def _test_1d_linreg(self):
        result = self.problem.solve_1d_linear_regression()
        print("theta: ", result['theta'])
        print("value: ", result['cost'])
        print("tau: ", result['tau'])
        print("lambda: ", result['lambda'])
        self.problem.ellipsoid.plot(1)
        x = np.linspace(0, 1, 5)
        y = result["theta"] * x
        plt.plot(x, y, label="theta")
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    generator = np.random.default_rng(seed=0)
    x = generator.uniform(0, 1, 20)
    y = 3 * x + generator.normal(0, 1, 20)
    data = np.vstack((x, y))
    ellipsoid = Ellipsoid.lj_ellipsoid(data, 3, 1)
    tester = RobustOptimizationTester(ellipsoid)
    tester.run()


