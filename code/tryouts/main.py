import time
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar, LinearConstraint, NonlinearConstraint
from scipy.special import betainc
from scipy.stats import dirichlet
from calibration.study_minimal import calibrate




def squared_loss(y, y_hat):
    """
    The squared loss function
    :param y: the actual value (scalar)
    :param x: the input value (scalar)
    :param y_hat: the predicted value (scalar)
    """
    return (y - y_hat) ** 2


# consider three algorithms
# 1. sample average approximation (SAA)

def sample_average_approximation(x, y):
    """
    The sample average approximation algorithm
    :param x: the input values (n x 1)
    :param y: the actual values (n x 1)
    :return: the weights (1 x 1)
    """
    loss_func = lambda theta: np.mean(squared_loss(y, theta * x))
    res = minimize(loss_func, x0=np.array([0]))
    theta = res.x
    return theta


# 2. Ordered risk minimization (ORM)

def ordered_risk_minimization_robust(x, y):
    """
    The ordered risk minimization algorithm as explained in
    [Coppens and Patrinos, 2023]. We do not restrict mu to any subset
    of the ambiguity set. Therefore, the only constraints on mu are
    that it is a probability distribution, i.e. sum(mu) = 1 and mu >= 0.
    :param x: input values (n x 1)
    :param y: actual values (n x 1)
    :return: the weights (1 x 1)
    """

    def ordered_loss(theta, x, y):
        loss_vector = np.array([squared_loss(yi, theta * xi) for xi, yi in zip(x, y)])
        loss_vector = np.sort(loss_vector, axis=0)
        # reshape to shape (n,)
        loss_vector = loss_vector.reshape(n)
        L = np.zeros(n)
        L[:len(loss_vector)] = loss_vector
        return L

    # Step 1: matrices C and a
    n = x.shape[0]
    C = np.vstack((np.ones((1, n)), -np.ones((1, n)), -np.eye(n)))
    b = np.zeros(n + 2)
    b[0:2] = [1, -1]

    # Step 2: loss vector and cost function
    L = lambda theta: ordered_loss(theta, x, y)
    def fun(z):
        return L(z[0]) - np.dot(C.T, z[1:])

    # Step 3: Nonlinear constraint

    ub, lb = 0, 0
    nonlinear_constraint = NonlinearConstraint(fun, lb, ub)

    # Step 3.2: linear constraint
    # lambda >= 0
    A = np.hstack((np.zeros((n + 2, 1)), np.eye(n + 2)))
    linear_constraint = LinearConstraint(A, 0, np.inf)

    # Step 4: solve the problem
    cost = lambda x: np.inner(b, x[1:])
    res = minimize(cost, x0=np.zeros(n + 3), constraints=[nonlinear_constraint, linear_constraint])
    theta = res.x[0]
    return theta


def ordered_risk_minimization_total_variation(x, y, m, beta, delta):
    """
    The ordered risk minimization algorithm as explained in
    [Coppens and Patrinos, 2023]. We use the total variation divergence to define
    an ambiguity set inside the probability simplex.

    The problem is solved via the dual formulation and using strong duality.

    :param x: input values (n x 1)
    :param y: actual values (n x 1)
    :param m: the amount of samples to draw in calibration of alpha
    :param beta: the confidence level for the regularized incomplete beta function
    :return: the weights (1 x 1)
    """
    def ordered_loss(theta, x, y):
        loss_vector = np.array([squared_loss(yi, theta * xi) for xi, yi in zip(x, y)])
        loss_vector = np.sort(loss_vector, axis=0)
        # reshape to shape (n,)
        loss_vector = loss_vector.reshape(n)
        L = np.zeros(2 * n)
        L[:len(loss_vector)] = loss_vector
        return L

    # actual algorithm
    n = x.shape[0]


    ### Step 1: calibrate alpha ###
    alpha = calibrate_alpha(n=n, m=m, beta=beta, delta=delta)


    ### Step 2: solve the dual problem ###
    n = x.shape[0]
    # Step 2.1: construct the matrix C and the vector a
    # These come from the constraints of the primal problem
    C = np.zeros((3 * n + 3, 2 * n))
    b = np.zeros(3 * n + 3)
    # we have two groups of constraints. One comes from mu being a probability distribution, i.e. in the simplex
    # the other comes from the fact that mu must be in the parametrized ambiguity set
    # 1. mu being in the simplex
    # 1.1. sum(mu) >= 1
    C[0, :n] = 1
    b[0] = 1
    # 1.2 sum(mu) <= 1
    C[1, :n] = -1
    b[1] = -1
    # 1.3. mu_i >= 0 for all i
    C[2:n + 2, :n] = -np.eye(n)
    # a[2:n+2] = 0

    # 2. mu being in the ambiguity set (using auxiliary variables nu)
    # 2.1. sum(nu) <= n * alpha
    C[n + 2, n:2 * n] = 1
    b[n + 2] = n * alpha
    # 2.2. -nu_i <= n * mu_i - 1 for all i
    C[n + 3:2 * n + 3, n:2 * n] = -np.eye(n)
    C[n + 3:2 * n + 3, :n] = - n * np.eye(n)
    b[n + 3:2 * n + 3] = -np.ones(n)
    # 2.3. n*mu_i - 1 <= nu_i for all i
    C[2 * n + 3:3 * n + 3, n:2 * n] = - np.eye(n)
    C[2 * n + 3:3 * n + 3, :n] = n * np.eye(n)
    b[2 * n + 3:3 * n + 3] = np.ones(n)

    # Step 2.2: construct the constraints for the dual problem
    # nonlinear constraint
    def fun(z):
        return ordered_loss(z[0], x, y) - np.dot(C.T, z[1:])
    nonlinear_constraint = NonlinearConstraint(fun, 0, 0)
    # linear constraint: lambda >= 0
    A = np.hstack((np.zeros((3 * n + 3, 1)), np.eye(3 * n + 3)))
    linear_constraint = LinearConstraint(A, 0, np.inf)

    # Step 2.3: solve the dual problem
    cost = lambda x: np.inner(b, x[1:])
    res = minimize(cost, x0=np.zeros(3 * n + 4), constraints=[nonlinear_constraint, linear_constraint])
    theta = res.x[0]
    return theta


def calibrate_alpha(n: int, m: int, beta: float, delta: float) -> float:
    """
    Calibrate the parameter alpha for the total variation ambiguity set
    using the pool adjacent violators algorithm and Dirichlet sampling.
    :param n: the dimension of the problem (int)
    :param m: the amount of samples to draw (int)
    :param beta: the confidence level (float)
    :param delta: the level for the regularized incomplete beta function will be 1 - delta (float)
    :return: the calibrated parameter alpha (float)
    """
    # repeat m times
    alpha_vector = np.zeros(m)
    for step in range(m):
        # sample mu from the Dirichlet distribution
        nu = dirichlet.rvs(np.ones(n))
        # reshape nu to column vector
        nu = nu.reshape(n)
        nu.sort(axis=0)
        alpha_vector[step] = pool_adjacent_violators(nu, lambda t: np.abs(t - 1))

    alpha_vector.sort(axis=0)
    # find smallest k such that I{1-delta}(k, m - k + 1) <= beta
    beta_func = lambda k: betainc(k, m - k + 1, 1 - delta) - beta
    try:
        k = root_scalar(beta_func, bracket=[0, m]).root
    except ValueError:
        print("Value Error triggered. Using k = m - 1")
        k = m - 1
    alpha = alpha_vector[int(np.ceil(k))]
    print("alpha: ", alpha)
    return alpha

def pool_adjacent_violators(nu: np.ndarray, phi: Callable):
    """
    Solves the minimization problem (14) in [Coppens and Patrinos, 2023]
    using the pool adjacent violators algorithm (PAVA) as explained in
    alg. 2.3 of [Best, Chakravarti and Ubhaya, 2000].
    :param nu: the random variable nu (n x 1)
    :param phi: the divergence function (R -> R)
    :return: the solution alpha (1 x 1)
    """
    def cost_function(nu, lambda_, B=None):
        if B is None:
            B = range(len(nu))
        nu = nu[list(B)]
        # constraint: t >= 0
        linear_constraint = LinearConstraint(np.array([[1]]), 0, np.inf)
        cost = 0
        n = len(B)
        for j in range(len(B)):
            cost1 = nu[j] * lambda_[j]
            cost2 = (1 / n) * minimize(lambda t: -t * lambda_[j] + phi(t),
                                       x0=np.array([0]), constraints=[linear_constraint]).x[0]
            cost -= (cost1 + cost2)
        return cost

    def constraint_matrix(size):
        """
        Generates the constraint matrix A for the monoticity constraints.
        :param size: the size of the constraint matrix
        :return: the constraint matrix A
        """
        if size == 1:
            # no constraints
            return 0

        # A is a n-1 x n matrix with ones on the diagonal and -1 on the superdiagonal
        A = np.zeros((size - 1, size))
        for i in range(size - 1):
            A[i, i] = 1
            A[i, i + 1] = -1
        return A

    n = nu.shape[0]
    J = [tuple([j]) for j in range(n)] # the partition of {1, ..., n}

    # compute minimizers for all B in J
    mu = {}  # dictionary of minimizers {B: mu_B}
    for j in range(n): # every B in J has only one element
        mu_B = minimize(lambda x: cost_function(nu, x, [j]), x0=np.array([0])).x[0]
        mu[tuple([j])] = mu_B

    B = tuple([1])
    B_plus = tuple([2])
    B_minus = tuple([])

    while len(B_plus) > 0:
        if mu[B] > mu[B_plus]:
            # set J
            J.remove(B)
            J.remove(B_plus)
            J.append(B + B_plus)
            # remove entries from mu
            mu.pop(B)
            mu.pop(B_plus)
            # set B
            B = B + B_plus
            # find new minimizer mu_B
            A = constraint_matrix(len(B))
            linear_constraint = LinearConstraint(A, ub=0, lb=-np.inf)
            x0 = np.zeros(len(B))
            res = - minimize(lambda x: cost_function(nu, x, B), x0=x0, constraints=[linear_constraint]).x[0]
            # we use minus because we minimize the negative of the function, so max f = - min -f
            mu[B] = res
            while len(B_minus) > 0 and mu[B_minus] > mu[B]:
                # set J
                J.remove(B)
                J.remove(B_minus)
                J.append(B + B_minus)
                # remove entries from mu
                mu.pop(B)
                mu.pop(B_minus)
                # set B
                B = B + B_minus
                # find new minimizer mu_B
                A = constraint_matrix(len(B))
                linear_constraint = LinearConstraint(A, ub=0, lb=-np.inf)
                x0 = np.zeros(len(B))
                res = - minimize(lambda x: cost_function(nu, x, B), x0=x0, constraints=[linear_constraint]).x[0]
                mu[B] = res
        else:
            B = B_plus
            # B_plus is the B containing the next element
            p = max(B) + 1
            for B_candidate in J:
                if p in B_candidate:
                    B_plus = B_candidate
                    break
            # B_minus is the B containing the previous element
            m = min(B) - 1
            for B_candidate in J:
                if m in B_candidate:
                    B_minus = B_candidate
                    break

        # any mu_B in mu is a minimizer for the problem
        minimizers = [mu[B] for B in mu]
        return min(minimizers)


# 3. Shifted Conditional Value-at-Risk (CV@R) Approach

def shifted_conditional_value_at_risk(x, y, delta):
    """
    The shifted conditional value-at-risk approach as explained in
    [Coppens and Patrinos, 2023]
    :param x: input values (n x 1)
    :param y: actual values (n x 1)
    :param delta: confidence level
    :return: the weights (1 x 1)
    """

    # auxiliary functions
    def ordered_loss(theta, x, y, mu):
        # construct for every (xi, yi) the loss_vector function
        loss_vector = np.array([squared_loss(yi, theta * xi) for xi, yi in zip(x, y)])
        # sort the loss_vector in ascending order
        loss_vector = np.sort(loss_vector, axis=0)
        # compute the ordered risk
        ordered_risk = np.dot(mu.T, loss_vector)
        return ordered_risk

    def get_mu(n, delta):
        mu = calibrate(length=n, level=delta, method="brentq", full_output=False)
        return mu

    # construct the ordered risk
    mu_star = get_mu(x.shape[0], delta)
    loss = lambda theta: ordered_loss(theta, x, y, mu_star)
    res = minimize(loss, x0=np.array([0]))
    theta = res.x
    return theta


if __name__ == '__main__':
    # set up a linear regression problem
    # y = 2x + noise
    k = 75
    x = np.random.rand(k, 1)
    y = 2 * x + .2 * np.random.randn(k, 1)

    # training sets contain 70% of the data
    # select 70% of the data randomly
    idx = np.random.choice(k, int(k * .7), replace=False)
    x_train = x[idx]
    y_train = y[idx]

    # test sets contain 30% of the data
    # difference between sets is that the test set does not contain the training set
    idx = np.setdiff1d(np.arange(k), idx)
    x_test = x[idx]
    y_test = y[idx]

    # plot the training and test sets
    plt.scatter(x_train, y_train, label='training set')
    plt.scatter(x_test, y_test, label='test set')
    plt.legend()
    plt.show()

    # # compute the weights using SAA
    # theta = sample_average_approximation(x_train, y_train)
    # print("SAA gives: ", theta)
    # print("MSE: ", np.mean(squared_loss(y_test, theta * x_test)))
    # plt.figure()
    # plt.scatter(x_test, y_test)
    # plt.plot(x_test, theta * x_test, color='red')
    # plt.title('SAA')
    #
    # # compute the weights using CV@R approach
    # theta = shifted_conditional_value_at_risk(x_train, y_train, delta=0.95)
    # print("CV@R gives: ", theta)
    # print("MSE: ", np.mean(squared_loss(y_test, theta * x_test)))
    # plt.figure()
    # plt.scatter(x_test, y_test)
    # plt.plot(x_test, theta * x_test, color='red')
    # plt.title('CV@R')
    #
    # # compute the weights using robust ORM
    # theta = ordered_risk_minimization_robust(x_train, y_train)
    # print("ORM gives: ", theta)
    # print("MSE: ", np.mean(squared_loss(y_test, theta * x_test)))
    # plt.figure()
    # plt.scatter(x_test, y_test)
    # plt.plot(x_test, theta * x_test, color='red')
    # plt.title('ORM')

    # compute the weights using total variation ORM
    start = time.time()
    theta = ordered_risk_minimization_total_variation(x_train, y_train, m=5, beta=0.005, delta=0.2)
    end = time.time()
    print("ORM with total variation gives: ", theta)
    print("Runtime is ", end - start, " seconds")
    print("MSE: ", np.mean(squared_loss(y_test, theta * x_test)))
    plt.figure()
    plt.scatter(x_test, y_test)
    plt.plot(x_test, theta * x_test, color='red')
    plt.title('ORM with total variation')

    plt.show()
