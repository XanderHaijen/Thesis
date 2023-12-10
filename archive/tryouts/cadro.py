from typing import Callable, Union

import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt
from tryouts.study_minimal import calibrate


def cadro(data: np.ndarray,
          data_range: int,
          tau: Callable[[int], int],
          beta: int,
          loss: Callable,
          constraint_x: Union[LinearConstraint, NonlinearConstraint],
          x0: np.ndarray) -> (float, np.ndarray):
    """
    Implements the CADRO (cost-aware distributionally robust optimization) algorithm.
    :param data: the input data set drawn from a set {1, ..., d}
    :param data_range: the range of the input data set (d)
    :param tau: the partition function (m -> N)
    :param beta: the confidence level (0 < beta < 1)
    :param loss: the element-wise loss function (R^d -> R)
    :param constraint_x: the constraint on the decision variable x (to be implemented)
    :param x0: the initial guess of the decision variable x
    :return: the optimal value V and the optimal solution x
    """
    def L(x, data):
        """
        The loss function of the calibration problem.
        :param x: the decision variable
        :param data: the input data set drawn from a set {1, ..., d}
        :return: the loss function (1 x d)
        """
        loss_vector = np.zeros(d)

        for i in range(d):
            loss_vector[i] = loss(x, i + 1) * np.count_nonzero(data == i + 1)

        return loss_vector

    def cost(x, lambda_, v, alpha):
        max_cost = np.zeros(d)
        for i in range(d):
            max_cost[i] = (loss(x, i + 1) - lambda_*v[i]) * np.count_nonzero(data == i + 1)
        cost = lambda_ * alpha + np.max(max_cost)
        return cost

    ### STEP 1: partition the data set into training and calibration sets
    m = data.shape[0]
    d = data_range
    split = tau(m)
    train_set = data[:split]
    calibrate_set = data[split:]

    ### STEP 2: Find v(tau) for use in calibration
    # construct the empirical distribution of the training set as the set of d basis vectors of R^d
    empirical_dist = np.zeros(d) # only store the diagonal entries
    for i in range(split):
        data_point = train_set[i]
        empirical_dist[data_point - 1] += 1 / split

    # search x_bar, such that it minimizes <L(x), empirical_dist>
    V_star_cost = lambda x: np.inner(L(x, train_set), empirical_dist)
    x_bar = minimize(V_star_cost, x0=x0, constraints=[constraint_x]).x

    # set v(tau) = L(x_bar, train_set)
    v = L(x_bar, train_set)

    ### STEP 3: Calibrate alpha
    # alpha = calibrate_alpha(m, 10, 0.005, beta)
    m_prime = calibrate_set.shape[0]
    calibration = calibrate(length=100, level=beta, method="brentq", full_output=True)
    gamma = calibration.info["radius"]
    kappa = int(np.ceil(m_prime * gamma))
    # eta_k = <v, e_xi{k}> for k = 1, ..., calibrate_set.shape[0]
    eta = np.zeros(m_prime)
    for k in range(calibrate_set.shape[0]):
        eta[k] = v[calibrate_set[k] - 1]
    eta.sort(axis=0)
    eta_bar = np.max(v[0:d])
    alpha = (kappa / m_prime - gamma) * eta[kappa - 1] + np.sum(eta[kappa:m_prime]) / m_prime + gamma * eta_bar

    ### STEP 4: Solve the CADRO problem
    dual_cost = lambda y: cost(y[:-1], y[-1], v, alpha)
    lambda_constraint_A = np.zeros([x0.shape[0] + 1, x0.shape[0] + 1])
    lambda_constraint_A[-1, -1] = 1
    lambda_constraint = LinearConstraint(lambda_constraint_A, ub=np.inf, lb=0)
    y0 = np.append(x0, 0)
    if constraint_x is None:
        problem = minimize(dual_cost, y0, constraints=lambda_constraint)
    else:
        constraint_matrix = np.eye(x0.shape[0] + 1)
        ub = np.append(np.ones(x0.shape[0]), np.inf)
        lb = np.append(np.zeros(x0.shape[0]), 0)
        total_constraint = LinearConstraint(constraint_matrix, ub=ub, lb=lb)
        problem = minimize(dual_cost, y0, constraints=[total_constraint])

    # optimal value
    V = problem.fun
    # optimal solution
    x = problem.x[:-1]

    return V, x


if __name__ == "__main__":
    # set up a resource allocation problem
    n_x = 3
    d = 50
    beta = 0.01
    m = 10000
    mu, nu = 0.01, 0.8
    tau = lambda m: int(np.floor(nu * mu * (m * (m + 1)) / (mu * m + nu)))
    data = np.random.randint(1, d + 1, m)
    # given points of interest: 2D vectors
    # generate d random 2D vectors in [0, 1]^2
    points = np.random.rand(d, 2)


    def cost(drop_locations: np.ndarray, poi: int):
        """
        The cost function of the resource allocation problem.
        :param drop_locations: the drop locations (n_x x 2)
        :param poi: the point of interest (int)
        :return: the maximum norm 2 difference between the poi and the drop locations
        """
        distances = [np.linalg.norm(points[poi - 1, :] - drop_locations[i:i+2]) for i in range(0, n_x * 2, 2)]
        return np.max(distances)

    # take x0 as three random 2D vectors in [0, 1]^2
    x0 = np.random.rand(n_x * 2)
    # restrict x0 to be in [0, 1]^2
    constraint_matrix = np.eye(n_x * 2)
    constraint = LinearConstraint(constraint_matrix, ub=1, lb=0)
    V, x = cadro(data, d, tau, beta, cost, constraint, x0)

    # plot the points of interest as red x's
    plt.scatter(points[:, 0], points[:, 1], c="red", marker="x")
    # reshape x from a 1 x n_x * 2 vector to a n_x x 2 matrix
    x = x.reshape(n_x, 2)
    # plot the drop locations as blue o's
    plt.scatter(x[:, 0], x[:, 1], c="blue", marker="o")
    optimal_cost = sum([cost(x, i) for i in range(1, d+1)])
    print("The optimal cost is: ", optimal_cost)
    print("The optimal value is: ", V)
    plt.show()