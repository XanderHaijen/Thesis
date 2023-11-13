from typing import Callable, Union
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import study_minimal
from scipy.optimize import brentq
import ellipsoids
import warnings


def ellipsoidal_cadro(data: np.ndarray,
                      data_test: np.ndarray,
                      tau: Callable,
                      plot: bool = False,
                      report: bool = False,
                      scaling_factor_ellipse: float = 1,
                      mu: float = 0.01,
                      nu: float = 0.8,
                      generate_outliers: bool = None,
                      outlier_fraction: float = None,
                      ellipse_alg: str = "lj",
                      ellipse_matrices=None,
                      theta: float = None,
                      theta_0: float = None) -> dict:
    """
    This function implements the ellipsoidal CADRO procedure for the least squares problem assuming a linear model and
    data with a finite support. The procedure is as follows:
    0) Divide the data into training, calibration, and test sets
    1) Solve some data-driven proxy (empirical risk minimization) to obtain an initial estimate of the parameter
    2) Find the surrounding ellipsoid for the training data
    3) Calibrate the ambiguity set using the cost-aware approach
    4) Solve the CADRO problem using cvxpy
    :param data: the data used to train the model (np.ndarray)
    :param data_test: the data used to test the model. This data is not used in the procedure (np.ndarray)
    :param tau: the function used to determine the size of the training set. (Callable)µ
    :param plot: whether to plot the results (bool)
    :param report: whether to print the results (bool)
    :param scaling_factor_ellipse: the scaling factor for the surrounding ellipse (float)
    :param mu: the parameter used in the definition of tau (float)
    :param nu: the parameter used in the definition of tau (float)
    :param generate_outliers: whether to generate outliers (bool)
    :param outlier_fraction: the fraction of data points which are outliers (float)
    :param ellipse_alg: the algorithm used to construct the surrounding ellipse. Either "pca", 'lj', "circ" or "manual"
    (str)
    :param ellipse_matrices: the matrices used to construct the surrounding ellipse. Only used if ellipse_alg is
    "manual" (tuple of np.ndarray in the order (A, a, c))
    :return: a dictionary containing the results of the procedure
        - theta_star: the optimal parameter (float)
        - test_loss: the test loss of the optimal parameter (float)
        - theta_0: the initial parameter (float)
        - test_loss_0: the test loss of the initial parameter (float)
        - theta_difference: the difference between the optimal parameter and the initial parameter (float)
        - loss_difference: the difference between the test loss of the optimal parameter and the test loss of the
        initial parameter (float)
        - alpha: the calibrated value of alpha (float)
        return is a dictionary
    """

    def least_squares_loss(theta, x, y):
        loss_vector = cp.sum_squares(y - cp.multiply(theta, x))
        return loss_vector

    def atomic_loss(theta, x, y):
        return (x * theta - y) ** 2

    def loss_matrices(theta, cvxpy=False):
        """
        The quadratic loss can be written in the form
        l(theta, xi) = xi^T B(theta) xi + b(theta)^T xi + beta(theta)
        This function explicitly calculates the matrices B, b, and beta
        for the given theta.
        :param theta: the parameter (float or cvxpy.Variable)
        :param cvxpy: whether to return cvxpy variables (bool)
        :return: B, b, beta (np.ndarray, np.ndarray, float)
        :raises: ValueError if cvxpy is False and theta is a cvxpy.Variable
        """
        if not cvxpy and isinstance(theta, cp.Variable):
            raise ValueError("theta must be a float if cvxpy is False")

        if not cvxpy:
            B = np.array([[theta ** 2, -theta], [-theta, 1]])
            # b is column vector of size 2
            b = np.zeros((2, 1))
            beta = 0
            return B, b, beta
        else:
            B = cp.vstack([cp.hstack([theta ** 2, -theta]), cp.hstack([-theta, 1])])
            b = cp.vstack([0, 0])
            beta = 0
            return B, b, beta

    if generate_outliers is not None or outlier_fraction is not None:
        print("generate_outliers and outlier_fraction are deprecated.")

    # Step 0: divide the data into training, calibration, and test sets
    m = data.shape[1]
    m_train = tau(m, mu=mu, nu=nu)
    if m_train < 2:
        if report:
            print("The training set is too small. Setting m_train to 2.")
        m_train = 2
    m_cal = m - m_train
    data_train = data[:, :m_train]
    x_train = data_train[0, :]
    y_train = data_train[1, :]
    data_cal = data[:, m_train:]
    x_cal = data_cal[0, :]
    y_cal = data_cal[1, :]
    x_test = data_test[0, :]
    y_test = data_test[1, :]
    m_test = data_test.shape[1]

    # Step 1: solve some data-driven proxy (empirical risk minimization)
    if theta_0 is None:
        theta_0 = cp.Variable(1)
        objective = cp.Minimize(least_squares_loss(theta_0, x_train, y_train))
        problem = cp.Problem(objective)
        problem.solve()
        test_loss_0 = least_squares_loss(theta_0, x_test, y_test).value / m_test
        theta_0 = theta_0.value[0]
    else: # theta_0 is float and not optimizable
        test_loss_0 = least_squares_loss(theta_0, x_test, y_test).value / m_test

    if report:
        print("The optimal theta is: ", theta_0)
        print("The test loss is: ", test_loss_0)

    if plot:
        plt.figure()
        plt.axes().set_aspect('equal')
        plt.plot(x_train, y_train, 'bo', label='training data')
        # plot predicted line
        plt.plot(x_test, theta_0 * x_test, 'r', label='first prediction')

    # Step 2: find the surrounding ellipsoid for data_train
    if ellipse_alg == "pca":
        A, a, c = ellipsoids.ellipse_from_pca_2d(data, scaling_factor=scaling_factor_ellipse, plot=plot)
    elif ellipse_alg == "lj":
        A, a, c = ellipsoids.minimum_volume_ellipsoid(data, scaling_factor=scaling_factor_ellipse,
                                                      theta0=3, plot=plot)
    elif ellipse_alg == "circ":
        A, a, c = ellipsoids.smallest_enclosing_sphere(data, scaling_factor=scaling_factor_ellipse, plot=plot)
    elif ellipse_alg == "manual":
        if ellipse_matrices is None:
            raise ValueError("ellipse_matrices must be given if ellipse_alg is 'manual'")
        A, a, c = ellipse_matrices
        if plot:
            ellipsoids.plot_ellipse_from_matrices(A, a, c, theta0=1, n=200, padding=4)
    else:
        raise ValueError("ellipse_alg must be either 'pca', 'lj', 'circ' or 'manual'")

    # # for debugging: manually plot the ellipse
    # def inside_ellipse(xi):
    #    return xi.T @ A @ xi + 2 * a @ xi + c => 0
    # # create meshgrid
    # x = np.linspace(-2, 2, 100)
    # y = np.linspace(-5, 5, 250)
    # # for every point in the meshgrid, check if it is inside the ellipse.
    # # If it is, plot a cross. If not, plot a circle.
    # for i in range(x.shape[0]):
    #     for j in range(y.shape[0]):
    #         if inside_ellipse(np.array([x[i], y[j]])):
    #             plt.plot(x[i], y[j], 'r+')
    # plt.show()
    # exit(0)

    # Step 3: Calibrate ambiguity set
    m_prime = y_cal.shape[0]
    method = "brentq" if m_prime <= 80 else "asymptotic"
    calibration = study_minimal.calibrate(length=m_prime, level=0.01, method=method, full_output=True)
    gamma = calibration.info["radius"]
    kappa = int(np.ceil(m_prime * gamma))
    eta = np.array([atomic_loss(theta_0, x_cal[i], y_cal[i]) for i in range(m_prime)])
    eta.sort(axis=0)
    B, b, beta = loss_matrices(theta_0)
    eta_bar = calculate_eta_bar(B, b, beta, A, a, c, report=report)
    if eta_bar is None:
        eta_bar = np.max(eta)
    alpha = (kappa / m_prime - gamma) * eta[kappa - 1] + np.sum(eta[kappa:m_prime]) / m_prime + gamma * eta_bar
    if report:
        print("The calibrated alpha using Prop III.3 is: ", alpha)

    ## Step 4: Solve the CADRO problem
    if theta is None:
        theta = cp.Variable(1)
    # else: theta is float and not optimizable
    lambda_ = cp.Variable(1)
    gamma = cp.Variable(1)
    tau_ = cp.Variable(1)
    objective = cp.Minimize(alpha * lambda_ + tau_)

    # Step 4.1 construct LMI constraint
    B_0 = np.array([[theta_0 ** 2, -theta_0], [-theta_0, 1]])
    # b_0 is column vector of size 2
    b_0 = np.zeros((2, 1))
    beta_0 = 0
    # constructing M_11
    a = cp.reshape(a, (2, 1))
    M_111 = lambda_ * B_0 - gamma * A
    M_111 = cp.reshape(M_111, (2, 2))
    M_112 = lambda_ * b_0 - gamma * a
    M_112 = cp.reshape(M_112, (2, 1))
    M_113 = tau_ + lambda_ * beta_0 - gamma * c
    M_113 = cp.reshape(M_113, (1, 1))
    M_11 = cp.bmat([[M_111, M_112], [cp.transpose(M_112), M_113]])
    # constructing M_12 and M_21
    M_12 = cp.vstack([theta, -1, 0])
    # constructing M_22
    M_22 = 1
    M_22 = np.reshape(M_22, (1, 1))
    # combine into M
    M = cp.bmat([[M_11, M_12], [cp.transpose(M_12), M_22]])
    # construct the constraints
    constraints = [M >> 0, lambda_ >= 0]

    # Step 4.3 solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    if isinstance(theta, cp.Variable):
        theta_star = theta.value[0]
    else:
        theta_star = theta  # theta is float and not optimizable

    train_loss = least_squares_loss(theta_star, data[0, :], data[1, :]).value / m
    test_loss = least_squares_loss(theta_star, x_test, y_test).value / m_test
    if report:
        if theta is None:
            print("The optimal theta is: ", theta_star)
        print("The test loss is: ", test_loss)
        print("The difference between the optimal theta and the initial theta is: ", theta_star - theta_0)
    if plot:
        # plot a line with slope theta_star
        plt.plot(x_test, theta_star * x_test, 'g', label='CADRO line', linestyle='--')
        plt.legend()
        plt.title(r"Initial model: $\theta_0$ = %.2f, CADRO model: $\theta^*$ = %.2f" % (theta_0, theta_star))
        plt.grid()
        plt.show()

    reporting = {"theta_star": theta_star,
                 "test_loss": test_loss,
                 "train_loss": train_loss,
                 "theta_0": theta_0,
                 "test_loss_0": test_loss_0,
                 "theta_difference": theta_star - theta_0,
                 "loss_difference": test_loss - test_loss_0,
                 "alpha": alpha,
                 "lambda": lambda_.value[0],
                 "tau": tau_.value[0],
                 "A": A, "a": a, "c": c}  # ellipse matrices
    return reporting


def tau(m, mu=None, nu=None):
    if mu is None or nu is None:
        return int(m / 2)
    else:
        return int(np.floor(nu * mu * (m * (m + 1)) / (mu * m + nu)))


def calculate_eta_bar(B, b, beta, A, a, c, report=False):
    """
    Eta bar is the maximal loss obtainable within the ellipsoid defined by
    x^T A x + 2 a^T x + c <= 0
    The loss is quadratic and given by
    l(theta, xi) = xi^T B(theta) xi + b(theta)^T xi + beta(theta)
    This function solves a convex SDP to find the maximal loss using the epigraph trick and the S-lemma.
    :param B: the B matrix for defining the loss
    :param b: the b vector for defining the loss
    :param beta: the beta scalar for defining the loss
    :param A: the A matrix for defining the ellipsoid
    :param a: the a vector for defining the ellipsoid
    :param c: the c scalar for defining the ellipsoid
    :param theta_0: the initial parameter (float) used to construct a point inside the ellipsoid
    :param report: whether to print the result (bool)
    :return: the maximal loss (float)
    """
    tau = cp.Variable(1)
    lambda_ = cp.Variable(1)
    M11 = -B - lambda_ * A
    M12 = -b - lambda_ * a
    M22 = -beta - lambda_ * c + tau
    M22 = np.reshape(M22, (1, 1))
    M = cp.vstack([cp.hstack([M11, M12]), cp.hstack([cp.transpose(M12), M22])])
    constraints = [M >> 0, lambda_ >= 0]
    objective = cp.Minimize(tau)
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status == cp.INFEASIBLE:
        print("The eta maximisation problem is infeasible.")
        return None
    if report:
        print("The optimal eta_bar is: ", tau.value[0])

    return tau.value[0]


def generate_data(x, rico, pdf: Union[str, Callable] = "normal",
                  set_zero: int = 0, mu: float = 0, sigma: float = 1) -> np.ndarray:
    if pdf == "normal":
        y = rico * x + sigma * np.random.randn(x.shape[0]) + mu
        if set_zero > 0:
            indices = np.random.choice(x.shape[0], size=set_zero, replace=False)
            y[indices] = rico * x[indices]
    else:
        probabilities = pdf(np.linspace(-2, 2, 2000), mu, sigma)
        probabilities /= np.sum(probabilities)
        errors = np.random.choice(np.linspace(-2, 2, 2000), size=x.shape[0], p=probabilities)
        if set_zero > 0:
            indices = np.random.choice(x.shape[0], size=set_zero, replace=False)
            errors[indices] = 0
        y = rico * x + errors
    return y


def generate_outliers(data: np.ndarray, rico: float, sigma: float, offset: float = 5, fraction: float = 0.1):
    """
    This function generates outliers in the given data.

    Parameters:
    data (np.ndarray): The original data where outliers are to be introduced. It's a 2D array where the first row represents 'x' values and the second row represents 'y' values.
    rico (float): The slope of the line that the outliers will roughly follow.
    sigma (float): The standard deviation of the normal distribution used to generate the 'y' values of the outliers.
    offset (float, optional): The offset used to shift the 'y' values of the outliers from the line defined by 'rico'. Defaults to 5.
    fraction (float, optional): The fraction of total data points that should be outliers. Defaults to 0.1.

    Returns:
    None. The function operates in-place on the 'data' array, modifying its 'y' values to introduce outliers.
    """
    m = data.shape[1]
    indices = np.random.choice(m, size=int(fraction * m), replace=False)
    data[1, indices] = rico * data[0, indices] + np.random.choice([-1, 1], size=int(fraction * m)) * offset * sigma


def p(x, mu, sigma, n=1):
    pdf_normal = (1 / sigma * (2 * np.pi) ** 0.5) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
    pdf = pdf_normal * np.sin(x / n) ** 2 * (np.tanh((x - mu) / n) + 1) / 2
    return pdf

# if __name__ == '__main__':
#     mu = 0
#     rico = 1.75
#     m_test = 1000
#     m = 100
#     sigma = 0.5
#     x = np.linspace(-2, 2, m)
#     np.random.shuffle(x)
#     y = generate_data(x, rico, pdf="normal", mu=mu, sigma=sigma)
#     data = np.vstack([x, y])
#     # generate outliers: for m / 7 random points, set y = rico * x +/- 5 * sigma
#     indices = np.random.choice(m, size=int(m / 7), replace=False)
#     data[1, indices] = rico * data[0, indices] + np.random.choice([-1, 1], size=int(m / 7)) * 5 * sigma
#
#     data_test = np.array([np.linspace(-2, 2, m_test), generate_data(np.linspace(-2, 2, m_test), rico, pdf="normal",
#                                                                         mu=mu, sigma=sigma)])
#
#     reporting = ellipsoidal_cadro(data, data_test, tau, plot=True, report=False, mu=0.01, nu=0.8,
#                                     scaling_factor_ellipse=1, generate_outliers=False, outlier_fraction=0.1,
#                                     ellipse_alg="lj")
#     print(f"loss difference is {reporting['loss_difference'] / reporting['test_loss_0']* 100} %")
