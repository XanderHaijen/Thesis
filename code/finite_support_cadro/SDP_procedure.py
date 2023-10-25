from typing import Callable, Union
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import study_minimal
import study_minimal_divergence
from sklearn.decomposition import PCA
from scipy.optimize import brentq, minimize, NonlinearConstraint, LinearConstraint

def ellipsoidal_cadro(data: np.ndarray,
                      data_test: np.ndarray,
                      tau: Callable,
                      plot: bool = False,
                      report: bool = False,
                      scaling_factor_ellipse: float = 1,
                      mu: float = 0.01,
                      nu: float = 0.8,
                      generate_outliers: bool = False,
                      outlier_fraction: float = 0.1) -> dict:
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

    # Step 0: divide the data into training, calibration, and test sets
    m = data.shape[1]
    m_train = tau(m, mu=mu, nu=nu)
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

    # Step 1: find the surrounding ellipsoid for data_train
    # using PCA, find the principal axes
    pca = PCA(n_components=2)
    pca.fit(data_train.T)
    # the principal axes
    u1 = pca.components_[0]
    # corresponding singular values
    s1 = pca.singular_values_[0]
    s2 = pca.singular_values_[1]
    # take the mean to find the center of the ellipse
    center = np.mean(data_train, axis=1)
    # find the angle between the first principal axis and the x-axis
    cos_angle = np.abs(np.dot(u1, np.array([1, 0])) / np.linalg.norm(u1))
    sin_angle = np.sqrt(1 - cos_angle ** 2)
    R = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    # quadratic matrix containing axis lengths
    A = np.diag(np.array([1 / s1, 1 / s2]))
    # find the surrounding ellipse parameters
    A = - R @ A @ R.T
    a = center.T @ A
    c = - center.T @ A @ center + 1

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

    if generate_outliers:
        # generate outliers: for some points, move them to the boundary of the ellipse
        indices_train = np.random.choice(m_train, size=int(m_train * outlier_fraction), replace=False)
        indices_cal = np.random.choice(m_cal, size=int(m_cal * outlier_fraction), replace=False)
        for index_train in indices_train:
            outlier_value = outlier(x_train[index_train], A, a, c, plot=False)
            y_train[index_train] = outlier_value if outlier_value is not None else y_train[index_train]
        for index_cal in indices_cal:
            outlier_value = outlier(x_cal[index_cal], A, a, c, plot=False)
            y_cal[index_cal] = outlier_value if outlier_value is not None else y_cal[index_cal]


    if plot:
        ellipse = Ellipse(xy=center, width=s1 * scaling_factor_ellipse, height=s2 * scaling_factor_ellipse,
                          angle=np.rad2deg(np.arccos(cos_angle)), edgecolor='r', fc='None', lw=2)
        # plot ellipse
        fig, ax = plt.subplots()
        ax.add_patch(ellipse)
        plt.plot(x_train, y_train, 'bo', label='training data')
        plt.legend()
        plt.title("Surrounding ellipse")
        plt.show()


    # Step 2: solve some data-driven proxy (empirical risk minimization)
    theta_0 = cp.Variable(1)
    objective = cp.Minimize(least_squares_loss(theta_0, x_train, y_train))
    problem = cp.Problem(objective)
    problem.solve()
    test_loss_0 = least_squares_loss(theta_0, x_test, y_test).value / m_test
    if report:
        print("The optimal theta is: ", theta_0.value)
        print("The test loss is: ", test_loss_0)

    if plot:
        plt.axes().set_aspect('equal')
        plt.plot(x_train, y_train, 'bo', label='training data')
        # plot predicted line
        plt.plot(x_test, theta_0.value * x_test, 'r', label='predicted line')
        plt.legend()
        plt.title(r"Initial model: $\theta_0$ = %.2f" % theta_0.value)
        plt.show()

    # Step 3: Calibrate ambiguity set
    m_prime = y_cal.shape[0]
    # divergence = study_minimal_divergence.Divergence.TV
    # calibration = study_minimal_divergence.calibrate(divergence, nb_samples=1000, level=0.1, nb_scenarios=1000, confidence=0.01, full_output=True)
    # alpha = calibration.radius
    # print("The calibrated alpha using divergence is: ", alpha)
    calibration = study_minimal.calibrate(length=100, level=0.01, method="brentq", full_output=True)
    gamma = calibration.info["radius"]
    kappa = int(np.ceil(m_prime * gamma))
    eta = np.array([atomic_loss(theta_0.value, x_cal[i], y_cal[i])[0] for i in range(m_prime)])
    eta.sort(axis=0)
    B, b, beta = loss_matrices(theta_0.value[0])
    eta_bar = calculate_eta_bar(B, b, beta, A, a.reshape((2, 1)), c, report=report)
    if eta_bar is None:
        eta_bar = np.max(eta)
    alpha = (kappa / m_prime - gamma) * eta[kappa - 1] + np.sum(eta[kappa:m_prime]) / m_prime + gamma * eta_bar
    if report:
        print("The calibrated alpha using Prop III.3 is: ", alpha)

    ## Step 4: Solve the CADRO problem
    theta = cp.Variable(1)
    lambda_ = cp.Variable(1)
    gamma = cp.Variable(1)
    tau_ = cp.Variable(1)
    objective = cp.Minimize(alpha * lambda_ + tau_)

    # Step 4.1 construct LMI constraint
    theta_0 = theta_0.value[0]
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
    problem.solve()
    theta_star = theta.value[0]
    test_loss = least_squares_loss(theta_star, x_test, y_test).value / m_test
    if report:
        print("The optimal theta is: ", theta_star)
        print("The test loss is: ", test_loss)
        print("The difference between the optimal theta and the initial theta is: ", theta_star - theta_0)

    reporting = {"theta_star": theta_star,
                 "test_loss": test_loss,
                 "theta_0": theta_0,
                 "test_loss_0": test_loss_0,
                 "theta_difference": theta_star - theta_0,
                 "loss_difference": test_loss - test_loss_0,
                 "alpha": alpha}
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


def outlier(x, A, a, c, up=None, plot=False):
    def ellipse(x, y, A, a, c):
        xi = np.array([x, y])
        return xi.T @ A @ xi + 2 * a.T @ xi + c

    if up is None:
        up = np.random.choice([True, False])
    # set up ellipse equation
    ellipse_eq = lambda y: ellipse(x, y, A, a, c)

    if plot:
        plt.plot(np.linspace(-5, 5, 100), [ellipse_eq(y) for y in np.linspace(-5, 5, 100)], 'r')
        plt.show()

    # find the roots of the equation
    try:
        if up:
            roots = brentq(ellipse_eq, 0, 5, full_output=False, xtol=0.1)
        else:
            roots = brentq(ellipse_eq, -5, 0, full_output=False, xtol=0.1)
    except ValueError:
        return None

    return roots

def p(x, mu, sigma, n=1):
    pdf_normal = (1 / sigma * (2 * np.pi) ** 0.5) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
    pdf = pdf_normal * np.sin(x / n) ** 2 * (np.tanh((x - mu) / n) + 1) / 2
    return pdf

if __name__ == "__main__":
    mu = 0
    sigma = 1
    rico = 3
    m = 150
    m_test = 1000
    x = np.linspace(0, 3, m)
    np.random.shuffle(x)
    pdf = lambda x, mu, sigma: p(x, mu, sigma, n=2)
    y = generate_data(x, rico, pdf=pdf, set_zero=int(m/7), mu=mu, sigma=sigma)

    data = np.vstack((x, y))
    x_test = np.linspace(0, 3, m_test)
    y_test = generate_data(x_test, rico, pdf=pdf, set_zero=int(m_test/7), mu=mu, sigma=sigma)
    data_test = np.vstack((x_test, y_test))
    results = ellipsoidal_cadro(data, data_test, tau, plot=True, report=False, generate_outliers=False,
                                outlier_fraction=0.2)
    print(f"Theta: {results['theta_0']:.8f} -> {results['theta_star']:.8f}")
    print(f"Relative loss difference: {results['loss_difference'] / results['test_loss_0'] * 100:.4f} %")
