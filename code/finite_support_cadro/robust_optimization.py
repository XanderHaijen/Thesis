import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
def solve_robust_quadratic_loss(A, a, c):
    """
    Solve the robust case for a quadratic loss function and an ellipsoidal ambiguity set in the 2D case.
    The ellipsoid is parameterized by A, a, c and given by the set
    {x | x^T A x + 2 a^T x + c => 0}.
    The loss is assumed to be quadratic for a linear regression problem, and is equal to
    l(xi, theta) = (xi_2 - theta * xi_1)^2.
    :param A: (2, 2) matrix
    :param a: (2, 1) vector
    :param c: (1, 1) scalar
    :return: optimal theta and optimal value as a tuple
    """
    assert A.shape == (2, 2)
    assert a.shape == (2, 1)
    assert c.shape == (1, 1)
    theta = cp.Variable()
    tau = cp.Variable()
    lambda_ = cp.Variable()
    lambda_positive = [lambda_ >= 0]
    A_bar = cp.bmat([[- lambda_ * A, - lambda_ * a], [- lambda_ * a.T, - lambda_ * c + tau]])
    theta_vector = cp.vstack([theta, -1, 0])
    M = cp.bmat([[A_bar, theta_vector], [theta_vector.T, cp.reshape(1, (1, 1))]])
    constraints = [M >> 0] + lambda_positive
    objective = cp.Minimize(tau)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    if problem.status == cp.INFEASIBLE:
        raise ValueError("Problem is infeasible")

    return theta.value, problem.value


if __name__ == "__main__":
    # test the function
    vertical_axis = 1
    horizontal_axis = 3
    rotation_angle = np.pi / 4
    A = np.array([[- 1 / horizontal_axis ** 2, 0], [0, - 1 / vertical_axis ** 2]])
    R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
    A = R.T @ A @ R
    a = np.array([[0], [0]])
    c = np.array([[1]])
    theta, value = solve_robust_quadratic_loss(A, a, c)
    print("theta: ", theta)
    print("optimal value: ", value)

    # plot the ellipsoid
    ellipse = Ellipse(xy=(0,0), width=2 * horizontal_axis, height=2 * vertical_axis, edgecolor='r', fc='None',
                      angle=np.rad2deg(rotation_angle))
    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    ax.add_artist(ellipse)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid()
    # add the optimal theta
    x = np.linspace(-2, 2, 100)
    y = theta * x
    plt.plot(x, y, label=r"$\theta^*$")
    plt.legend()
    plt.show()

