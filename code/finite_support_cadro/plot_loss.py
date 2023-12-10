from robust_optimization import RobustOptimization
from continuous_cadro import CADRO1DLinearRegression
from ellipsoids import Ellipsoid
from utils.data_generator import ScalarDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import warnings

def main():
    # for different values of theta, plot the loss function
    np.random.seed(0)
    rico = 3
    m_test = 1000
    m = 50
    sigma = 1
    theta = np.linspace(0, 5, 100)
    theta_0 = 0.35
    loss = np.zeros(len(theta))
    objective = np.zeros(len(theta))

    # setup data
    x = np.linspace(-2, 2, m)
    x_test = np.linspace(-2, 2, m_test)
    np.random.shuffle(x)
    data_gen = ScalarDataGenerator(x, seed=0)
    test_data_gen = ScalarDataGenerator(x_test, seed=0)
    y = data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
    # beta distribution
    # y = data_gen.generate_linear_beta_disturbance(0, sigma, rico, outliers=True)
    data = np.vstack((x, y))

    y_test = test_data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
    data_test = np.vstack((x_test, y_test))
    R = np.array([[np.cos(np.pi / 3), -np.sin(np.pi / 3)], [np.sin(np.pi / 3), np.cos(np.pi / 3)]])
    ellipsoid = Ellipsoid.from_principal_axes(R, data, rico)
    for i in range(len(theta)):
        problem = CADRO1DLinearRegression(data, ellipsoid)
        results = problem.solve(theta=theta[i])
        loss[i] = problem.test_loss(data_test)
        objective[i] = results['lambda'] * results['alpha'] + results['tau']

    # get theta_0, theta_r and theta_star
    problem = CADRO1DLinearRegression(data, ellipsoid)
    results = problem.solve()
    theta_0 = results['theta_0']
    theta_star = results['theta']
    robust_opt = RobustOptimization(ellipsoid)
    results_r = robust_opt.solve_1d_linear_regression()
    theta_r = results_r['theta']
    print("m: ", m, " sigma: ", sigma)
    print("theta_0: ", theta_0)
    print("theta_r: ", theta_r)
    print("theta_star: ", theta_star)

    # plot the loss function
    plt.figure()
    # plt.plot(theta, loss, label="loss function")
    plt.plot(theta, objective, label="objective")
    # add a vertical line for theta_0, theta_r and theta_star
    plt.axvline(x=theta_0, linestyle='--', color='k', label=r"$\theta_0$")
    plt.axvline(x=theta_r, linestyle='--', color='r', label=r"$\theta_r$")
    plt.axvline(x=theta_star, linestyle='--', color='g', label=r"$\theta^*$")
    plt.legend()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\ell(\theta, \xi)$")
    plt.title("Loss function")
    plt.grid()
    # plt.savefig("loss_function_lj.png")
    plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()