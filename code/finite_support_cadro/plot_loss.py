from robust_optimization import RobustOptimization
from one_dimension_cadro import CADRO1DLinearRegression
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
    m = 100
    sigma = 1
    theta = np.linspace(0, 5, 100)
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
    ellipsoid = Ellipsoid.smallest_enclosing_sphere(data)
    for i in range(len(theta)):
        problem = CADRO1DLinearRegression(data, ellipsoid)
        results = problem.solve(theta=theta[i], nb_theta_0=2, theta0=None)
        loss[i] = problem.test_loss(data_test)
        objective[i] = results['objective']

    # get theta_0, theta_r and theta_star
    problem = CADRO1DLinearRegression(data, ellipsoid)
    results = problem.solve(theta0=None, nb_theta_0=2)
    theta_0 = results['theta_0']
    theta_star = results['theta']
    problem.set_theta_r()
    theta_r = problem.theta_r
    print("m: ", m, " sigma: ", sigma)
    problem.print_results(include_robust=True)



    # plot the loss function
    plt.figure()
    # plt.plot(theta, loss, label="loss function")
    plt.plot(theta, objective, label="objective")
    # add a vertical line for theta_0, theta_r and theta_star
    for i, theta0 in enumerate(theta_0):
        plt.axvline(x=theta0, linestyle='--', color='k', label=r"$\theta_{" + str(i+1) + r"}$")
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