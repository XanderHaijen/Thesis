import warnings
import numpy as np
import matplotlib.pyplot as plt
from ellipsoids import Ellipsoid
from robust_optimization import RobustOptimization
from continuous_cadro import CADRO1DLinearRegression
from utils.data_generator import ScalarDataGenerator
import pandas as pd

def experiment1(seed):
    """
    For an LJ ellipsoid and a circle, plot the realized values of theta_r, theta_star and theta_0 for sigma=1 and
    m=30.
    """
    n = 50
    sigma = 1
    rico = 3

    x = np.linspace(0, 1, n)
    data_gen = ScalarDataGenerator(x, seed)
    y = data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
    data = np.vstack([x, y])
    lj_ellipsoid = Ellipsoid.lj_ellipsoid(data)
    circle = Ellipsoid.smallest_enclosing_sphere(data)

    # set up problem
    m = 30
    sigma = 1
    x_train = np.linspace(0, 1, m)
    train_data_gen = ScalarDataGenerator(x_train, seed)

    # 2.1 Löwner-John ellipsoid
    train_data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
    train_data_gen.contain_within_ellipse(lj_ellipsoid)
    data_train = np.vstack([x_train, train_data_gen.y])
    problem = CADRO1DLinearRegression(data_train, lj_ellipsoid)
    problem.solve(nb_theta_0=2)
    problem.set_theta_r()

    results = problem.results
    theta_r = results["theta_r"]
    theta_star = results["theta"]
    theta_1 = results["theta_0"][0]
    theta_2 = results["theta_0"][1]

    plt.figure()
    plt.scatter(data_train[0, :], data_train[1, :], label="data", marker=".")
    x_plot = np.linspace(-10, 10, 10)
    plt.plot(x_train, theta_r * x_train, label=r"$\theta_r = {:.4f}$".format(theta_r), linestyle="--", color="blue")
    plt.plot(x_train, theta_star * x_train, label=r"$\theta^* = {:.4f}$".format(theta_star), color="green", linestyle="--")
    plt.plot(x_train, theta_1 * x_train, label=r"$\theta_1 = {:.4f}$".format(theta_1), linestyle="--", color="grey")
    plt.plot(x_train, theta_2 * x_train, label=r"$\theta_2 = {:.4f}$".format(theta_2), linestyle="--", color="grey")
    problem.ellipsoid.plot(ax=plt.gca(), color="red", label="ellipsoid")

    plt.legend()
    plt.grid()
    plt.show()

    # 2.2 Smallest enclosing circle
    train_data_gen.generate_linear_norm_disturbance(0, sigma, rico, outliers=True)
    train_data_gen.contain_within_ellipse(circle)
    data_train = np.vstack([x_train, train_data_gen.y])
    problem = CADRO1DLinearRegression(data_train, circle)
    problem.solve(nb_theta_0=2)
    problem.set_theta_r()

    results = problem.results
    theta_r = results["theta_r"]
    theta_star = results["theta"]
    theta_1 = results["theta_0"][0]
    theta_2 = results["theta_0"][1]

    plt.figure()
    plt.scatter(data_train[0, :], data_train[1, :], label="data", marker=".")
    plt.plot(x_plot, theta_r * x_plot, label=r"$\theta_r = {:.4f}$".format(theta_r), linestyle="--", color="blue")
    plt.plot(x_plot, theta_star * x_plot, label=r"$\theta^* = {:.4f}$".format(theta_star), color="green", linestyle="--")
    plt.plot(x_plot, theta_1 * x_plot, label=r"$\theta_1 = {:.4f}$".format(theta_1), linestyle="--", color="grey")
    plt.plot(x_plot, theta_2 * x_plot, label=r"$\theta_2 = {:.4f}$".format(theta_2), linestyle="--", color="grey")
    problem.ellipsoid.plot(ax=plt.gca(), color="red", label="ellipsoid")

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    seed = 42
    experiment1(seed)