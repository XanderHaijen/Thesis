from SDP_procedure import ellipsoidal_cadro, tau, generate_data, generate_outliers
from robust_optimization import RobustOptimization
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
    theta = np.linspace(0.30, 0.40, 100)
    theta_0 = 0.35
    scaling_factor_ellipse = 1
    loss = np.zeros(len(theta))
    objective = np.zeros(len(theta))

    # setup data
    x = np.linspace(-2, 2, m)
    np.random.shuffle(x)
    y = generate_data(x, rico, pdf="normal", mu=0, sigma=sigma)
    data = np.vstack((x, y))
    generate_outliers(data, rico, sigma, 5, 1 / 7)

    data_test = np.array([np.linspace(-2, 2, m_test), generate_data(np.linspace(-2, 2, m_test), rico,
                                                                    pdf="normal", mu=0, sigma=sigma)])
    R = np.array([[np.cos(np.pi / 3), -np.sin(np.pi / 3)], [np.sin(np.pi / 3), np.cos(np.pi / 3)]])
    for i in range(len(theta)):
        results = ellipsoidal_cadro(data, data_test, tau, theta=theta[i], ellipse_alg='princ', theta_0=theta_0,
                                    scaling_factor_ellipse=scaling_factor_ellipse, R=R)
        loss[i] = results['train_loss']
        objective[i] = results['lambda'] * results['alpha'] + results['tau']

    # get theta_0, theta_r and theta_star
    results = ellipsoidal_cadro(data, data_test, tau, ellipse_alg='princ', theta_0=theta_0,
                                scaling_factor_ellipse=scaling_factor_ellipse, plot=True, R=R)
    theta_0 = results['theta_0']
    theta_star = results['theta_star']
    theta_r, _ = solve_robust_quadratic_loss(results['A'], results['a'], results['c'])
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