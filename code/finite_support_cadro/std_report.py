import matplotlib.pyplot as plt
from SDP_procedure import *

m = 70
m_test = 1000
rico = 3
mu = 0
sigma_list = [0.1, 0.5, 0.7, 1.0, 1.5, 2.0]
theta_means = np.zeros(len(sigma_list))
theta_stds = np.zeros(len(sigma_list))
loss_means = np.zeros(len(sigma_list))
loss_stds = np.zeros(len(sigma_list))
for i, sigma in enumerate(sigma_list):
    x = np.linspace(0, 1, m)
    np.random.shuffle(x)
    nb_iter = 40
    differences = np.zeros(nb_iter)
    loss_changes = np.zeros(nb_iter)
    pdf = lambda x, mu, sigma: p(x, mu, sigma, n=2)
    for _ in range(nb_iter):
        y = generate_data(x, rico, pdf="normal", set_zero=int(m / 6), mu=mu, sigma=sigma)
        data = np.vstack((x, y))
        x_test = np.linspace(0, 1, m_test)
        y_test = generate_data(x_test, rico, pdf="normal", set_zero=int(m_test / 6), mu=mu, sigma=sigma)
        data_test = np.vstack((x_test, y_test))
        results = ellipsoidal_cadro(data, data_test, tau, plot=False, report=False, generate_outliers=True,
                                    outlier_fraction=0.2)
        differences[_] = results["theta_difference"]
        loss_changes[_] = results["loss_difference"] / results["test_loss_0"]

    # print to file
    mean_difference = np.mean(differences)
    std_difference = np.std(differences)
    max_difference = np.max(differences)
    mean_loss_change = np.mean(loss_changes)  # relative change
    std_loss_change = np.std(loss_changes)
    best_change = np.min(loss_changes)
    print("sigma = ", sigma)
    print("theta differences: ", mean_difference, " +- ", std_difference)
    with open("cadro_results_std.txt", "a") as file:
        file.write("sigma = " + str(sigma) + "\n")
        # only keep 4 digits
        file.write("theta differences: " + str(np.round(mean_difference, 4)) + " +- " +
                   str(np.round(std_difference, 4)) + "\n")
        file.write("Maximum theta change: " + str(np.round(max_difference, 4)) + "\n")
        file.write("loss change: " + str(np.round(mean_loss_change * 100, 4)) +
                   " +- " + str(np.round(std_loss_change * 100, 4)) + "%\n")
        file.write("Best loss change: " + str(np.round(best_change * 100, 4)) + "%\n")
        file.write("\n")
    theta_means[i] = mean_difference
    theta_stds[i] = std_difference
    loss_means[i] = mean_loss_change * 100  # percentages
    loss_stds[i] = std_loss_change * 100
# plot results
plt.figure()
plt.errorbar(sigma_list, theta_means, yerr=theta_stds, fmt='o-', label=r"$\theta^* - \theta_0$")
plt.xlabel(r"$\sigma$")
plt.xticks(sigma_list)
plt.grid()
plt.ylabel(r"$\theta$ difference")
plt.legend()
plt.savefig("theta_difference_std.png")
plt.figure()
plt.errorbar(sigma_list, loss_means, yerr=loss_stds, fmt='o-', label="$L^*-L_0$")
plt.xlabel(r"$\sigma$")
plt.xticks(sigma_list)
plt.grid()
plt.ylabel("loss change (%)")
plt.legend()
plt.savefig("loss_change_std.png")
plt.show()