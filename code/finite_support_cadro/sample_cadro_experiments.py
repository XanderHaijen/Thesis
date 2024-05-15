import numpy as np
from sample_cadro import SampleCadro
import utils.sample_experiments as aux
import matplotlib.pyplot as plt
from datetime import datetime

def linear_features_experiment(seed):
    """
    Test problem for the sampling cadro case: binary classification
    given points in a square [0, 1] x [0, 1], the goal is to find a linear classifier that identifies the boundary
    line between the two groups. The loss function is the binary cross-entropy loss.
    """
    # generate the data
    generator = np.random.default_rng(seed)
    m = 200  # training data
    n = 2000  # samples

    # x = [1, x1, x2]
    x = np.vstack((np.ones((m, )), generator.uniform(0, 1, m), generator.uniform(0, 1, m)))
    # y = step(2x1 - x2 - 0.5)
    y = 2 * x[1] - x[2] - 0.5
    y = np.where(y > 0, 1, 0)
    x += generator.normal(0, 0.1, x.shape)
    data = np.vstack((x, y)).T
    # put a uniform grid over the square
    sz = int(np.sqrt(n / 2))
    ls1 = np.linspace(np.min(x[1]) - 0.25, np.max(x[1]) + 0.25, sz)
    ls2 = np.linspace(np.min(x[2]) - 0.25, np.max(x[2]) + 0.25, sz)
    xx, yy = np.meshgrid(ls1, ls2)
    samples = np.vstack((np.ones_like(xx.flatten()), xx.flatten(), yy.flatten()))

    i = 0
    while i < samples.shape[1]:
        sample = samples[:, i]
        if sample[1] + sample[2] > 2.5 or sample[1] + sample[2] < 0 \
                or sample[1] - sample[2] < -1.5:
            samples = np.delete(samples, i, axis=1)
        else:
            i += 1

    # pre-processing: get an accurate support

    samples_1 = list()
    samples_0 = list()

    # for every sample, find the k nearest training points
    k = 6
    for i in range(samples.shape[1]):
        sample = samples[:, i]
        distances = np.linalg.norm(x[1:] - sample[1:, np.newaxis], axis=0, ord=1)
        idx = np.argsort(distances)
        # if all training data points have label 1
        if np.all(y[idx[:k]] == 1):
            samples_1.append(sample)
        elif np.all(y[idx[:k]] == 0):
            samples_0.append(sample)
        else:
            samples_1.append(sample)
            samples_0.append(sample)

    # get a band around the decision boundary
    # for i in range(samples.shape[1]):
    #     if np.abs(2 * samples[1, i] - samples[2, i] - 0.5) <= 0.5:
    #         samples_1.append(samples[:, i])
    #         samples_0.append(samples[:, i])
    #     elif 2 * samples[1, i] - samples[2, i] - 0.5 > 0.5:
    #         samples_1.append(samples[:, i])
    #     else:
    #         samples_0.append(samples[:, i])

    # add the labels
    samples_1 = np.vstack(samples_1).T
    samples_1 = np.vstack((samples_1, np.ones(samples_1.shape[1])))
    samples_0 = np.vstack(samples_0).T
    samples_0 = np.vstack((samples_0, np.zeros(samples_0.shape[1])))
    samples = np.hstack((samples_1, samples_0))
    generator.shuffle(samples.T)

    # plot the training data together with the labels as a color
    import matplotlib.pyplot as plt
    plt.scatter(x[1], x[2], c=y)
    # plt.show()

    # plot the samples
    # all 1-labeled samples are red, all 0-labeled samples are blue
    plt.scatter(samples_1[1], samples_1[2], c='red', marker='o', alpha=0.5)
    plt.scatter(samples_0[1], samples_0[2], c='blue', marker='x', alpha=0.5)
    plt.show()

    plt.figure()
    plt.scatter(samples_1[1], samples_1[2], c='red', marker='o', alpha=0.5)
    plt.scatter(samples_0[1], samples_0[2], c='blue', marker='x', alpha=0.5)
    x_lin = np.linspace(-0, 1.3, 50)
    plt.plot(x_lin, 2 * x_lin - 0.5, 'k--')
    plt.show()

    # create the CADRO object
    cadro = SampleCadro(data.T, samples, loss_function=aux.hinge_loss, seed=seed, variable_dimension=3,
                        regularization=aux.regularizer)
    results = cadro.solve()

    cadro.print_results(include_robust=True)

    print(f"Robust cost: {cadro.robust_cost}")

    # predict the label for all samples
    unlabeled_samples = samples[:-1, :]
    # labels = np.array([aux.predict(cadro.theta, sample, classifier) for sample in unlabeled_samples.T])
    labels = np.array([aux.classifier(cadro.theta, sample) for sample in unlabeled_samples.T])
    # plot the samples together with the labels
    plt.scatter(samples[1], samples[2], c=labels)
    plt.show()


def hyperbolic_features_experiment(generator: np.random.Generator, m: int, nb_samples: int, plot=False, verbose=False):
    """
    Test problem for the sampling cadro case: binary classification
    given points in a square [0, 1] x [0, 1], the goal is to find a linear classifier that identifies the boundary
    line between the two groups. The loss function is the binary cross-entropy loss.
    :param generator: numpy random generator
    :param m: number of training samples
    :param nb_samples: number of samples
    :param plot: boolean, whether to plot the results
    :param verbose: boolean, whether to print the results
    """

    # generate and set-up the data
    x = np.vstack((np.ones((m,)), generator.uniform(-1, 1, m), generator.uniform(-1, 1, m)))
    x = np.vstack((x, x[1] * x[2]))
    # y = step(2x1 - x2 - 0.5)
    y = x[1] * x[2]
    y = np.where(y > 0, 1, 0)
    x += generator.normal(0, 0.1, x.shape)
    data = np.vstack((x, y)).T

    # put a uniform grid over the square for the sample points
    sz = int(np.sqrt(nb_samples / 2))
    ls1 = np.linspace(np.min(x[1]) - 0.25, np.max(x[1]) + 0.25, sz)
    ls2 = np.linspace(np.min(x[2]) - 0.25, np.max(x[2]) + 0.25, sz)
    xx, yy = np.meshgrid(ls1, ls2)
    samples = np.vstack((np.ones_like(xx.flatten()), xx.flatten(), yy.flatten(), xx.flatten() * yy.flatten()))

    # pre-processing of the support
    samples_1 = list()
    samples_0 = list()

    # Method 1: for every sample, find the k nearest training points. If the labels are not all the same,
    # add both possibilities to the support
    k = 5
    for i in range(samples.shape[1]):
        sample = samples[:, i]
        distances = np.linalg.norm(x[1:] - sample[1:, np.newaxis], axis=0, ord=1)
        idx = np.argsort(distances)
        if np.all(y[idx[:k]] == 1):  # all training data points have label 1 -> add one point
            samples_1.append(sample)
        elif np.all(y[idx[:k]] == 0):  # all training data points have label 0 -> add one point
            samples_0.append(sample)
        else:  # add both possibilities
            samples_1.append(sample)
            samples_0.append(sample)

    # Method 2: get a band around the decision boundary
    # for i in range(samples.shape[1]):
    #     if np.abs(samples[1, i] * samples[2, i]) <= 0.1:
    #         samples_1.append(samples[:, i])
    #         samples_0.append(samples[:, i])
    #     elif samples[1, i] * samples[2, i] > 0.1:
    #         samples_1.append(samples[:, i])
    #     else:
    #         samples_0.append(samples[:, i])

    # add the labeled support points
    samples_1 = np.vstack(samples_1).T
    samples_1 = np.vstack((samples_1, np.ones(samples_1.shape[1])))
    samples_0 = np.vstack(samples_0).T
    samples_0 = np.vstack((samples_0, np.zeros(samples_0.shape[1])))
    samples = np.hstack((samples_1, samples_0))
    generator.shuffle(samples.T)

    # Method 3: add all samples
    # samples_1 = np.vstack((samples, np.ones(samples.shape[1])))
    # samples_0 = np.vstack((samples, np.zeros(samples.shape[1])))
    # samples = np.hstack((samples_1, samples_0))
    # generator.shuffle(samples.T)


    # plot the samples
    if plot:
        plt.figure()
        plt.scatter(samples_1[1], samples_1[2], c='green', marker='x', alpha=0.5)
        plt.scatter(samples_0[1], samples_0[2], c='orange', marker='o', alpha=0.5)
        # plot the training data together with the labels as a color
        plt.scatter(x[1], x[2], c=y, cmap='coolwarm', marker='o')
        # plot the decision boundary
        x_lin = np.linspace(-1.3, 1.3, 50)
        plt.plot(np.zeros(50), x_lin, 'k--', label="Decision boundary")
        plt.plot(x_lin, np.zeros(50), 'k--')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # create the CADRO object
    cadro = SampleCadro(data.T, samples, loss_function=aux.hinge_loss, seed=seed, variable_dimension=4,
                        regularization=None)
    results = cadro.solve()

    if verbose == True:
        cadro.print_results(include_robust=True)
        print(f"Robust cost: {cadro.robust_cost}")

    # predict the label for all samples
    unlabeled_samples = samples[:-1, :]
    # labels = np.array([aux.predict(cadro.theta, sample, classifier) for sample in unlabeled_samples.T])
    labels = np.array([aux.classifier(results["theta"], sample) for sample in unlabeled_samples.T])

    if plot:
        # plot the samples together with the labels
        plt.scatter(samples[1], samples[2], c=labels, cmap='coolwarm', marker='o')

        # plot the decision boundary
        th = results["theta"]
        x_lin = np.linspace(-1.5, 1.5, 500)
        y_lin = (- th[0] - th[1] * x_lin) / (th[2] + th[3] * x_lin)
        y_lin[np.abs(y_lin) > 1.5] = np.nan
        plt.plot(x_lin, y_lin, 'k--', label="Decision boundary")
        plt.show()

    labels = np.sign(labels)
    if plot:
        # plot the samples together with the labels
        plt.figure()
        plt.scatter(samples[1], samples[2], c=labels, cmap='coolwarm', marker='o')
        plt.show()

    # accuracy
    accuracy = np.mean(labels == samples[-1])

    # accuracy at theta_0
    labels_0 = np.array([aux.predict(results["theta_0"], sample, aux.classifier) for sample in unlabeled_samples.T])
    accuracy_0 = np.mean(labels_0 == samples[-1])

    # check if the solution has collapsed
    collapse = True if np.abs(results["lambda"]) < 1e-6 else False

    return accuracy, accuracy_0, collapse


if __name__ == "__main__":
    seed = 0
    plt.rcParams.update({'font.size': 15})
    generator = np.random.default_rng(seed)

    # linear_features_experiment(seed)

    m = [20, 40, 80, 160] #, 320, 640, 1280]
    nb_samples = 200
    nb_tries = 10
    accuracies = np.zeros((len(m), nb_tries))
    accuracies_0 = np.zeros((len(m), nb_tries))
    collapses = np.zeros(len(m))
    for i in range(len(m)):
        with open("progress.txt", "a") as f:
            f.write(f"{datetime.now()} - Running experiment for m = {m[i]}...\n")
        for j in range(nb_tries):
            accuracy, accuracy_0, collapse = hyperbolic_features_experiment(generator, m[i], nb_samples)
            accuracies[i, j] = accuracy
            accuracies_0[i, j] = accuracy_0
            collapses[i] += collapse

    median_accuracies = np.median(accuracies, axis=1)
    p75_accuracies = np.percentile(accuracies, 75, axis=1)
    p25_accuracies = np.percentile(accuracies, 25, axis=1)

    median_accuracies_0 = np.median(accuracies_0, axis=1)
    p75_accuracies_0 = np.percentile(accuracies_0, 75, axis=1)
    p25_accuracies_0 = np.percentile(accuracies_0, 25, axis=1)

    plt.errorbar(m, median_accuracies, yerr=[median_accuracies - p25_accuracies, p75_accuracies - median_accuracies],
                    fmt='o-', label=r'Accuracy at $\theta^\star$')
    plt.errorbar(m, median_accuracies_0, yerr=[median_accuracies_0 - p25_accuracies_0, p75_accuracies_0 - median_accuracies_0],
                    fmt='o-', label=r'Accuracy at $\theta_0$')
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig("accuracy_hyperbolic_features.png")

    # create bar plot for the collapses
    collapses /= nb_tries
    plt.plot(m, collapses, 'o-')
    plt.xlabel('Number of training samples')
    plt.ylabel('Collapse rate')
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("collapse_hyperbolic_features.png")

    plt.rcParams.update({'font.size': 10})
