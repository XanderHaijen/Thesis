import numpy as np
import matplotlib.pyplot as plt
import sys

# add parent directory to path
sys.path.append('..')

from finite_support_cadro.ellipsoids import Ellipsoid

def plot_alphas(gca, alpha_array, lambda_array, loss_r, title=None, boxplot=False, scale='linear', marker='.'):
    ind_lambdas_1 = np.where(lambda_array > 0.99)
    ind_lambdas_0 = np.where(lambda_array < 0.01)
    ind_lambdas_else = np.where((lambda_array <= 0.99) & (lambda_array >= 0.01))
    # plot a horizontal line at the robust cost
    gca.axhline(loss_r, color='black', linestyle='dashed', linewidth=1)
    # boxplot, overlayed with the actual values of alpha
    if boxplot:
        gca.boxplot(alpha_array, showfliers=False)
    if scale == 'log':
        gca.set_yscale('log')
    gca.scatter(np.ones(len(ind_lambdas_1[0])), alpha_array[ind_lambdas_1],
                label=r"$\lambda \approx 1$", color='b', marker=marker)
    gca.scatter(np.ones(len(ind_lambdas_0[0])), alpha_array[ind_lambdas_0],
                label=r"$\lambda \approx 0$", color='r', marker=marker)
    gca.scatter(np.ones(len(ind_lambdas_else[0])), alpha_array[ind_lambdas_else],
                label=r"$\lambda$ otherwise", color='g', marker=marker)
    if title is not None:
        gca.set_title(title)
    # remove x ticks
    gca.set_xticks([])


def plot_loss_histograms(gca, loss_0_array, loss_star_array, loss_r, title=None, bins=10):
    hist_range = (min(np.min(loss_0_array), np.min(loss_star_array)),
                  max(np.max(loss_0_array), np.max(loss_star_array)))
    gca.hist(loss_0_array, bins=bins, alpha=0.5, label=r"$\theta_0$", range=hist_range)
    gca.hist(loss_star_array, bins=bins, alpha=0.5, label=r"$\theta$", range=hist_range)
    # add a vertical line for the robust cost if it is in the picture
    if hist_range[1] > loss_r > hist_range[0]:
        gca.axvline(loss_r, color='black', linestyle='dashed', linewidth=1)
    if title is not None:
        gca.set_title(title)


def plot_timings(timings_mean_array, timings_std_array, dimensions, scale='log'):
    # linear scale
    fig, ax = plt.subplots()
    ax.plot(dimensions, timings_mean_array, marker='o', linestyle='-', color='b')
    ax.fill_between(dimensions, timings_mean_array - timings_std_array, timings_mean_array + timings_std_array,
                    alpha=0.2, color='b')
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Time (s)")
    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
    plt.grid()
    plt.tight_layout()
    plt.savefig("timings.png")
    plt.close()


def plot_loss_m(gca, mean_loss_0, upper_bound_0, lower_bound_0, mean_loss_star, upper_bound_star, lower_bound_star
                , m, title=None, scale='log'):
    gca.errorbar(m, mean_loss_0, yerr=[mean_loss_0 - lower_bound_0, upper_bound_0 - mean_loss_0], label=r"$\theta_0$",
                 color='orange', fmt='o-')
    gca.errorbar(m, mean_loss_star, yerr=[mean_loss_star - lower_bound_star, upper_bound_star - mean_loss_star],
                 label=r"$\theta$", color='b', fmt='o-')
    gca.set_xlabel("m")
    gca.set_ylabel("Loss")
    if title is not None:
        gca.set_title(title)
    if scale == 'log':
        gca.set_yscale('log')
    gca.legend()
    gca.grid()


def ellipse_from_corners(corners_x: np.ndarray, theta: np.ndarray,
                         ub: float, lb: float, scaling_factor: int = 1,
                         return_corners: bool = False, plot: bool = False,
                         kind: str = "lj"):
    """
    Create the d-dimensional circumcircle based on the x-corners and the data hyperplane.
    :param corners_x: the corners of the data hypercube
    :param theta: the data hyperplane slope
    :param ub: the upper bound of for the data deviation
    :param lb: the lower bound for the data deviation
    :param scaling_factor: the scaling factor for the ellipse
    :param return_corners: whether to return the corners of the bounding box
    :param plot: whether to plot the corners of the bounding box (only for d = 2 or d = 3)
    :param kind: the type of ellipsoid to return (either lj or ses)
    """
    d = corners_x.shape[0] + 1
    m = corners_x.shape[1]
    assert m >= d * (d + 1) / 4
    # for each corner, get the hyperplane value
    corners_y = np.array([np.dot(corners_x[:, i], theta) for i in range(corners_x.shape[1])])
    corners_y_plus = corners_y + ub
    corners_y_min = corners_y - lb
    corners = np.zeros((d, 2 * m))
    corners[:d - 1, :m] = corners_x
    corners[d - 1, :m] = corners_y_plus
    corners[:d - 1, m:] = corners_x
    corners[d - 1, m:] = corners_y_min

    if plot and d == 2:
        plt.scatter(corners[:d - 1, :m], corners[d - 1, :m], label="upper bound")
        plt.scatter(corners[:d - 1, m:], corners[d - 1, m:], label="lower bound")
        plt.legend()
        plt.show()
    elif plot and d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(corners[0, :m], corners[1, :m], corners[2, :m], label="upper bound")
        ax.scatter(corners[0, m:], corners[1, m:], corners[2, m:], label="lower bound")
        plt.legend()
        plt.show()

    if kind == 'lj':
        ellipsoid = Ellipsoid.lj_ellipsoid(corners, scaling_factor=scaling_factor)
    elif kind == 'ses':
        ellipsoid = Ellipsoid.smallest_enclosing_sphere(corners, scaling_factor=scaling_factor)
    else:
        raise ValueError("kind should be either 'lj' or 'ses'")

    if return_corners:
        return ellipsoid, corners
    else:
        return ellipsoid


def hypercube_corners(a, b, d, d_max=1e6, generator=None):
    # create an array with all the corners of the d - dimensional hypercube
    M = min(2 ** d, d_max)  # maximum number of corners
    if d**2 > d_max and generator is None:
        raise ValueError("The number of corners is too large. Please provide a generator.")
    corners_x = np.zeros((int(M), d))
    k = 0
    i = 0
    while k < M:
        new_corner = np.zeros((d,))
        for j in range(d):
            if i & (1 << j):
                new_corner[j] = b
            else:
                new_corner[j] = a
        # add the corner with probability M/2^d
        if M < 2 ** d and generator.uniform() <= M / (2 ** d):
            corners_x[k, :] = new_corner
            k += 1
        elif M >= 2 ** d:
            corners_x[k, :] = new_corner
            k += 1
        i += 1
    corners_x = corners_x[:k, :]

    return corners_x


def rotation_matrix(d, theta, components: list):
    """
    Generate a rotation matrix for the d-dimensional space. All the components of the vector are rotated by theta.
    """
    # components give the two components that are rotated
    R = np.eye(d)
    R[components[0], components[0]] = np.cos(theta)
    R[components[1], components[1]] = np.cos(theta)
    R[components[0], components[1]] = -np.sin(theta)
    R[components[1], components[0]] = np.sin(theta)
    return R