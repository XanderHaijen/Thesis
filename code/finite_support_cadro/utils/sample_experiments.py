from typing import Union, Callable
import cvxpy as cp
import numpy as np


def classifier(theta: Union[np.ndarray, cp.Variable], x: np.ndarray) -> Union[float, cp.Expression]:
        """
        Linear classifier: theta^T x
        """
        if isinstance(theta, cp.Variable):
            return cp.matmul(theta.T, x)
        elif isinstance(theta, np.ndarray):
            return np.matmul(theta, x)


def predict(theta: np.ndarray, x: np.ndarray, classifier: Callable[[np.ndarray, np.ndarray], float]):
    """
    Prediction function for binary +1/-1 classification. The function is for use in inference and returns
    sgn(theta^T x).
    """
    return np.sign(classifier(theta, x))


def log_loss(theta: Union[cp.Variable, np.ndarray], xi: np.ndarray) -> Union[cp.Expression, float]:
    """
    Log loss function for binary classification. The function returns the loss for one data point.
    We also add a regularization term to the loss function.
    """
    data, label = xi[:-1], xi[-1]
    label = 2 * label - 1  # convert 0/1 to -1/1
    if isinstance(theta, cp.Variable):
        return cp.logistic(-label * classifier(theta, data))
    else:
        if np.exp(-label * classifier(theta, data)) > 1e3:
            return -label * classifier(theta, data)
        else:
            return np.log(1 + np.exp(-label * classifier(theta, data)))


def hinge_loss(theta: Union[cp.Variable, np.ndarray], xi: np.ndarray) -> Union[cp.Expression, float]:
    """
    Hinge loss function for binary classification. The function returns the loss for one data point.
    We also add a regularization term to the loss function.
    """
    data, label = xi[:-1], xi[-1]
    label = 2 * label - 1  # convert 0/1 to -1/1
    if isinstance(theta, cp.Variable):
        return cp.pos(1 - label * classifier(theta, data))
    else:
        return max(0, 1 - label * classifier(theta, data))


def square_loss(theta: Union[cp.Variable, np.ndarray], xi: np.ndarray) -> Union[cp.Expression, float]:
    """
    Square loss function for binary classification. The function returns the loss for one data point.
    We also add a regularization term to the loss function.
    """
    data, label = xi[:-1], xi[-1]
    label = 2 * label - 1  # convert 0/1 to -1/1
    if isinstance(theta, cp.Variable):
        return cp.square(1 - label * classifier(theta, data))
    else:
        return (1 - label * classifier(theta, data)) ** 2


def regularizer(theta: Union[cp.Variable, np.ndarray]) -> Union[cp.Expression, float]:
    """
    Regularization function for the CADRO problem. The function returns the regularization term for the objective.
    """
    if isinstance(theta, cp.Variable):
        return 0.05 * cp.norm(theta, 2)
    else:
        return 0.05 * np.linalg.norm(theta, 2)