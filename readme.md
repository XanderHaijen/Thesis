# Finite support CADRO project

This project is an implementation of the CADRO algorithm for bounded and real-valued disturbances. It is a generalization
of the algorithm by Schuurmans & Patrinos (2023), who developed it for finite and discrete distributions. This code was
developed in the context of a Master's thesis at the STADIUS group of the KU Leuven (Haijen, 2024). This code is specifically for the
case of linear least squares regression, since this allows for an implementation as a convex Semidefinite Program (SDP).

## Project structure
The main project directory (`code/finite_support_cadro`) contains the following files and subdirectories:

**Subdirectories**:
- `utils`: contains several auxiliary functions used in the implementation of the CADRO algorithm and in experiments.

**Algorithm files**:
- `continuous_cadro.py`: contains the general implementation of the CADRO algorithm for continuous disturbances.
- `ellipsoids.py`: contains the implementation of the ellipsoidal support set and several functions to generate one.
- `multiple_dimension_cadro.py`: contains the implementation of the CADRO algorithm for multivariate linear regression
- `multiple_guess_cadro.py`: contains the implementation of the CADRO algorithm for multivariate linear regression with
  multiple guesses for the initial decision variable estimate.
- `one_dimension_cadro.py`: contains the implementation of the CADRO algorithm for univariate linear regression.
- `robust_optimization.py`: contains the implementation of the robust optimization problem over ellipsoidal support sets.
- `sample_cadro.py`: contains the implementation of the CADRO algorithm where the support set is discretized.
- `stochastic_dominance_cadro.py`: contains the implementation of the CADRO algorithm for linear regression, where
the ambiguity set is defined by stochastic dominance constraints on the cdf of the cost function.

**Experimentation files**
- `1d_linreg_experiments.py`: contains the experiments for the univariate linear regression case.
- `multiple_d_linreg_experiments.py`: contains the experiments for the multivariate linear regression case.
- `sample_cadro_experiments.py`: contains the experiments for the discretized support set case.
- `stochastic_dominance_experiments.py`: contains the experiments for the stochastic dominance case.

**Other files**

- `cadro_test.py`: contains the tests for the CADRO algorithm.
- `moment_dro.py`: contains the implementation of the moment-based distributionally robust optimization problem which
  serves as a benchmark for the CADRO algorithm.
- `study_minimal.py` and `study_minimal_divergence.py`: two files that contain some code used to calibrate the
ambiguity set of the CADRO algorithm using ordered mean bounds. See Coppens & Patrinos (2023) for more details.

## Use
The CADRO algorithm can be used by downloading the code and running the desired algorithm file. The algorithm files
contain python classes which can be used like any other python class. The experiment files provide plenty of examples
on how to use the algorithms.

## Dependencies
The code was developed using Python 3.8 and several libraries. The main dependencies for the algorithms are
`numpy` and `cvxpy`. For the algorithms, we use the MOSEK solver, which is a commercial solver. The code can be run
using other solvers, but the performance might be worse. The experiments also use `matplotlib` and `pandas`.

## Copyright
**© Copyright KU Leuven**

Without written permission of the supervisor and the author it is forbidden to
reproduce or adapt in any form or by any means any part of this repository.
Requests for obtaining the right to reproduce or utilize parts of this publication
should be addressed to the Departement Computerwetenschappen, Celestijnenlaan
200A bus 2402, B-3001 Leuven, +32-16-327700 or by email info@cs.kuleuven.be.

A written permission of the supervisor is also required to use the methods, products,
schematics and programmes described in this repository for industrial or commercial use,
and for submitting this publication in scientific contests.



## References
- Schuurmans, M., & Patrinos, P. (2023). Distributionally Robust Optimization Using Cost-Aware Ambiguity Sets. _IEEE Control Systems Letters_, 7, 1855–1860. https://doi.org/10.1109/LCSYS.2023.3281974
- Coppens, P., & Patrinos, P. (2023). Ordered Risk Minimization: Learning More from Less Data. _2023 62nd IEEE Conference on Decision and Control (CDC)_, 2003–2009. https://doi.org/10.1109/CDC49753.2023.10383728
- Haijen, X. (2024). Cost-aware distributionally robust optimization. [Unpublished Master's thesis]. KU Leuven, Belgium.
```