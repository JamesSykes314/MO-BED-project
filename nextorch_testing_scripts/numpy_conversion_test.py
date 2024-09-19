import bayesian_optimisation as bayopt
import numpy as np


def squared(x):
    return x ** 2


sq = bayopt.function_numpy_conversion(squared)


print(sq(np.array([[1], [2]])))
