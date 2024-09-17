import benchmark_functions as bf
import numpy as np
import gpytorch
import bayesian_optimisation as bayopt
from nextorch import plotting, bo, doe, utils, io, parameter

func = bf.Schwefel(n_dimensions=10)
count = 0
iterations = 10
for i in range(1,iterations+1):
    opt1 = func.minimum_random_search(bounds=func.suggested_bounds(), n_samples=10000)
    params = [parameter.Parameter(x_type="continuous", x_range=[func.suggested_bounds()[0][i], func.suggested_bounds()[1][i]]) for i in range(func.n_dimensions())]
    try:
        opt2 = bayopt.run_single_obj_experiment(func, params, "LHS", n_init=200, n_trials=20)
        print("The minimum is {} (at {}).".format(func.minimum().score, func.minimum().position))
        print(opt1, opt2)
    except:
        print("Error")
    else:
        if opt2[1]<opt1[1]:
            count += 1
    print(i, count)
print("Bayesian Optimisation was better than randomised sampling in {} of the {} experiments.".format(count, iterations))
