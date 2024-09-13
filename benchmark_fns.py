import benchmark_functions as bf
import numpy as np
import gpytorch
import bayesian_optimisation as bayopt
from nextorch import plotting, bo, doe, utils, io, parameter

func = bf.Schwefel(n_dimensions=2)
count = 0
for i in range(1,11):
    opt1 = func.minimum_random_search(bounds=func.suggested_bounds(), n_samples=10000)
    params = [parameter.Parameter(x_type="continuous", x_range=[func.suggested_bounds()[0][i], func.suggested_bounds()[1][i]]) for i in range(func.n_dimensions())]
    try:
        opt2 = bayopt.run_single_obj_experiment(func, params, "LHS", n_init=100, n_trials=400)
    except:
        print("Error")
    else:
        if opt2[1]<opt1[1]:
            count += 1
    print(i, count)
print(count)
