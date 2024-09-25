import benchmark_functions as bf
import numpy as np
import bayesian_optimisation as bayopt
from nextorch import plotting, bo, doe, utils, io, parameter


def mean_min_benchmark(bench_func, n_dims, total_samples, n_trials, iterations):
    func = bench_func(n_dimensions=n_dims)
    print("Function is {} ({} dimensions). Total samples in each optimisation process is {} and we are running {} "
          "iterations of this each time we test a certain number of trials.".format(func, n_dims, total_samples, iterations))
    if not isinstance(n_trials, int):
        n_trials = int(round(n_trials))
    print("Testing {} trials.".format(n_trials))
    min_values = []
    for i in range(1, iterations+1):
        params = [parameter.Parameter(x_type="continuous", x_range=[func.suggested_bounds()[0][i], func.suggested_bounds()[1][i]]) for i in range(func.n_dimensions())]
        try:
            opt = bayopt.run_single_obj_experiment(func, params, "LHS", n_init=total_samples-n_trials, n_trials=n_trials)
            if isinstance(opt[1], (int, float, np.float32, np.float64)):
                min_values.append(opt[1])
            else:
                raise ValueError("Not a valid result")
        except Exception as e:
            print("Error: {}".format(e))
    return sum(min_values)/len(min_values)


def choose_mean_min_bench_args(func, n_dims, total_samples, iterations):
    def new_function(n_trials):
        return mean_min_benchmark(func, n_dims, total_samples, n_trials, iterations)
    return new_function
