import benchmark_functions as bf
import numpy as np
import gpytorch
import statistics
import bayesian_optimisation as bayopt
from nextorch import plotting, bo, doe, utils, io, parameter

def run_bayes_opt_benchmark_fns(n_trials):
    func = bf.Ackley(n_dimensions=4)
    total_samples = 100
    iterations = 30
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
    return statistics.fmean(min_values)


param = parameter.Parameter(x_type="ordinal", x_range=[0, 96], interval=1)
print(bayopt.run_single_obj_experiment(run_bayes_opt_benchmark_fns, parameter_list=[param], sampling_method="LHS", n_init=15, n_trials=10, plotting_flag=True, save_fig_flag=True))
