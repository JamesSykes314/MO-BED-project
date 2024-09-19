from bayesian_optimisation import run_single_obj_experiment
import math
from nextorch import plotting, bo, doe, utils, io, parameter

def simple_nonlinear(x):
    return (6*x - 2)**2 * math.sin(12*x - 4)

param = parameter.Parameter(x_type="continuous", x_range=[0, 1])
print(run_single_obj_experiment(simple_nonlinear, parameter_list=[param], sampling_method="LHS", n_init=5, n_trials=20, plotting_flag=True, save_fig_flag=True))