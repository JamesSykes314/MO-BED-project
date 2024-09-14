import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from nextorch import plotting, bo, doe, utils, io, parameter

def function_numpy_conversion(func):
    def new_func(X):
        try:
            X.shape[1]
        except:
            X = np.array(X)
        if len(X.shape) < 2:
            X = np.expand_dims(X, axis=0)
        y = np.array([], dtype=float)
        for i in range(X.shape[0]):
            if X.shape[1] == 1:
                y_new = func(X[i, 0])
                y = np.append(y, y_new)
            else:
                y_new = func(X[i, :].tolist())
                y = np.append(y, y_new)
        y = y.reshape(X.shape[0],-1)
        return y
    return new_func

def run_single_obj_experiment(objective_function, parameter_list, sampling_method, n_init, n_trials, acq_function="EI", kernel="default", maximise=False):
    start_time = time.time()
    Exp = bo.Experiment('Experiment')
    Exp.define_space(parameter_list)
    objective_func = function_numpy_conversion(objective_function)
    n_dim = len(parameter_list)

    try:
        if sampling_method=="LHS":
            X_init = doe.latin_hypercube(n_dim=n_dim, n_points=n_init)
        elif sampling_method=="randomized_design":
            X_init = doe.randomized_design(n_dim=n_dim, n_points=n_init)
        else:
            raise Exception("Choose either LHS or randomized_design for the sampling method.")
    except:
        return "Experiment failed."

    Y_init = bo.eval_objective_func_encoding(X_init, Exp.parameter_space, objective_func)
    Exp.input_data(X_init,
                   Y_init,
                   unit_flag=True,
                   standardized=False)

    Exp.set_optim_specs(objective_func=objective_func, maximize=maximise)

    opt_check_freq = 20
    opt_array_rows = 2+(n_trials-1)//opt_check_freq
    opt_array = np.array(np.zeros((opt_array_rows, n_dim+1)))
    for i in range(n_trials):
        if i % 20 == 0:
            print("{} trials completed".format(i))
        if i % opt_check_freq == 0:
            Y_optim, X_optim, _ = Exp.get_optim()
            opt_array[i//opt_check_freq, :n_dim] = X_optim
            opt_array[i//opt_check_freq, n_dim] = Y_optim
            print(X_optim, Y_optim)
        # Generate the next experiment point
        X_new, X_new_real, acq_func = Exp.generate_next_point(acq_func_name=acq_function, n_candidates=1)
        # Get the response at this point
        Y_new_real = objective_func(X_new_real)
        # Retrain the model by input the next point into Exp object
        Exp.run_trial(X_new, X_new_real, Y_new_real)

    final_time = time.time()
    time_taken = final_time - start_time
    Y_optim, X_optim, _ = Exp.get_optim()
    opt_array[opt_array_rows-1, :n_dim] = X_optim
    opt_array[opt_array_rows-1, n_dim] = Y_optim
    print(opt_array)
    return X_optim, Y_optim, time_taken
