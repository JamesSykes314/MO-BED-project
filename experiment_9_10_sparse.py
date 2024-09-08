import os
import sys
import time
from IPython.display import display
project_path = os.getcwd()
sys.path.insert(0, project_path)

# Set the path for objective function
objective_path = os.path.join(project_path, '9in_10out')
sys.path.insert(0, objective_path)

import numpy as np
import pandas as pd
import torch
from nextorch import plotting, bo, doe, utils, io, parameter

def biomarker_inflammatory_response(variables):
    return variables[:, 2] * np.exp(-variables[:, 4] / 100)


def biomarker_oxidative_stress(variables):
    return variables[:, 1] * np.sin(variables[:, 0] / 50) + variables[:, 5]


def biomarker_neuroprotection(variables):
    return np.tanh(variables[:, 7] + variables[:, 8] / 10)


def biomarker_mitochondrial_function(variables):
    return np.sqrt(variables[:, 0]) / (variables[:, 3] + 1)


def biomarker_synaptic_function(variables):
    return np.log1p(variables[:, 6]) * np.exp(-variables[:, 5] / 100)


def biomarker_motor_function(variables):
    return np.sin(variables[:, 0] / 20) * variables[:, 8]


def biomarker_cognitive_function(variables):
    return np.cos(variables[:, 1] / 2) * np.sqrt(variables[:, 8] + 1)


def biomarker_cardiac_function(variables):
    return np.exp(-variables[:, 4] / 100) + np.tanh(variables[:, 2])


def biomarker_liver_function(variables):
    return variables[:, 8] * np.log1p(variables[:, 5])


def biomarker_kidney_function(variables):
    return variables[:, 3] * np.sin(variables[:, 0] / 100) + np.exp(-variables[:, 6])


def compute_biomarkers(X):
    """
    Computes all biomarkers based on the input variables and combines them into a single array.
    """
    X = X.astype(float)
    inflammatory_response = -biomarker_inflammatory_response(X)
    oxidative_stress = -biomarker_oxidative_stress(X)
    neuroprotection = biomarker_neuroprotection(X)
    mitochondrial_function = biomarker_mitochondrial_function(X)
    synaptic_function = biomarker_synaptic_function(X)
    motor_function = biomarker_motor_function(X)
    cognitive_function = biomarker_cognitive_function(X)
    cardiac_function = biomarker_cardiac_function(X)
    liver_function = biomarker_liver_function(X)
    kidney_function = biomarker_kidney_function(X)

    # Combine all the biomarker outputs into a single numpy array
    responses = np.column_stack((
        inflammatory_response,
        oxidative_stress,
        neuroprotection,
        mitochondrial_function,
        synaptic_function,
        motor_function,
        cognitive_function,
        cardiac_function,
        liver_function,
        kidney_function
    ))

    return responses

def run_experiment(n_init, n_trials):
    ##%% Initialize a multi-objective Experiment object
    # Set its name, the files will be saved under the folder with the same name
    start_time = time.time()
    Exp_9_10 = bo.EHVIMOOExperiment('mice_9_10_open')

    # Set the type and range for each parameter
    par_mw = parameter.Parameter(x_type='continuous', x_range=[20, 1000])
    par_logP = parameter.Parameter(x_type='continuous', x_range=[-10, 10])
    par_hbd = parameter.Parameter(x_type='ordinal', x_range=[0, 15], interval=1)
    par_hba = parameter.Parameter(x_type='ordinal', x_range=[0, 20], interval=1)
    par_tpsa = parameter.Parameter(x_type='continuous', x_range=[0, 3000])
    par_mr = parameter.Parameter(x_type='continuous', x_range=[20, 150])
    par_rb = parameter.Parameter(x_type='ordinal', x_range=[0, 15], interval=1)
    par_ar = parameter.Parameter(x_type='ordinal', x_range=[0, 10], interval=1)
    par_ha = parameter.Parameter(x_type='ordinal', x_range=[1, 100], interval=1)

    parameters = [par_mw, par_logP, par_hbd, par_hba, par_tpsa, par_mr, par_rb, par_ar, par_ha]
    Exp_9_10.define_space(parameters)

    objective_func = compute_biomarkers

    # Define the design space
    X_name_list = [
        "Molecular Weight",
        "LogP",
        "Hydrogen Bond Donors",
        "Hydrogen Bond Acceptors",
        "Topological Polar Surface Area",
        "Molar Refractivity",
        "Rotatable Bonds",
        "Aromatic Rings",
        "Heavy Atoms"
    ]

    Y_name_list = [
        "Inflammatory Response",
        "Oxidative Stress",
        "Neuroprotection",
        "Mitochondrial Function",
        "Synaptic Function",
        "Motor Function",
        "Cognitive Function",
        "Cardiac Function",
        "Liver Function",
        "Kidney Function"
    ]

    var_names = X_name_list + Y_name_list

    # Get the information of the design space
    n_dim = len(X_name_list)  # the dimension of inputs
    n_objective = len(Y_name_list)  # the dimension of outputs

    ##%% Initial Sampling
    # Latin hypercube design
    X_init = doe.latin_hypercube(n_dim=n_dim, n_points=n_init)
    # Get the initial responses
    Y_init = bo.eval_objective_func_encoding(X_init, Exp_9_10.parameter_space, objective_func)

    # Import the initial data
    Exp_9_10.input_data(X_init,
                        Y_init,
                        X_names=X_name_list,
                        Y_names=Y_name_list,
                        unit_flag=True,
                        standardized=False)

    # Set the optimization specifications
    ref_point = [-0.8,-90,0.98,4,0.55,-5,-3,0.93,55,0.45]

    Exp_9_10.set_ref_point(ref_point)
    Exp_9_10.set_optim_specs(objective_func=objective_func,
                             maximize=True)

    for i in range(n_trials):
        if i % 20 == 0:
            print("{} trials completed".format(i))
        # Generate the next experiment point
        X_new, X_new_real, acq_func = Exp_9_10.generate_next_point(n_candidates=1)
        # Get the response at this point
        Y_new_real = objective_func(X_new_real)
        # Retrain the model by input the next point into Exp object
        Exp_9_10.run_trial(X_new, X_new_real, Y_new_real)

    # Results:
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    sample_size = Exp_9_10.Y_real.shape[0]
    front_size = Exp_9_10.get_optim()[0].shape[0]
    final_hypervolume_obj = Hypervolume(torch.tensor(ref_point))
    final_hypervolume = final_hypervolume_obj.compute(torch.from_numpy(Exp_9_10.get_optim()[0]))
    end_time = time.time()
    time_taken = end_time - start_time
    return front_size / sample_size, final_hypervolume, time_taken

print(run_experiment(100,100))