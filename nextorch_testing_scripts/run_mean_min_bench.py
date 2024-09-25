import single_obj_benchmarks as sob
import bayesian_optimisation as bayopt
import benchmark_functions as bf
from nextorch import parameter
import argparse


parser = argparse.ArgumentParser(description='Perform BO to test impact of number of BO trials on BO performance.')
parser.add_argument('--bench_func', type=str, required=True, help='Benchmark function being used for testing.')
parser.add_argument('--n_dims', type=int, required=True, help='Number of inputs of benchmark function')
parser.add_argument('--total_samples', type=int, required=True, help='Total number of samples being tested')
args = parser.parse_args()

desired_func = getattr(bf, args.bench_func)
iterations = int(round(10000 / args.total_samples))
run_bayes_opt_benchmark_fns = sob.choose_mean_min_bench_args(desired_func, args.n_dims, args.total_samples, iterations)
param = parameter.Parameter(x_type="ordinal", x_range=[0, args.total_samples - 4], interval=1)
experiment_name = f"{args.bench_func}_{args.n_dims}_{args.total_samples}_{iterations}"
print(bayopt.run_single_obj_experiment(run_bayes_opt_benchmark_fns,
                                       parameter_list=[param], sampling_method="LHS", n_init=15, n_trials=10,
                                       plotting_flag=False, save_final_fig=True, exp_name=experiment_name))
