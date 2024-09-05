import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv('CSV_results/results_df_1')
rows_to_plot = [0,1]
sampling_methods = ["LHS", "BO"]

plt.hist([results.iloc[rows_to_plot[0],3:].dropna(), results.iloc[rows_to_plot[1],3:].dropna()], rwidth=0.6, label=[sampling_methods[0], sampling_methods[1]])
plt.legend(loc='upper left', framealpha=0.25)
plt.title("Log10(hypervolume)")
# plt.show()
plt.savefig('plot.png', format = 'png')