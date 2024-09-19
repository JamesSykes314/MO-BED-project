import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results = pd.read_csv('results_df')
rows_to_plot = [14, 15]
sampling_methods = ["LHS", "BO"]
transformation_function = lambda x: x

data_1 = (results.iloc[rows_to_plot[0], 3:].dropna()).apply(transformation_function)
data_2 = (results.iloc[rows_to_plot[1], 3:].dropna()).apply(transformation_function)

plt.hist([data_1, data_2], rwidth=0.6, label=[sampling_methods[0], sampling_methods[1]])
plt.legend(loc='upper left', framealpha=0.25)
plt.title("Hypervolume")
plt.show()
# plt.savefig('plot.png', format = 'png')
