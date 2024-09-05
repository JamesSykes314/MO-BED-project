import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
results = pd.read_csv('results_df')
plt.hist([results.iloc[0,:], results.iloc[1,:]], label=['LHS', 'BO'], rwidth=0.6)
plt.legend(loc='upper left', framealpha=0.25)
plt.title("Log10(hypervolume)")
plt.show()