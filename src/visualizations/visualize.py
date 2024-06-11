import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import sys, os
import numpy as np

base_directory = os.path.dirname(os.path.abspath(__file__)) 
data_directory = os.path.join(base_directory, '../../')
sys.path.insert(0, data_directory)

from src.data.data_process import  read_data
import matplotlib.pyplot as plt


def plot_matrix():
    # Correlation matrix
    data = read_data()
    corrmat = data.corr(method='pearson')
    fig = plt.figure(figsize = (12, 9))
    sns.heatmap(corrmat, vmax = .8, square = True)
    plt.show()

def plot_results(results, metrics, title, filename):
    models = list(results.keys())
    values = np.array([list(metrics_dict.values()) for metrics_dict in results.values()])
    fig, axs = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    for i, metric in enumerate(metrics):
        axs[i].bar(models, values[:, i], color='skyblue')
        axs[i].set_title(metric)
        axs[i].set_xticklabels(models, rotation=45, ha='right')
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.show()
    
