import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import sys, os

base_directory = os.path.dirname(os.path.abspath(__file__)) 
data_directory = os.path.join(base_directory, '../../')
sys.path.insert(0, data_directory)

from src.data.data_process import  read_data
import matplotlib.pyplot as plt


def plot_matrix():
    # Correlation matrix
    data = read_data()
    # print(data)
    corrmat = data.corr(method='pearson')
    # print(corrmat)
    fig = plt.figure(figsize = (12, 9))
    sns.heatmap(corrmat, vmax = .8, square = True)
    plt.show()
    

# print(plot_matrix())