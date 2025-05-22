import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_analysis.data_analysis import *


data = load_data('data/data.csv')
correlation_matrix(data)
outliers = detect_outliers(data, )
plot_histogram(data)
plot_boxplot(data)
data = drop_correlated_features(data, threshold=0.8)
data.head()