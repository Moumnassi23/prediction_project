import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_analysis.data_analysis import *
from src.data_analysis.data_preprocessing import DataPreprocessor
from src.data_modeling.data_modeling import Modeling

model = Modeling("data/data.csv")
data = model.load_data()
model_1 = model.pycaret_model(data, target="label", model_selection="classification")
model_eval = model.pycaret_evaluate_model(model_1)
model.plot_pycaret_model(model_1)
model.pycaret_compare_models(model_1)
model.pycaret_predict_model(model_1)
model.pycaret_save_model(model_1)
