import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pycaret.datasets import get_data
from pycaret.classification import *

class Modeling:
    """
    Data modeling class for data preprocessing and analysis.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    # Load data from a CSV file
    def load_data(self):
        """
        Load data from a CSV file.
        """
        data = pd.read_csv(self.file_path)
        return data
    #initiate data modeling class

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        """
        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size, random_state=random_state)
        return self.train_data, self.test_data
    
    # pycaret
    def pycaret_model(
            self,
            dataset:pd.DataFrame,
            target: str,
            model_selection: str = 'classification',
    )-> None:
        data = dataset
        if model_selection == 'classification':
            classification_setup = setup(data, target=target, session_id=123)
            best_model = compare_models()
            save_model(best_model, 'result/classification')

        elif model_selection == 'regression':
            regression_setup = setup(data, target=target, session_id=123)
            best_model = compare_models()
            save_model(best_model, 'result/regression')
        else:
            raise ValueError("model_selection must be either 'classification' or 'regression'")
        return best_model
    
    def pycaret_evaluate_model(
            self,
            model,
    )-> None:
        """
        Evaluate the model using pycaret.
        """
        eval_ = evaluate_model(model)
        # save the evaluation results
        plt.savefig('result/evaluation.png')

        return eval_
    
    def plot_pycaret_model(
            self,
            model,

    )-> None:
        plot_model(model, plot='auc')
        #save the plot
        plt.savefig('result/auc.png')
        plot_model(model, plot='feature')
        #save the plot
        plt.savefig('result/feature.png')
        plot_model(model, plot='confusion_matrix')
        #save the plot
        plt.savefig('result/confusion_matrix.png')
        plot_model(model, plot='pr')
        #save the plot
        plt.savefig('result/pr.png')
        plot_model(model, plot='learning')
        #save the plot
        plt.savefig('result/learning.png')
        
    

    def pycaret_predict_model(
            self,
            model,
            test_data: pd.DataFrame,

    ):
        prediction = predict_model(model, data=test_data)
        #save the prediction
        prediction.to_csv('result/prediction.csv', index=False)
        return prediction
    
    def pycaret_save_model(
            self,
            model,
    ):
        save_model(model, 'result/model')
        return model
    def pycaret_load_model(
            self,
            model_name: str,):
        model = load_model(model_name)
        return model
    
    def pycaret_compare_models(
            self,
            model_selection: str = 'classification',
    ):
        if model_selection == 'classification':
            best_model = compare_models()
            # save the best model
            save_model(best_model, 'result/classification')
            return best_model
        elif model_selection == 'regression':
            best_model = compare_models()
            # save the best model
            save_model(best_model, 'result/regression')
            return best_model
        else:
            raise ValueError("model_selection must be either 'classification' or 'regression'")
    

    


    




