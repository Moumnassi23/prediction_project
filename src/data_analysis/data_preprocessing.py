import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



class DataPreprocessor:
    """
    A class for data preprocessing tasks including loading data, plotting histograms,
    detecting outliers, and plotting boxplots.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def impute__numerical_features_nulls(
            self,
            method='mean',
            knn_k=None,
    )->pd.DataFrame:
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        # Impute numeric columns
        if method not in ["mean", "median", "mode", "drop", "knn"]:
            raise ValueError("Method must be one of ['mean', 'median', 'mode', 'drop', 'knn']")
        
        for col in numeric_cols :
            if method == "mean":
                self.data[col].fillna(self.data[col].mean(), inplace=True)
            elif method == "median":
                self.data[col].fillna(self.data[col].median(), inplace=True)
            elif method == "mode":
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
            elif method == "drop":
                self.data.drop(col, axis=1, inplace=True)
            elif method == "knn":
                if knn_k is None:
                    raise ValueError("k must be specified for KNN imputation")
                imputer = KNNImputer(n_neighbors=knn_k)
                self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
        return self.data
    
    def impute_categorical_features_nulls(
            self,
            method='mode',
    )->pd.DataFrame:
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        # Impute categorical columns
        if method not in ["mode", "drop", "knn"]:
            raise ValueError("Method must be one of ['mode', 'drop']")
        
        for col in categorical_cols:
            if method == "mode":
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
            elif method == "drop":
                self.data.drop(col, axis=1, inplace=True)
            elif method == "knn":
                imputer = KNNImputer(n_neighbors=5)
                self.data[categorical_cols] = imputer.fit_transform(self.data[categorical_cols])
        return self.data
    
    def encode_categorial_features(
            self,
            method='onehot',
    )->pd.DataFrame:
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        # Encode categorical columns
        if method not in ["onehot", "label"]:
            raise ValueError("Method must be one of ['onehot', 'label']")
        
        for col in categorical_cols:
            if method == "onehot":
                self.data = pd.get_dummies(self.data, columns=[col], drop_first=True)
            elif method == "label":
                self.data[col] = self.data[col].astype('category').cat.codes
        return self.data
    
    def scale_data(
            self,
            method: str = 'standard',
      )-> pd.DataFrame:
        if method not in ["standard", "minmax"]:
            raise ValueError("Method must be one of ['standard', 'minmax']")
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
        return self.data
    
    
    

