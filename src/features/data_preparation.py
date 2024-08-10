import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from src.utils.decorators import timer_decorator, log_decorator, error_handler


class DataPreparation:
    def __init__(self, data):
        self.data = data.copy()
        self.numeric_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        self.categorical_columns = ['Geography', 'Gender']
        self.binary_columns = ['HasCrCard', 'IsActiveMember']

    @timer_decorator
    @error_handler
    @log_decorator
    def remove_irrelevant_columns(self):
        """Elimina columnas que no son relevantes para el análisis si existen."""
        columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        existing_columns = [col for col in columns_to_drop if col in self.data.columns]
        if existing_columns:
            self.data.drop(columns=existing_columns, inplace=True)
        return self

    @timer_decorator
    @error_handler
    @log_decorator
    def handle_missing_values(self):
        """Maneja los valores faltantes en el conjunto de datos."""
        num_imputer = SimpleImputer(strategy='mean')
        self.data[self.numeric_columns] = num_imputer.fit_transform(self.data[self.numeric_columns])

        cat_imputer = SimpleImputer(strategy='most_frequent')
        self.data[self.categorical_columns] = cat_imputer.fit_transform(self.data[self.categorical_columns])

        return self

    @timer_decorator
    @error_handler
    @log_decorator
    def encode_categorical_variables(self):
        """Codifica las variables categóricas."""
        le = LabelEncoder()
        for col in self.categorical_columns:
            self.data[col] = le.fit_transform(self.data[col])
        return self

    @timer_decorator
    @error_handler
    @log_decorator
    def normalize_numeric_features(self):
        """Normaliza las características numéricas."""
        scaler = StandardScaler()
        self.data[self.numeric_columns] = scaler.fit_transform(self.data[self.numeric_columns])
        return self

    @timer_decorator
    @error_handler
    @log_decorator
    def create_binary_flags(self):
        """Crea flags binarios para algunas características."""
        self.data['HasBalance'] = (self.data['Balance'] > 0).astype(int)
        self.data['IsOlderThan40'] = (self.data['Age'] > 40).astype(int)
        self.binary_columns.extend(['HasBalance', 'IsOlderThan40'])
        return self

    @timer_decorator
    @error_handler
    @log_decorator
    def prepare_data(self):
        """Ejecuta todos los pasos de preparación de datos."""
        return (self.remove_irrelevant_columns()
                .handle_missing_values()
                .encode_categorical_variables()
                .normalize_numeric_features()
                .create_binary_flags())

    @timer_decorator
    @error_handler
    @log_decorator
    def get_prepared_data(self):
        """Devuelve los datos preparados."""
        return self.data
