import pandas as pd
import mlflow
import mlflow.pyfunc
import joblib
import numpy as np


class Prediction:
    def __init__(self, model_path, data_prep_path):
        self.model = self.load_model(model_path)
        self.data_prep = self.load_data_prep(data_prep_path)
        self.expected_columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                                 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                                 'HasBalance', 'IsOlderThan40']

    @staticmethod
    def load_model(model_path):
        return mlflow.pyfunc.load_model(model_path)

    @staticmethod
    def load_data_prep(data_prep_path):
        return joblib.load(data_prep_path)

    def prepare_data(self, data):
        """Prepara los datos para la predicción."""
        data_prep_instance = self.data_prep.__class__(data)
        prepared_data = data_prep_instance.prepare_data().get_prepared_data()

        # Asegurarse de que todas las columnas esperadas estén presentes
        for col in self.expected_columns:
            if col not in prepared_data.columns:
                if col == 'HasBalance':
                    prepared_data[col] = (prepared_data['Balance'] > 0).astype(int)
                elif col == 'IsOlderThan40':
                    prepared_data[col] = (prepared_data['Age'] > 40).astype(int)
                else:
                    raise ValueError(f"Columna esperada '{col}' no está presente en los datos preparados")

        return prepared_data[self.expected_columns]

    def predict(self, data):
        """Realiza predicciones usando el modelo MOA."""
        prepared_data = self.prepare_data(data)
        return self.model.predict(prepared_data)

    def predict_proba(self, data):
        """Realiza predicciones de probabilidad usando el modelo MOA."""
        prepared_data = self.prepare_data(data)
        proba = self.model.predict(prepared_data)
        # Asegurarse de que proba sea un array 2D
        if proba.ndim == 1:
            proba = np.column_stack((1 - proba, proba))
        return proba

    def predict_and_explain(self, data):
        """Realiza predicciones y proporciona explicaciones básicas."""
        predictions = self.predict(data)
        probabilities = self.predict_proba(data)

        results = []
        for i, pred in enumerate(predictions):
            explanation = f"Predicción: {'Abandono' if pred == 1 else 'No Abandono'}"
            explanation += f"\nProbabilidad de Abandono: {probabilities[i][1]:.2f}"
            explanation += f"\nProbabilidad de No Abandono: {probabilities[i][0]:.2f}"
            results.append(explanation)

        return results
