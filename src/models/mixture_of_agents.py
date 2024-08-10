import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.decorators import timer_decorator, log_decorator, error_handler



class MixtureOfAgents(BaseEstimator, ClassifierMixin):
    def __init__(self, models=None, weights=None, experiment_name="MOA_Experiment", data_dir="../data"):
        self.models = models or []
        self.weights = weights
        self.classes_ = None
        self.experiment_name = experiment_name
        self.data_dir = data_dir
        mlflow.set_experiment(experiment_name)

    @timer_decorator
    @error_handler
    @log_decorator
    def add_model(self, model, weight=1.0):
        """Añade un modelo al conjunto."""
        self.models.append(model)
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)

    @timer_decorator
    @error_handler
    @log_decorator
    def split_and_save_data(self, X, y, test_size=0.2, val_size=0.2):
        """Divide los datos en conjuntos de entrenamiento, validación y prueba, y los guarda."""
        # Primera división: separar el conjunto de prueba
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, stratify=y,
                                                                    random_state=42)

        # Segunda división: separar el conjunto de validación del de entrenamiento
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted,
                                                          stratify=y_train_val, random_state=42)

        # Crear directorios si no existen
        for dir_name in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.data_dir, dir_name), exist_ok=True)

        # Guardar los conjuntos de datos
        pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(self.data_dir, 'train', 'train_data.csv'),
                                                     index=False)
        pd.concat([X_val, y_val], axis=1).to_csv(os.path.join(self.data_dir, 'val', 'val_data.csv'), index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(self.data_dir, 'test', 'test_data.csv'), index=False)

        return X_train, X_val, X_test, y_train, y_val, y_test

    @timer_decorator
    @error_handler
    @log_decorator
    def load_data(self, data_type='train'):
        """Carga los datos del directorio especificado."""
        file_path = os.path.join(self.data_dir, data_type, f'{data_type}_data.csv')
        data = pd.read_csv(file_path)
        y = data.pop('target')  # Asumimos que la columna objetivo se llama 'target'
        X = data
        return X, y

    @timer_decorator
    @error_handler
    @log_decorator
    def fit(self, X, y):
        """Entrena todos los modelos en el conjunto."""
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        with mlflow.start_run(run_name="MOA_Training") as run:
            for i, model in enumerate(self.models):
                with mlflow.start_run(run_name=f"Model_{i}_Training", nested=True):
                    model.fit(X, y)
                    mlflow.log_param(f"model_{i}", type(model).__name__)
                    mlflow.log_param(f"weight_{i}", self.weights[i])

            mlflow.log_param("n_models", len(self.models))

            # Guardar el modelo completo
            mlflow.sklearn.log_model(self, "moa_model")

            # Registrar el modelo en el Registro de Modelos de MLflow
            model_uri = f"runs:/{run.info.run_id}/moa_model"
            mv = mlflow.register_model(model_uri, "MixtureOfAgents")

            # Transicionar el modelo a 'Production' si es la mejor versión
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name="MixtureOfAgents",
                version=mv.version,
                stage="Production"
            )

        return self

    @timer_decorator
    @error_handler
    @log_decorator
    def predict_proba(self, X):
        """Predice las probabilidades de clase para X."""
        check_is_fitted(self)
        X = check_array(X)

        predictions = []
        for model, weight in zip(self.models, self.weights):
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)
            else:
                pred = np.eye(len(self.classes_))[model.predict(X)]
            predictions.append(weight * pred)

        return np.sum(predictions, axis=0) / np.sum(self.weights)

    @timer_decorator
    @error_handler
    @log_decorator
    def predict(self, X):
        """Predice las etiquetas de clase para X."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @timer_decorator
    @error_handler
    @log_decorator
    def evaluate(self, X, y):
        """Evalúa el rendimiento del modelo."""
        y_pred = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }

        with mlflow.start_run(run_name="MOA_Evaluation"):
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

        return metrics

    def save(self, path):
        """Guarda el modelo usando MLflow"""
        mlflow.sklearn.save_model(self, path)

    @classmethod
    def load(cls, path):
        """Carga el modelo usando MLflow"""
        return mlflow.sklearn.load_model(path)