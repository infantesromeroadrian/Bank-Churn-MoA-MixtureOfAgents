import pytest
import pandas as pd
import numpy as np
import mlflow
from src.models.prediction import Prediction
from src.models.mixture_of_agents import MixtureOfAgents
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def sample_model_and_data(tmp_path):
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    moa = MixtureOfAgents()
    moa.add_model(RandomForestClassifier(n_estimators=10, random_state=42))
    moa.add_model(LogisticRegression(random_state=42))
    moa.fit(X_train, y_train)

    model_path = tmp_path / "test_model"
    moa.save(model_path)

    return str(model_path), X_test, y_test


def test_prediction(sample_model_and_data):
    model_path, X_test, y_test = sample_model_and_data

    prediction = Prediction(model_path)

    # Crear el DataFrame con los nombres de columna originales
    X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])

    # Mapear las columnas a los nombres esperados por el modelo
    column_mapping = {
        'feature_0': 'CreditScore',
        'feature_1': 'Geography',
        'feature_2': 'Gender',
        'feature_3': 'Age',
        'feature_4': 'Tenure',
        'feature_5': 'Balance',
        'feature_6': 'NumOfProducts',
        'feature_7': 'HasCrCard',
        'feature_8': 'IsActiveMember',
        'feature_9': 'EstimatedSalary'
    }
    X_test_df = X_test_df.rename(columns=column_mapping)

    # Añadir las columnas calculadas
    X_test_df['HasBalance'] = (X_test_df['Balance'] > 0).astype(int)
    X_test_df['IsOlderThan40'] = (X_test_df['Age'] > 40).astype(int)

    # Asegurarse de que todas las columnas originales estén presentes
    for i in range(10, 20):
        if f'feature_{i}' not in X_test_df.columns:
            X_test_df[f'feature_{i}'] = X_test_df[f'feature_{i - 10}']

    predictions = prediction.predict(X_test_df)
    probabilities = prediction.predict_proba(X_test_df)

    assert len(predictions) == len(y_test)
    assert probabilities.shape == (len(y_test), 2)

    sample_data = X_test_df.iloc[:1]
    result = prediction.predict_and_explain(sample_data)

    assert isinstance(result[0], str)
    assert "Predicción:" in result[0]
    assert "Probabilidad de Abandono:" in result[0]