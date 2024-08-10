import pytest
import pandas as pd
from src.models.prediction import Prediction
from src.models.mixture_of_agents import MixtureOfAgents
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@pytest.fixture
def sample_model_and_data(tmp_path):
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    moa = MixtureOfAgents()
    moa.fit(X_train, y_train)

    model_path = tmp_path / "test_model"
    moa.save(model_path)

    return model_path, X_test, y_test


def test_prediction(sample_model_and_data):
    model_path, X_test, y_test = sample_model_and_data

    prediction = Prediction(model_path, None)  # Assuming no data_prep is needed for this test

    predictions = prediction.predict(X_test)
    probabilities = prediction.predict_proba(X_test)

    assert len(predictions) == len(y_test)
    assert probabilities.shape == (len(y_test), 2)

    # Test prediction with explanation
    sample_data = pd.DataFrame(X_test[:1])  # Take first sample
    result = prediction.predict_and_explain(sample_data)

    assert isinstance(result[0], str)  # Explanation should be a string
    assert "Predicción:" in result[0]
    assert "Probabilidad de Abandono:" in result[0]