import pytest
from src.models.mixture_of_agents import MixtureOfAgents
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_mixture_of_agents(sample_data):
    X_train, X_test, y_train, y_test = sample_data

    moa = MixtureOfAgents()
    moa.add_model(RandomForestClassifier(n_estimators=100, random_state=42))
    moa.add_model(LogisticRegression(random_state=42))

    moa.fit(X_train, y_train)

    predictions = moa.predict(X_test)
    probabilities = moa.predict_proba(X_test)

    assert len(predictions) == len(y_test)
    assert probabilities.shape == (len(y_test), 2)
    assert moa.evaluate(X_test, y_test)['accuracy'] > 0.8  # Assuming the model performs well