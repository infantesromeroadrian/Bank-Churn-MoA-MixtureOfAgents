import pytest
from src.features.data_preparation import DataPreparation
import pandas as pd


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'CreditScore': [600, 700],
        'Geography': ['France', 'Spain'],
        'Gender': ['Male', 'Female'],
        'Age': [30, 40],
        'Tenure': [5, 10],
        'Balance': [50000, 75000],
        'NumOfProducts': [1, 2],
        'HasCrCard': [1, 0],
        'IsActiveMember': [1, 1],
        'EstimatedSalary': [60000, 80000],
        'Exited': [0, 1]
    })


def test_data_preparation(sample_data):
    data_prep = DataPreparation(sample_data)
    prepared_data = data_prep.prepare_data().get_prepared_data()

    assert 'CreditScore' in prepared_data.columns
    assert 'Geography' in prepared_data.columns
    assert 'Gender' in prepared_data.columns
    assert 'Exited' in prepared_data.columns
    assert 'HasBalance' in prepared_data.columns
    assert 'IsOlderThan40' in prepared_data.columns
    assert prepared_data.shape[0] == 2  # Ensure no rows were dropped

    # Check if numerical encoding is applied
    assert prepared_data['Geography'].dtype in ['int64', 'float64']
    assert prepared_data['Gender'].dtype in ['int64', 'float64']

    # Print prepared_data for debugging
    print(prepared_data)