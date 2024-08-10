import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from src.utils.decorators import timer_decorator, log_decorator, error_handler
from src.utils.logging_config import setup_logging
from src.features.load_data import BankData
from src.features.data_preparation import DataPreparation
from src.models.mixture_of_agents import MixtureOfAgents
from src.models.prediction import Prediction
from src.features.data_analysis import ExploratoryDataAnalysis
from src.models.together_ai import TogetherAIIntegration
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import mlflow

import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logger = setup_logging()


def get_next_version(base_path):
    """Determina el siguiente número de versión basado en los directorios existentes."""
    existing_versions = [d for d in os.listdir(base_path) if d.startswith("version_")]
    if not existing_versions:
        return 1
    return max(int(v.split("_")[1]) for v in existing_versions) + 1


def find_data_file(file_name, search_paths):
    """Busca el archivo de datos en múltiples ubicaciones posibles."""
    for path in search_paths:
        full_path = os.path.join(path, file_name)
        if os.path.exists(full_path):
            return full_path
    return None


def main():
    # Definir posibles ubicaciones para el archivo de datos
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    search_paths = [
        current_dir,
        parent_dir,
        os.path.join(parent_dir, 'data'),
        os.path.join(parent_dir, 'data', 'raw_data'),
        os.path.join(current_dir, 'data'),
        os.path.join(current_dir, 'data', 'raw_data')
    ]

    # Buscar el archivo de datos
    data_file_name = "Churn_Modelling.csv"
    data_file = find_data_file(data_file_name, search_paths)

    if data_file is None:
        logger.error(f"No se pudo encontrar el archivo '{data_file_name}'.")
        logger.info("Por favor, asegúrate de que el archivo de datos esté en una de las siguientes ubicaciones:")
        for path in search_paths:
            logger.info(f"  - {path}")
        sys.exit(1)

    logger.info(f"Archivo de datos encontrado en: {data_file}")

    # Cargar y preparar los datos
    bank_data = BankData(data_file)
    try:
        bank_data.load_data()
    except Exception as e:
        logger.error(f"Error al cargar los datos: {e}")
        sys.exit(1)

    if bank_data.data is None:
        logger.error("Los datos no se cargaron correctamente.")
        sys.exit(1)

    data_prep = DataPreparation(bank_data.data)
    prepared_data = data_prep.prepare_data().get_prepared_data()

    # Realizar análisis exploratorio de datos
    eda = ExploratoryDataAnalysis(prepared_data)
    eda.plot_histograms(['Age', 'Balance', 'EstimatedSalary'])
    eda.plot_correlation_matrix()

    # Crear y entrenar el modelo MixtureOfAgents
    moa = MixtureOfAgents()
    moa.add_model(RandomForestClassifier(n_estimators=100, random_state=42))
    moa.add_model(LogisticRegression(random_state=42))
    moa.add_model(SVC(probability=True, random_state=42))

    X = prepared_data.drop('Exited', axis=1)
    y = prepared_data['Exited']

    X_train, X_val, X_test, y_train, y_val, y_test = moa.split_and_save_data(X, y)

    moa.fit(X_train, y_train)

    # Evaluar el modelo
    val_metrics = moa.evaluate(X_val, y_val)
    logger.info(f"Validation evaluation: {val_metrics}")

    test_metrics = moa.evaluate(X_test, y_test)
    logger.info(f"Test evaluation: {test_metrics}")

    # Guardar el modelo con control de versiones
    base_model_path = os.path.join(parent_dir, "models", "moa_model")
    next_version = get_next_version(os.path.dirname(base_model_path))
    model_path = f"{base_model_path}_version_{next_version}"

    os.makedirs(model_path, exist_ok=True)
    mlflow.sklearn.save_model(moa, model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()