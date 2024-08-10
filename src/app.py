import os
import sys

# Añade el directorio raíz del proyecto al sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import joblib
from src.models.prediction import Prediction
from src.models.mixture_of_agents import MixtureOfAgents
from src.features.data_preparation import DataPreparation
from src.models.together_ai import TogetherAIIntegration
from src.utils.logging_config import setup_logging

# Configuración de logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Cargar o entrenar el modelo
@st.cache_resource
def load_or_train_model():
    # Obtener la ruta del directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construir las rutas relativas al directorio actual
    model_path = os.path.join(current_dir, "..", "models", "moa_model_version_1")
    data_prep_path = os.path.join(current_dir, "..", "models", "data_preparation.pkl")

    logger.info(f"Intentando cargar el modelo desde: {model_path}")
    logger.info(f"Intentando cargar el preprocesador desde: {data_prep_path}")

    if os.path.exists(model_path) and os.path.exists(data_prep_path):
        try:
            return Prediction(model_path, data_prep_path)
        except Exception as e:
            st.warning(f"Error al cargar el modelo existente: {str(e)}. Se entrenará un nuevo modelo.")
            logger.warning(f"Error al cargar el modelo existente: {str(e)}. Se entrenará un nuevo modelo.")
    else:
        st.warning("No se encontró un modelo existente. Se entrenará un nuevo modelo.")
        logger.warning("No se encontró un modelo existente. Se entrenará un nuevo modelo.")

    # Si no se pudo cargar el modelo, entrenar uno nuevo
    return train_new_model()


def train_new_model():
    # Cargar y preparar los datos
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw_data",
                             "Churn_Modelling.csv")
    if not os.path.exists(data_path):
        st.error(f"No se encontró el archivo de datos en: {data_path}")
        logger.error(f"No se encontró el archivo de datos en: {data_path}")
        return None

    data = pd.read_csv(data_path)
    data_prep = DataPreparation(data)
    prepared_data = data_prep.prepare_data().get_prepared_data()

    # Entrenar el modelo
    moa = MixtureOfAgents()
    X = prepared_data.drop('Exited', axis=1)
    y = prepared_data['Exited']
    moa.fit(X, y)

    # Guardar el modelo y el preprocesador
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "moa_model_new")
    data_prep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models",
                                  "data_preparation_new.pkl")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    mlflow.sklearn.save_model(moa, model_path)
    joblib.dump(data_prep, data_prep_path)

    return Prediction(model_path, data_prep_path)


# Inicializar TogetherAI
together_ai = TogetherAIIntegration()

# Cargar o entrenar el modelo
prediction_model = load_or_train_model()

st.title('Predicción de Abandono de Clientes y Asistente IA')

if prediction_model is None:
    st.error("No se pudo cargar ni entrenar el modelo. Por favor, verifica los archivos de datos y las rutas.")
else:
    # Crear inputs para las características del cliente
    st.header('Ingrese los datos del cliente:')

    credit_score = st.slider('Puntuación de Crédito', 300, 850, 600)
    geography = st.selectbox('Geografía', ['France', 'Spain', 'Germany'])
    gender = st.selectbox('Género', ['Male', 'Female'])
    age = st.slider('Edad', 18, 100, 30)
    tenure = st.slider('Antigüedad (años)', 0, 10, 5)
    balance = st.number_input('Balance', 0.0, 250000.0, 0.0)
    num_products = st.slider('Número de Productos', 1, 4, 1)
    has_credit_card = st.checkbox('¿Tiene Tarjeta de Crédito?')
    is_active_member = st.checkbox('¿Es Miembro Activo?')
    estimated_salary = st.number_input('Salario Estimado', 0.0, 250000.0, 50000.0)

    if st.button('Predecir'):
        # Preparar los datos para la predicción
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Geography': [geography],
            'Gender': [gender],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_products],
            'HasCrCard': [int(has_credit_card)],
            'IsActiveMember': [int(is_active_member)],
            'EstimatedSalary': [estimated_salary]
        })

        try:
            # Hacer la predicción
            prediction = prediction_model.predict(input_data)
            probabilities = prediction_model.predict_proba(input_data)

            # Mostrar resultados
            st.subheader('Resultado de la Predicción:')
            if prediction[0] == 1:
                st.write('El cliente tiene una alta probabilidad de abandonar.')
            else:
                st.write('El cliente tiene una baja probabilidad de abandonar.')

            st.write(f'Probabilidad de abandono: {probabilities[0][1]:.2f}')

            # Generar interpretación con LLM
            interpretation = together_ai.interpret_results(prediction[0], probabilities[0], input_data.iloc[0])
            st.subheader('Interpretación del Asistente IA:')
            st.write(interpretation)
        except Exception as e:
            st.error(f"Error al realizar la predicción: {str(e)}")
            logger.exception("Error al realizar la predicción")

    # Área para preguntas adicionales
    st.header('Preguntas Adicionales:')
    user_question = st.text_input('Haga una pregunta sobre el cliente o la predicción:')

    if user_question:
        try:
            # Generar respuesta con LLM
            llm_response = together_ai.generate_response(user_question, input_data.iloc[0], prediction[0],
                                                         probabilities[0])
            st.subheader('Respuesta del Asistente IA:')
            st.write(llm_response)
        except Exception as e:
            st.error(f"Error al generar la respuesta: {str(e)}")
            logger.exception("Error al generar la respuesta")

st.sidebar.info(
    'Esta aplicación predice la probabilidad de que un cliente abandone el banco y proporciona interpretaciones basadas en IA.')