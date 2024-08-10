# Usa una imagen base de Python
FROM python:3.12-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Instala las dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instala Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copia los archivos de configuración de Poetry
COPY pyproject.toml poetry.lock* /app/

# Instala las dependencias del proyecto
RUN poetry install --no-root --no-dev

# Copia el resto del código de la aplicación
COPY . /app

# Establece la variable de entorno para MLflow
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Expone el puerto en el que se ejecutará la aplicación Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["poetry", "run", "streamlit", "run", "src/app.py"]