# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy pyproject.toml and poetry.lock (if it exists)
COPY pyproject.toml poetry.lock* /app/

# Install Poetry
RUN pip install poetry

# Install project dependencies
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application
CMD ["poetry", "run", "python", "app.py"]