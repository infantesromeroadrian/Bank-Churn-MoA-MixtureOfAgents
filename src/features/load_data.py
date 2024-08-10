import pandas as pd
from src.utils.decorators import timer_decorator, log_decorator, error_handler

class BankData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    @timer_decorator
    @error_handler
    @log_decorator
    def load_data(self):
        """Carga los datos del archivo CSV."""
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Datos cargados exitosamente. Shape: {self.data.shape}")
        except Exception as e:
            print(f"Error al cargar los datos: {e}")

    @timer_decorator
    @error_handler
    @log_decorator
    def get_info(self):
        """Obtiene información básica del dataset."""
        if self.data is not None:
            print(self.data.info())
        else:
            print("Los datos no han sido cargados aún.")

    @timer_decorator
    @error_handler
    @log_decorator
    def check_nulls(self):
        """Verifica y muestra la cantidad de valores nulos por columna."""
        if self.data is not None:
            null_counts = self.data.isnull().sum()
            print("Conteo de valores nulos por columna:")
            print(null_counts)
        else:
            print("Los datos no han sido cargados aún.")

    @timer_decorator
    @error_handler
    @log_decorator
    def handle_nulls(self, strategy='drop'):
        """
        Maneja los valores nulos en el dataset.

        :param strategy: Estrategia para manejar nulos ('drop' o 'impute')
        """
        if self.data is not None:
            if strategy == 'drop':
                self.data.dropna(inplace=True)
                print("Filas con valores nulos eliminadas.")
            elif strategy == 'impute':
                for column in self.data.columns:
                    if self.data[column].dtype == 'object':
                        self.data[column].fillna(self.data[column].mode()[0], inplace=True)
                    else:
                        self.data[column].fillna(self.data[column].mean(), inplace=True)
                print("Valores nulos imputados.")
            else:
                print("Estrategia no reconocida. Use 'drop' o 'impute'.")
        else:
            print("Los datos no han sido cargados aún.")