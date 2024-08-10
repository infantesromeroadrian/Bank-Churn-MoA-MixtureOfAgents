import sys
import os
from pathlib import Path

# Obtén la ruta absoluta del directorio raíz del proyecto
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"

# Añade el directorio 'src' al Python path
sys.path.insert(0, str(src_path))

# Imprime los paths para depuración
print("Python path:")
for path in sys.path:
    print(path)