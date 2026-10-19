import pandas as pd
import os

def limpiar_datos():
    # 1. Cargar dataset crudo
    ruta_entrada = "datos/datos_ini/dataset_raw.csv" # Ajusta el nombre si tu archivo es distinto
    print(f"Cargando datos desde {ruta_entrada}...")
    df = pd.read_csv(ruta_entrada)
    
    # 2. Procesamiento y limpieza (ejemplos)
    # - Manejo de nulos
    df = df.dropna() # O rellenar con .fillna() según tu lógica
    
    # - Transformaciones / Codificación si aplica
    
    # 3. Guardar dataset limpio
    os.makedirs("datos/datos_limp", exist_ok=True)
    ruta_salida = "datos/datos_limp/dataset_clean.csv"
    df.to_csv(ruta_salida, index=False)
    print(f"Dataset limpio guardado exitosamente en {ruta_salida}")

if __name__ == "__main__":
    limpiar_datos()