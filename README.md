# T07175440 Gestión de proyectos IA Actividad 5: Pipeline MLOps y Modelado Predictivo

Repositorio oficial del proyecto MLOps para la gestión, limpieza, entrenamiento y trazabilidad de modelos de Machine Learning.

---

## 1. Descripción del Dataset
* **Fuente:** Los datos iniciales provienen de `datos/datos_ini/dataset_raw.csv` (recolectados de repositorio/fuente institucional del proyecto).
* **Diccionario de datos:**
  * `feature_1` / `feature_n`: Variables independientes de entrada (numéricas y categóricas).
  * `target`: Variable objetivo que define la clase o etiqueta a predecir.
* **Supuestos:** Se asume que los registros con valores nulos extremos o fuera de rango lógico representan anomalías del sistema original y deben ser filtrados o imputados.

## 2. Proceso de Limpieza y Estandarización
Implementado a través del script `fuentes/datos_prep.py`:
* **Manejo de nulos:** Imputación de valores faltantes mediante medidas de tendencia central o eliminación selectiva de filas críticas.
* **Tipos de datos:** Conversión estricta de formatos de texto y numéricos para asegurar compatibilidad con scikit-learn.
* **Escalado y Normalización:** Estandarización de variables numéricas mediante `StandardScaler`.
* **Codificación de categóricas:** Aplicación de *One-Hot Encoding* para variables nominales.
* **Salida:** El dataset procesado se almacena de forma automatizada en `datos/datos_limp/dataset_clean.csv`.

## 3. Estrategia de Versionamiento
El proyecto sigue los principios de control de versiones con Git y GitHub (`github.com/geocastro-lab/Actividad5.git`). Los cambios mayores se registran formalmente en el archivo `CHANGELOG.md` bajo el estándar de versionamiento semántico.

## 4. Visualizaciones Relevantes
Durante el análisis exploratorio de datos (EDA) en Google Colab, se identificaron:
* Distribuciones de clases desbalanceadas en la variable objetivo.
* Gráficos de correlación (Heatmaps) para descartar variables con alta multicolinealidad.
* Diagramas de caja (*Boxplots*) para el aislamiento y visualización de valores atípicos (*outliers*).

## 5. Descripción de los Modelos Entrenados
Implementados en el script `fuentes/train.py`:
* **Regresión Logística:** Modelo lineal base utilizado para estimar probabilidades de pertenencia a clases y evaluar la separabilidad lineal.
* **Random Forest Classifier:** Modelo basado en ensamblaje de árboles de decisión, capaz de capturar relaciones no lineales complejas y mitigar el sobreajuste.

## 6. Configuración de Hiperparámetros
* **Regresión Logística:** `max_iter=200`, regularización estándar L2 (`C=1.0`).
* **Random Forest:** `n_estimators=100`, `max_depth=5`, semilla aleatoria fija (`random_state=42`) para garantizar reproducibilidad.

## 7. Métricas Obtenidas
Evaluación cuantitativa sobre el conjunto de pruebas (*test set*):
* **Regresión Logística:** Accuracy global y métricas base de control.
* **Random Forest:** 
  * `Accuracy`: Rendimiento global de aciertos.
  * `Precision` / `Recall`: Balance de calidad y sensibilidad de las predicciones positivas.
  * `F1-Score`: Media armónica para robustez de evaluación.

## 8. Comparación Estadística de Resultados
| Modelo | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Regresión Logística | *0.6* | *0.36* | *0.60* | *0.45* |
| Random Forest | *0.8* | *0.85* | *0.80* | *0.78* |

*Conclusión:* Random Forest demostró una superioridad estadística al capturar interacciones no lineales entre las variables del dataset.

## 9. Evidencia del Uso de MLflow
El proyecto integra el registro automatizado de experimentos mediante `mlflow`:
* Seguimiento de parámetros de entrenamiento, métricas de evaluación y artefactos del modelo en la ruta local `mlruns/`.
* Trazabilidad completa garantizada mediante la ejecución de corridas (*runs*) independientes por modelo.

## 10. Reflexión sobre Deuda Técnica y MLOps
* **Deuda Técnica:** Las limitaciones de compatibilidad en entornos locales (Windows / `pyarrow`) requirieron la migración de la ejecución a Google Colab. A futuro, se recomienda estandarizar el pipeline mediante contenedores Docker.
* **Escalabilidad MLOps:** El flujo actual de scripts y versionamiento sienta las bases para automatizar un pipeline de integración continua (CI/CD) y despliegue del modelo en producción.

---
### Instrucciones de Ejecución
1. Clonar el repositorio: `git clone https://github.com/geocastro-lab/Actividad5.git`
2. Ejecutar la limpieza: `python fuentes/datos_prep.py`
3. Entrenar y registrar con MLflow: `python fuentes/train.py`