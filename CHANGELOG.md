# Changelog

Todas las modificaciones notables de este proyecto serán documentadas en este archivo.
El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere al [Semantic Versioning](https://semver.org/lang/es/).
# [1.2.0] - 2026-10-19
### Añadido
* Integración del script de entrenamiento (`train.py`) y registro en MLflow con métricas de evaluación completas (**Accuracy, Precision, Recall y F1-Score**).
* Configuración del notebook en Google Colab (`entrena.ipynb`) para ejecución en la nube y trazabilidad de múltiples modelos (Regresión Logística y Random Forest).

## [1.2.0] - 2026-10-19
### Añadido
* Integración del script de entrenamiento (`train.py`) con registro de métricas y parámetros en MLflow.
* Configuración del notebook en Google Colab (`entrena.ipynb`) para ejecución en la nube.

## [1.1.0] - 2026-10-18
### Añadido
* Script de limpieza y preprocesamiento de datos (`datos_prep.py`).
* Creación de la estructura de directorios para datos crudos (`datos/datos_ini/`) y limpios (`datos/datos_limp/`).

## [1.0.0] - 2026-10-15
### Añadido
* Inicialización del repositorio Git y estructura base de carpetas (`fuentes/`).
* Carga inicial del dataset crudo (`dataset_raw.csv`).
* Documentación inicial en `README.md` y `CHANGELOG.md`.

