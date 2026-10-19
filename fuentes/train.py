import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Cargar dataset limpio
df = pd.read_csv("datos/datos_limp/dataset_clean.csv")

# Separar variables predictoras (X) y objetivo (y)
X = df.drop(columns=["target"]) # Ajusta "target" por el nombre real de tu columna objetivo
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar experimento en MLflow
mlflow.set_experiment("Actividad5_Experimentos")

# Función auxiliar para calcular y registrar todas las métricas
def evaluar_y_registrar(model, model_name, X_test, y_test):
    preds = model.predict(X_test)
    
    # Cálculo de métricas
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    
    # Registro en MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    print(f"=== Métricas de {model_name} ===")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}\n")

# --- Experimento 1: Regresión Logística ---
with mlflow.start_run(run_name="LogisticRegression_Baseline"):
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    
    evaluar_y_registrar(lr, "Regresión Logística", X_test, y_test)
    mlflow.sklearn.log_model(lr, "model_lr")

# --- Experimento 2: Random Forest ---
with mlflow.start_run(run_name="RandomForest_Model"):
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    evaluar_y_registrar(rf, "Random Forest", X_test, y_test)
    mlflow.sklearn.log_model(rf, "model_rf")