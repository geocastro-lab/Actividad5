import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def entrenar_modelo():
    # 1. Cargar datos limpios
    df = pd.read_csv("datos/datos_limp/dataset_clean.csv")
    
    # Separar features (X) y target (y) - *Ajusta 'target' por el nombre real de tu columna*
    X = df.drop(columns=["target"]) 
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Iniciar seguimiento con MLflow
    mlflow.set_experiment("Actividad5_MLOps")
    
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        # Hiperparámetros
        n_estimators = 100
        max_depth = 5
        
        # Registrar parámetros
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Entrenar modelo
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Predicciones y métricas
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average="weighted", zero_division=0)
        rec = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
        
        # Registrar métricas en MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Registrar el modelo
        mlflow.sklearn.log_model(model, "modelo_random_forest")
        
        print(f"Entrenamiento finalizado. Accuracy obtenido: {acc:.4f}")

if __name__ == "__main__":
    entrenar_modelo()