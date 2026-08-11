import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from pathlib import Path

# 1. Configurar o nome do nosso "Projeto" no MLflow
mlflow.set_experiment("Previsao_de_Churn_Telecom")

# Descobre ONDE este arquivo .py está salvo de forma absoluta (ex: /home/.../src/)
script_dir = Path(__file__).parent

# Sobe um nível para a raiz e entra na pasta data/processed/
processed_dir = script_dir.parent / 'data' / 'processed'

def run_pipeline():
    print("Iniciando Pipeline de MLOps...")
    # 2. Carregar os dados (Como estamos em src/, voltamos um nível com '../')
    X_train = joblib.load(processed_dir / 'X_train_resampled.pkl')
    y_train = joblib.load(processed_dir / 'y_train_resampled.pkl')
    X_test = joblib.load(processed_dir / 'X_test_processed.pkl')
    y_test = joblib.load(processed_dir / 'y_test.pkl')

    # 3. Iniciar o "Diário de Bordo" do MLflow
    with mlflow.start_run(run_name="Champion_Logistic_Regression"):
        
        # Instanciar e Treinar
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Prever
        y_pred = model.predict(X_test)
        
        # Calcular Métricas Focadas na Classe 1 (Churn)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        

        
        # Logando parâmetros (Configurações)
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        
        # Logando métricas (Resultados)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Salvando o modelo treinado dentro do MLflow!
        mlflow.sklearn.log_model(model, "logistic_regression_model")
        joblib.dump(model, processed_dir / 'model.pkl')


        
        print(f"✅ Treinamento finalizado! Recall: {recall:.2f} | F1: {f1:.2f}")
        print("Tudo foi registrado no MLflow.")
if __name__ == "__main__":
    run_pipeline()