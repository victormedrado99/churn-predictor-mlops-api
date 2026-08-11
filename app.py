import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# --- 1. O Segurança da Porta (Pydantic) ---
# Define os campos exatos que a API espera receber no JSON
class CustomerData(BaseModel):
    Tenure_Months: int
    Monthly_Charges: float
    Total_Charges: float
    CLTV: int
    Senior_Citizen: str
    Partner: str
    Dependents: str
    Gender: str
    Phone_Service: str
    Multiple_Lines: str
    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str

# --- 2. Carregando os Cérebros ---
app = FastAPI(title="API de Previsão de Churn Telecom")

# Encontrar o diretório base para carregar os arquivos processados
base_dir = Path(__file__).parent / 'data' / 'processed'

# Carregamos o preprocessor da Etapa 2
preprocessor = joblib.load(base_dir / 'preprocessor.pkl')
model = joblib.load(base_dir / 'model.pkl') 

# --- 3. O Endpoint de Previsão ---
@app.post("/predict")
def predict_churn(customer: CustomerData):
    # Transforma o JSON recebido em um DataFrame do Pandas de 1 linha
    # Convertendo os underscores para espaços para bater com as colunas que o preprocessor espera
    data_dict = {key.replace('_', ' '): [value] for key, value in customer.dict().items()}
    df = pd.DataFrame(data_dict)
        
    # 1. Pré-processar os dados usando a fábrica que criamos lá atrás!
    try:
        X_processed = preprocessor.transform(df)
            
        # 2. Fazer a previsão
        prediction = model.predict(X_processed)
        probability = model.predict_proba(X_processed)[0][1] # Pega a propabilidade da classe 1
            
        return {
            "status": "sucesso",
            "message": "Dados processados perfeitamente pelo ColumnTransformer!",
            "churn_prediction": int(prediction[0]),
            "churn_probability": round(float(probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

