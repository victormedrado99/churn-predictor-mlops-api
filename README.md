# 🚀 Churn Predictor MLOps API

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=mlflow&logoColor=blue)

## 📌 Visão Geral
Este é um projeto End-to-End de Machine Learning desenvolvido como parte do portfólio acadêmico da **FIAP**, utilizando o clássico dataset de evasão de clientes (Telco Customer Churn) fornecido pela **IBM**.

O objetivo técnico não é apenas treinar um modelo, mas construir uma **esteira de MLOps completa**: desde a análise exploratória e tratamento de dados, passando pelo rastreamento de experimentos com MLflow, até o deploy de uma API conteinerizada via FastAPI e Docker pronta para ser consumida em produção (ex: AWS EC2).

## 💼 O Problema de Negócio
No mercado de telecomunicações, prever se um cliente vai cancelar o serviço (Churn) é vital. **Adquirir um novo cliente custa muito mais caro do que reter um atual.**
O objetivo deste modelo é identificar precocemente potenciais cancelamentos para que a equipe de retenção possa agir oferecendo descontos ou suporte proativo.

## 🧠 A Decisão do Modelo (A visão de Negócio)
Durante a fase de testes, avaliamos algoritmos complexos como **Random Forest** e **XGBoost**. No entanto, a nossa Vencedora (Champion Model) foi a **Regressão Logística**. 
* **Por quê?** Em cenários de Churn, um falso positivo (dar desconto a quem não ia cancelar) é mais barato do que um falso negativo (perder um cliente de surpresa). A Regressão Logística obteve um **Recall de 78%**, superando o XGBoost e capturando quase todos os verdadeiros cancelamentos, alinhando a IA com a métrica financeira mais sensível do negócio.

## 🏗️ Arquitetura e Tecnologias
1. **Análise Exploratória (EDA):** `pandas`, `seaborn` e `matplotlib` para isolar variáveis e encontrar vazamento de dados.
2. **Engenharia de Features:** `scikit-learn` (Pipelines e ColumnTransformers) e balanceamento de classes com **SMOTE** (`imbalanced-learn`).
3. **Rastreamento de Experimentos (MLOps):** Uso do `MLflow` para logar hyperparâmetros, métricas (F1, Precision, Recall) e serializar os artefatos de forma profissional.
4. **Deploy & API:** `FastAPI` + `Pydantic` para validação rigorosa de dados de entrada, empacotados num `Dockerfile` enxuto.

## 🚀 Como Executar o Projeto

### Pré-requisitos
Ter o [Docker](https://www.docker.com/) e o `docker-compose` instalados na máquina.

### Passos para subir a API
1. Clone este repositório:
   ```bash
   git clone https://github.com/SEU-USUARIO/churn-predictor-mlops-api.git
   cd churn-predictor-mlops-api
   ```
2. Construa e inicie o container:
   ```bash
   docker compose up -d --build
   ```
3. A API estará rodando! Acesse a documentação interativa (Swagger) no seu navegador:
   🔗 `http://localhost:8000/docs` ou `http://<IP-DA-AWS>/docs`

## 👨‍💻 Desenvolvedor
Desenvolvido por **Victor Medrado**
