# 🎓 Plano de Aprendizado: Sistema de Previsão de Churn com MLOps

**Instrutor:** Professor de Engenharia de Machine Learning  
**Aluno:** Estudante de ML Engineering  
**Dataset:** Telecom Churn (Excel `data/raw/churn.xlsx`).  
**Variável Alvo (Target):** `Churn Value` (1 = Cancelou, 0 = Ativo). A coluna `Churn Label` deve ser descartada.
**Objetivo:** Construir pipeline completo de ML com CI/CD e deploy em VPS  

---

## 📊 ETAPA 1: Exploração e Análise de Dados (EDA)

### 🎯 Objetivo
Criar notebook `01_eda_churn.ipynb` para entender estrutura, distribuições e correlações do dataset.

### 📚 O que você deve aprender
1. **Limpeza e Tratamento Específico**
   - **Vazamento de Dados (Data Leakage):** Descartar `Churn Reason` e `Churn Score`.
   - **Colunas Inúteis/Estáticas:** Descartar `CustomerID`, `Country` e `State`.
   - **Tratamento de Tipos:** A coluna `Total Charges` possui espaços em branco (' ') para novos clientes e deve ser convertida para float (NaNs substituídos por 0).
   - Separar variáveis numéricas (`Tenure Months`, `Monthly Charges`, `Total Charges`) de categóricas (`Gender`, `Internet Service`, `Contract`).

2. **Questões para responder**
   - Qual a distribuição de `Churn Value` por gênero?
   - Como `Tenure Months` se relaciona com churn? (Correlação linear)
   - Quais serviços de internet (DSL vs Fiber optic) têm maior churn? (Gráfico de Barras)
   - Contratos Month-to-month geram mais churn que contratos anuais?
   - Monthly charges correlacionam com churn?
   - Qual o impacto de streaming TV/movies no churn?

3. **Métricas de qualidade**
   - Imbalance de classes (churn rate atual: 55,5%)
   - Ausentes por coluna
   - Valores outliers em charges/CLTV

### ✅ Entregáveis
- [ ] Gráfico de distribuição de churn por variáveis categóricas
- [ ] Heatmap de correlação numérica-churn
- [ ] Boxplots de charges/CLTV/tenure por status churn
- [ ] Relatório escrito com insights

---

## ⚙️ ETAPA 2: Preparação de Dados (Train/Test Split, Encoding, Normalização)

### 🎯 Objetivo
Criar notebook `02_data_preparation.ipynb` para treinar modelos em diferentes configurações.

### 📚 O que você deve aprender
1. **Divisão Train/Test (80/20)**
   - Shuffle antes de dividir
   - Seed fixa para reprodutibilidade
   - Manter proporção de churn (stratified split)

2. **Encoding de variáveis categóricas**
   - Label Encoding (variáveis numéricas para ordinal)
   - One-Hot Encoding (variáveis nominais como Internet Service)
   - Quando usar cada técnica?

3. **Normalização/Padronização**
   - StandardScaler para modelos lineares (Linear/Logistic Regression)
   - MinMaxScaler para redes neurais
   - Quando NÃO normalizar (arboreto, KNN)

4. **Tratamento de desbalanceamento**
   - SMOTE para balancear classes
   - Class weights no modelo

### ✅ Entregáveis
- [ ] DataFrame final com train (80%) e test (20%)
- [ ] Pipeline completo (preprocessing + modelos)
- [ ] Comparativo de desempenho entre encodings

---

## 🤖 ETAPA 3: Treinamento e Avaliação de Modelos

### 🎯 Objetivo
Criar notebook `03_model_comparison.ipynb` para comparar múltiplos algoritmos.

### 📚 O que você deve aprender
1. **Modelos a testar**
   - **Base:** Logistic Regression (interpretabilidade)
   - **Árvore:** Random Forest (não-linear)
   - **Boosting:** XGBoost/LightGBM (estado da arte)
   - **Neural:** MLP Classifier (aprendizado profundo)

2. **Métricas de avaliação** (crucial para dados desbalanceados!)
   - Accuracy (pouco útil com imbalance)
   - Precision/Recall/F1 (foco na classe minoritária)
   - AUC-ROC (curva de classificação)
   - Confusion Matrix (visual)

3. **Validação cruzada**
   - Stratified K-Fold (manter proporção churn)
   - Grid Search para hyperparameters

### ✅ Entregáveis
- [ ] Tabela de comparison (F1, AUC, Precision, Recall)
- [ ] Curvas ROC para melhores modelos
- [ ] Melhores hyperparameters encontrados
- [ ] Recomendação de melhor modelo

---

## 📦 ETAPA 4: CI/CD com MLflow e MLOps

### 🎯 Objetivo
Criar `04_mlops_pipeline.py` para automatizar treinamento com rastreabilidade.

### 📚 O que você deve aprender
1. **MLflow Architecture**
   - **Experiments:** Espaço de experimentos
   - **Runs:** Cada execução de treino
   - **Artifacts:** Modelos salvos
   - **Metrics:** F1, AUC, Accuracy logados
   - **Parameters:** Hyperparameters registrados

2. **Pipeline automatizado**
   ```
   Carregar dados → Preprocessar → Train/Valid → 
   Grid Search → Salvar modelo → Logar métricas → 
   Salvar em MLflow
   ```

3. **CI/CD Básico**
   - GitHub Actions para trigger
   - Testes unitários de dados
   - Deploy automático de novo modelo

### ✅ Entregáveis
- [ ] Script de training com MLflow
- [ ] Experimentos registrados
- [ ] Modelo production-ready salvo
- [ ] Pipeline CI/CD documentado

---

## 🌐 ETAPA 5: API e Deploy em VPS

### 🎯 Objetivo
Criar `app.py` (API FastAPI) + deploy em servidor Linux.

### 📚 O que você deve aprender
1. **API com FastAPI**
   - Endpoint `POST /predict` com payload JSON
   - Input validation (Pydantic)
   - Response com probability + prediction
   - Docker containerização

2. **Deploy em VPS**
   - AWS EC2 / DigitalOcean Droplet / Linode
   - Systemd service para auto-start
   - Nginx reverse proxy
   - SSL com Let's Encrypt
   - Supervisão com PM2 ou systemd

3. **Monitoramento**
   - Logging de requisições
   - Health check endpoint
   - Métricas de performance

### ✅ Entregáveis
- [ ] API FastAPI rodando em porta 8000
- [ ] Dockerfile otimizado
- [ ] Script de deploy em VPS
- [ ] Documentação de endpoints

---

## 🚀 Sequência de Execução

```
1. 01_eda_churn.ipynb     → Execute e ENTENDA os dados
2. 02_data_preparation.ipynb → Crie pipeline de preprocessing
3. 03_model_comparison.ipynb → Compare modelos e escolha melhor
4. 04_mlops_pipeline.py    → Automatize com MLflow
5. app.py + deploy         → API e produção
```

---

## 💡 Dicas do Professor

- **Não copie código pronto** - escreva linha por linha entendendo cada parte
- **Documente cada etapa** - comentários explicando DECISÕES
- **Valide tudo** - cheque se o preprocessing faz sentido
- **Versione tudo** - git commit após cada notebook
- **Teste incrementalmente** - teste cada função isoladamente

---

## 📁 Estrutura de Pastas Final

```
preditor-churn/
├── data/
│   └── raw/
│       └── churn.xlsx
├── notebooks/
│   ├── 01_eda_churn.ipynb
│   ├── 02_data_preparation.ipynb
│   └── 03_model_comparison.ipynb
├── src/
│   └── mlops_pipeline.py
├── app.py
├── Dockerfile
├── requirements.txt
├── agente.md
└── README.md
```

---

**Próximo passo:** Comece a **ETAPA 1**. Abra um terminal, carregue pandas, explore suas primeiras 100 linhas, identifique tipos de cada coluna.

> ⚠️ **RECORDAÇÃO:** Eu sou seu professor. Vou explicar TEORIA antes de mostrar CÓDIGO. Quando pedir ajuda, pergunte "como fazer X" em vez de "escreva código para X".
