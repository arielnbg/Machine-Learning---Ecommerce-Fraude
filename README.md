# Detecção de Fraudes em Transações — Machine Learning (PaySim/Fraude.csv)

Projeto de **classificação binária** para detectar **fraudes em transações financeiras** a partir de uma base de dados de comércio eletrônico (`Fraude.csv`).  
O pipeline cobre: **carregamento**, **pré-processamento**, **tratamento de desbalanceamento**, **treinamento de múltiplos modelos**, **avaliação** e **ensemble**.

---

## ✨ Objetivo

Construir e comparar modelos de Machine Learning para prever a coluna **`isFraud`**:

- `0` → transação legítima  
- `1` → transação fraudulenta  

Como o problema é **altamente desbalanceado**, o foco não é “acurácia”, e sim métricas como **Recall, Precision e F1-score** para a classe minoritária (fraude).

---

## 🧾 Dataset

O arquivo utilizado no notebook é:

- `Fraude.csv`

Estrutura inicial (exemplo de colunas):
- `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`,
- `nameDest`, `oldbalanceDest`, `newbalanceDest`,
- `isFraud`, `isFlaggedFraud`

> Observação: o dataset é grande (milhões de linhas). Rodar tudo pode exigir bastante RAM.

---

## 🧠 Pipeline (o que o notebook faz)

### 1) Carregamento
- Monta o Google Drive no Colab
- Carrega o CSV com `pandas`

### 2) Validação de integridade
- Verifica **duplicatas**
- Verifica **valores faltantes (NaN)**

### 3) Encoding de variáveis categóricas
- Aplica `LabelEncoder` em colunas do tipo `object` (ex.: `type`, `nameOrig`, `nameDest`)

### 4) Análise exploratória (rápida)
- Mostra **desbalanceamento** da variável alvo `isFraud`
- Plota **matriz de correlação** com heatmap

### 5) Seleção/remoção de colunas
Removeu:
- `nameOrig`, `nameDest` (sensíveis / identificadores)
- `isFlaggedFraud` (potencial vazamento/feature “entregando” o alvo)
- `newbalanceOrig`, `newbalanceDest` (redução de dimensionalidade)

Features finais:
- `step`, `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`
- alvo: `isFraud`

> Após remover colunas, podem surgir “duplicatas” por perda de distinção; o notebook remove essas linhas duplicadas.

### 6) Normalização
- Aplica `MinMaxScaler` para colocar as features na escala `[0, 1]`

### 7) Split treino/teste
- Divide em `80/20` com `train_test_split`

### 8) Balanceamento (undersampling)
- Usa `RandomUnderSampler` para equalizar as classes no treino

### 9) Treinamento e avaliação
Modelos testados:
- Decision Tree
- Random Forest
- Extra Trees
- Passive Aggressive
- XGBoost
- SVM
- KNN

Avaliação:
- **Curva ROC + AUC**
- **Acurácia**
- **Especificidade**
- **Classification Report**
- **Matriz de confusão** (valores + %)

### 10) Ensemble heterogêneo
- `VotingClassifier` (hard voting) com os modelos acima
- Avalia as mesmas métricas e plota matriz de confusão

---

## ✅ Resultados (resumo)

O notebook indica o **XGBoost** como modelo escolhido, por ter bom desempenho na classe minoritária (`isFraud=1`), principalmente em **Recall** e **F1-score**.

> Importante: em bases desbalanceadas, **acurácia** pode ser alta mesmo com desempenho ruim para fraudes. O ideal é focar em **Recall/Precision/F1 da classe 1** e calibrar o limiar de decisão quando necessário.

---

## 🔧 Como executar

### Opção A) Google Colab (recomendado)
1. Abra o notebook no Colab
2. Monte o Drive quando solicitado
3. Ajuste o caminho do dataset:
   ```python
   dataset = "/content/drive/MyDrive/Colab Notebooks/Comércio Eletrônico/Fraude.csv"
