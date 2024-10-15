# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
file_path = 'ForExport.csv'  # Substituir pelo caminho correto do arquivo
data = pd.read_csv(file_path)

# Seleção de colunas numéricas e definição da variável alvo
data_numeric = data.select_dtypes(include=['float64', 'int64']).dropna()
target_column = 'JamsDelay'
X = data_numeric.drop(target_column, axis=1)
y = data_numeric[target_column]

# Normalização e divisão em treino e teste
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Definição dos modelos
models = {
    'Linear Regression': LinearRegression(),
    'PLS Regression': PLSRegression(n_components=2),
    'Lasso Regression': Lasso(alpha=0.01),
    'Ridge Regression': Ridge(alpha=0.5),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
    'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42)
}

# Treinamento, predição e cálculo das métricas
results = {'Model': [], 'RMSE': [], 'R2': [], 'MAPE': []}
predictions = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Armazenar métricas e previsões
    results['Model'].append(model_name)
    results['RMSE'].append(mean_squared_error(y_test, y_pred, squared=False))
    results['R2'].append(r2_score(y_test, y_pred))
    results['MAPE'].append(mean_absolute_error(y_test, y_pred) / abs(y_test.mean()) * 100)
    predictions[model_name] = y_pred

# Exibir resultados
results_df = pd.DataFrame(results)
print(results_df)

# Selecionar 30 amostras e criar gráfico de comparação
y_test_limited = y_test[:30]
predictions_limited = {model_name: y_pred[:30] for model_name, y_pred in predictions.items()}

plt.figure(figsize=(10, 6))
plt.plot(y_test_limited.values, label='Real', marker='o', color='blue', linewidth=3, markersize=8)

# Adicionar cores, estilo e aumentar espessura das linhas para visibilidade
colors = ['orange', 'green', 'red', 'purple', 'cyan', 'magenta']
markers = ['x', 'd', 's', 'v', '^', 'P']  # Diferentes marcadores para cada modelo
for idx, (model_name, y_pred) in enumerate(predictions_limited.items()):
    plt.plot(y_pred, label=model_name, color=colors[idx], linewidth=2, marker=markers[idx], markersize=8)

plt.xlabel('Amostras')
plt.ylabel('Valor')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.title('Comparação de Modelos na Porção de Teste')
plt.show()
