"""
Hiperparâmetros para KNN e SVM na base Ionosphere (UCI)
Autora: Pedro Barbosa
"""

# Bibliotecas essenciais
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Carregamento e Pré-processamento dos Dados
# Fonte: UCI Machine Learning Repository (Ionosphere)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
df = pd.read_csv(url, header=None)

# Separação em features e target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].map({'g': 1, 'b': 0}).values  # Convertendo classes para numérico

# Divisão em treino (70%), teste (15%) e validação (15% implícito no GridSearch)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, stratify=y_test, random_state=42
)

# Padronização dos dados (crucial para SVM e KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 2. Modelo KNN com Grid Search
knn = KNeighborsClassifier()

# Grade de hiperparâmetros testados:
knn_params = {
    'n_neighbors': [3, 5, 7, 9],  # Valores ímpares para evitar empates
    'weights': ['uniform', 'distance'],  # Ponderação por distância ou igual
    'p': [1, 2]  # 1=Manhattan, 2=Euclidiana
}

# Busca exaustiva com validação cruzada 5-fold
knn_grid = GridSearchCV(
    estimator=knn,
    param_grid=knn_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
knn_grid.fit(X_train, y_train)

# 3. Modelo SVM com Grid Search
svm = SVC(random_state=42)

# Grade de hiperparâmetros testados:
svm_params = {
    'C': [0.1, 1, 10],  # Trade-off entre margem e erros
    'kernel': ['rbf', 'linear'],  # Kernels testados
    'gamma': ['scale', 'auto', 0.1]  # Influência dos pontos de suporte
}

svm_grid = GridSearchCV(
    estimator=svm,
    param_grid=svm_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
svm_grid.fit(X_train, y_train)

# 4. Avaliação dos Modelos
# Função para resultados padronizados
def print_results(model, X_train, X_val, X_test):
    print(f"Melhores parâmetros: {model.best_params_}")
    print(f"Acurácia média na validação cruzada: {model.best_score_:.3f}")
    print(f"Acurácia no treino: {accuracy_score(y_train, model.predict(X_train)):.3f}")
    print(f"Acurácia na validação: {accuracy_score(y_val, model.predict(X_val)):.3f}")
    print(f"Acurácia no teste: {accuracy_score(y_test, model.predict(X_test)):.3f}\n")

# Resultados KNN
print("=== Resultados KNN ===")
print_results(knn_grid, X_train, X_val, X_test)

# Resultados SVM
print("=== Resultados SVM ===")
print_results(svm_grid, X_train, X_val, X_test)