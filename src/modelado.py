import os
import pandas as pd
import numpy as np
from etl import cargar_datos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. Cargar datos
# =========================
df = cargar_datos("datos_codificados.json")

if df.empty:
    exit("❌ No se pudieron cargar los datos para modelado.")

if "Churn" not in df.columns:
    exit("❌ No se encontró la columna 'Churn' en los datos.")

df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce").fillna(0).astype(int)

print("📊 Distribución general de Churn:")
print(df["Churn"].value_counts())

X = df.drop(columns=["Churn"])
y = df["Churn"]

# =========================
# 2. Separación de datos
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\n📊 Distribución en entrenamiento:")
print(y_train.value_counts())
print("\n📊 Distribución en prueba:")
print(y_test.value_counts())

# =========================
# 3. Modelos
# =========================
def entrenar_log_reg(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if len(np.unique(y_train)) < 2:
        print("⚠️ Regresión Logística omitida: solo hay una clase en el entrenamiento.")
        return None, None

    log_reg = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )
    log_reg.fit(X_train_scaled, y_train)
    y_pred = log_reg.predict(X_test_scaled)
    return log_reg, y_pred

def entrenar_rf(X_train, X_test, y_train, y_test):
    if len(np.unique(y_train)) < 2:
        print("⚠️ Random Forest omitido: solo hay una clase en el entrenamiento.")
        return None, None

    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return rf, y_pred

# =========================
# 4. Evaluación
# =========================
def evaluar_modelo(nombre, y_test, y_pred):
    if y_pred is None:
        print(f"❌ {nombre} no se pudo entrenar por falta de clases.")
        return

    print(f"\n📊 Resultados para {nombre}:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Activo","Cancelado"],
                yticklabels=["Activo","Cancelado"])
    plt.title(f"Matriz de confusión - {nombre}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/confusion_matrix_{nombre.replace(' ','_').lower()}.png")
    plt.close()

# =========================
# 5. Entrenar y Evaluar
# =========================
log_reg, y_pred_log = entrenar_log_reg(X_train, X_test, y_train, y_test)
rf, y_pred_rf = entrenar_rf(X_train, X_test, y_train, y_test)

evaluar_modelo("Regresión Logística", y_test, y_pred_log)
evaluar_modelo("Random Forest", y_test, y_pred_rf)

print("\n✅ Modelado completado. Resultados mostrados en consola y gráficas guardadas en 'plots/'")
