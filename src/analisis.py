import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from etl import cargar_datos

ARCHIVO_JSON = "datos_codificados.json"
PLOTS_PATH = "plots/"

os.makedirs(PLOTS_PATH, exist_ok=True)

df = cargar_datos(ARCHIVO_JSON)
if df.empty:
    exit("❌ No se pudieron cargar datos, abortando...")

bad_cols = [c for c in df.columns if "Unnamed" in c or " " in c]
if bad_cols:
    print(f"⚠️ Eliminando {len(bad_cols)} columnas mal expandidas...")
    df = df.drop(columns=bad_cols)

for col in df.columns:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

if "Churn" not in df.columns:
    exit("❌ No se encontró la columna 'Churn' en los datos.")

df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce").fillna(0).astype(int)

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

if "Churn" not in num_cols:
    num_cols.append("Churn")

plt.figure(figsize=(10, 8))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("Matriz de correlación - Variables numéricas")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "correlacion_matriz.png"))
plt.close()

corr_churn = corr["Churn"].drop("Churn")
print("\n📊 Correlación con Churn:")
print(corr_churn.sort_values(ascending=False))

plt.figure(figsize=(6, 6))
sns.barplot(x=corr_churn.values, y=corr_churn.index, palette="coolwarm", legend=False)
plt.title("Correlación con Churn")
plt.xlabel("Coeficiente de correlación")
plt.ylabel("Variables")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "correlacion_churn.png"))
plt.close()

posibles_tiempo = [c for c in df.columns if "tenure" in c.lower()]
if posibles_tiempo:
    col_tiempo = posibles_tiempo[0]
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Churn", y=col_tiempo, data=df, palette="Set2")
    plt.title(f"{col_tiempo} × Churn")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, f"{col_tiempo}_vs_churn.png"))
    plt.close()
else:
    print("⚠️ No se encontró columna de tiempo de contrato para el análisis.")

posibles_total = [c for c in df.columns if "Total" in c or "Charges" in c]
if posibles_total:
    col_total = posibles_total[0]
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Churn", y=col_total, data=df, palette="Set2")
    plt.title(f"{col_total} × Churn")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, f"{col_total}_vs_churn.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=col_total, y="Churn", data=df, alpha=0.5)
    plt.title(f"{col_total} × Churn (scatter)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, f"{col_total}_scatter_churn.png"))
    plt.close()
else:
    print("⚠️ No se encontró columna de gasto total para el análisis.")

print("✅ Análisis completado. Gráficas guardadas en 'plots/'")
