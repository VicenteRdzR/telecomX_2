import pandas as pd
from etl import cargar_datos

if __name__ == "__main__":
    df = cargar_datos("datos_codificados.json")

    if "Churn" not in df.columns:
        print("❌ No se encontró la columna 'Churn'.")
    else:
        print("✅ Columna 'Churn' presente.")

        df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce").fillna(0).astype(int)

        total = len(df)
        churn_count = df["Churn"].sum()
        active_count = total - churn_count

        print(f"Total clientes: {total}")
        print(f"Clientes que cancelaron: {churn_count} ({churn_count/total:.2%})")
        print(f"Clientes activos: {active_count} ({active_count/total:.2%})")

        print("Información general del DataFrame:")
        print(df.info())

        print("Primeras filas del DataFrame:")
        print(df.head())
