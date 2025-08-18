import pandas as pd
from etl import cargar_datos

def aplanar_columnas_anidadas(df: pd.DataFrame, columnas_anidadas: list) -> pd.DataFrame:
    for col in columnas_anidadas:
        if col in df.columns:
            nuevas_columnas = pd.json_normalize(df[col])
            nuevas_columnas.columns = [f"{col}_{subcol}" for subcol in nuevas_columnas.columns]
            df = pd.concat([df.drop(columns=[col]), nuevas_columnas], axis=1)
    return df

def aplicar_encoding(df: pd.DataFrame) -> pd.DataFrame:
    columnas_objeto = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if "Churn" in columnas_objeto:
        columnas_objeto.remove("Churn")  
    df_codificado = pd.get_dummies(df, columns=columnas_objeto, drop_first=True)
    return df_codificado

if __name__ == "__main__":
    df = cargar_datos("datos_limpios.json")

    columnas_anidadas = ["customer", "phone", "internet", "account"]
    df = aplanar_columnas_anidadas(df, columnas_anidadas)

    df_codificado = aplicar_encoding(df)

    df_codificado.to_json("data/datos_codificados.json", orient="records", indent=4, force_ascii=False)

    print("✅ Encoding aplicado y datos guardados en 'data/datos_codificados.json'")
    print(df_codificado.head())

