import pandas as pd
from etl import cargar_datos
from pathlib import Path

def eliminar_columnas(df: pd.DataFrame, columnas_a_eliminar: list) -> pd.DataFrame:
    columnas_existentes = [col for col in columnas_a_eliminar if col in df.columns]
    if columnas_existentes:
        df = df.drop(columns=columnas_existentes, errors="ignore")
        print(f"Columnas eliminadas: {columnas_existentes}")
    else:
        print("No se encontraron columnas para eliminar.")
    return df

if __name__ == "__main__":
    df = cargar_datos("TelecomX_Data.json")

    columnas_a_quitar = ["customerID"]

    df_limpio = eliminar_columnas(df, columnas_a_quitar)

    base_dir = Path(__file__).resolve().parents[1]  
    salida = base_dir / "data" / "datos_limpios.json"
    df_limpio.to_json(salida, orient="records", indent=4, force_ascii=False)
    print(f"Datos limpios guardados en: {salida}")

    print("Columnas actuales:", df_limpio.columns.tolist())
