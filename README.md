# 📊 TelecomX - Predicción de Cancelación de Clientes

Proyecto de analítica y modelado para predecir **Churn** (cancelación) en una telco ficticia.

## 🚀 Pipeline

1. **ETL y Limpieza**
   - `src/limpieza.py`: elimina columnas innecesarias.
   - `src/encoding.py`: aplana columnas anidadas y aplica One-Hot Encoding.
   - `src/verificacion.py`: valida la presencia y consistencia de `Churn`.

2. **Análisis Exploratorio**
   - `src/analisis.py`: genera matriz de correlación y gráficas (en `/plots`).

3. **Modelado Predictivo**
   - `src/modelado.py`: separación train/test y modelos (Regresión Logística con normalización y Random Forest sin normalización).
   - Métricas: Accuracy, Precision, Recall, F1, Matriz de Confusión.

## 📁 Estructura del Proyecto

├─ data/ # (Ignorada en git) datasets locales
│ └─ .gitkeep
├─ plots/ # Gráficas generadas
├─ src/ # Código fuente
│ ├─ etl.py
│ ├─ limpieza.py
│ ├─ encoding.py
│ ├─ verificacion.py
│ ├─ analisis.py
│ └─ modelado.py
├─ Conclusiones_TelecomX.docx
└─ README.md
---


## ⚠️ Nota sobre el dataset

El archivo `data/datos_codificados.json` supera el límite de GitHub (100 MB), **no se incluye en el repositorio**.  
Coloca los datos en `data/` de forma local antes de ejecutar los scripts.  
Si quieres compartirlo, súbelo a un servicio externo (Drive/Dropbox/Kaggle) y enlázalo aquí.

## ✅ Estado del Proyecto

- El pipeline funciona de punta a punta.
- **Limitación detectada**: la variable `Churn` en el dataset actual solo contiene la clase `0` (activos), sin `1` (cancelados).
- Por ello, los modelos no se entrenan (requieren ambas clases). El código maneja esta situación e informa en consola.

## 🔧 Requerimientos

pip install pandas numpy scikit-learn matplotlib seaborn

## Autor
José Vicente Rodríguez Rivera

jvicente.rdz.r@gmail.com