# 📊 TelecomX - Predicción de Cancelación de Clientes

Este proyecto implementa un pipeline completo de análisis y modelado predictivo para **detectar la cancelación de clientes (Churn)** en una empresa de telecomunicaciones ficticia.

---

## 🚀 Pipeline del Proyecto

1. **ETL y Limpieza**
   - `limpieza.py`: elimina columnas innecesarias.
   - `encoding.py`: aplana columnas anidadas y aplica codificación one-hot.
   - `verificacion.py`: valida la existencia y consistencia de la variable objetivo `Churn`.

2. **Análisis Exploratorio**
   - `analisis.py`: genera correlaciones y gráficas exploratorias.
   - Resultados visuales guardados en la carpeta `/plots`.

3. **Modelado Predictivo**
   - `modelado.py`: entrena modelos de clasificación (Regresión Logística, Random Forest).
   - Evaluación con métricas: Accuracy, Precision, Recall, F1-score y Matriz de Confusión.

---

## 📁 Estructura del Proyecto

TelecomX-Predictivo/
│── data/ # Datos originales, limpios y codificados
│── plots/ # Gráficas generadas en el análisis y modelado
│── src/ # Scripts de Python
│ ├── etl.py
│ ├── limpieza.py
│ ├── encoding.py
│ ├── verificacion.py
│ ├── analisis.py
│ └── modelado.py
│── Conclusiones_TelecomX.docx # Documento de conclusiones
│── README.md


---

## 📌 Limitaciones Encontradas

Durante el modelado se identificó que el dataset contiene únicamente clientes activos (`Churn = 0`), sin ejemplos de cancelación (`Churn = 1`).  
Esto imposibilitó el entrenamiento de modelos predictivos, pero permitió demostrar un **pipeline completo de analítica de datos**.

---

## ✅ Conclusiones

- El pipeline implementado (ETL → análisis → modelado) funciona correctamente.  
- La ausencia de clientes cancelados en los datos originales impidió el modelado.  
- En un caso real, sería necesario **corregir el dataset**, balancear clases y reentrenar modelos.  

---

## 🔧 Requerimientos

Instalar dependencias principales:

pip install pandas numpy scikit-learn matplotlib seaborn

## Autor
José Vicente Rodríguez Rivera

jvicente.rdz.r@gmail.com