# Proyecto: Predicción de Enfermedades de Alto Costo en Colombia

## Descripción

Este proyecto analiza datos de mortalidad en Colombia (2019) para predecir enfermedades de alto costo utilizando técnicas de machine learning. El objetivo es clasificar las defunciones en cuatro grupos principales:
- **CANCER**: Cáncer
- **ER CRONICA**: Enfermedad Renal Crónica
- **VIH**: Virus de Inmunodeficiencia Humana
- **HEMOFILIA**: Hemofilia

## Estructura del Proyecto

```
Proyecto_Enfermedades_Alto_Costo/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── Mortalidad_2019_colombia1.xlsx
│   └── processed/
├── notebooks/
│   ├── 01_data_load_and_checks.ipynb
│   ├── 02_eda_plotly.ipynb
│   ├── 03_imputation_experiments.ipynb
│   ├── 04_preprocessing_pipelines.ipynb
│   ├── 05_model_baseline_and_eval.ipynb
│   ├── 06_smotenn_and_retraining.ipynb
│   ├── 07_model_tuning_and_comparison.ipynb
│   └── 08_export_and_streamlit_demo.ipynb
├── src/
│   ├── __init__.py
│   ├── io_utils.py
│   ├── imputation.py
│   ├── preprocessing.py
│   ├── sampling.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── export.py
└── streamlit_app/
    └── app.py
```

## Inicio Rápido 

### Requisitos
- Python 3.8 o superior .
- Jupyter Notebook o JupyterLab
- pip

### Instalación

1. descargar el proyecto

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecutar los notebooks en orden:
```bash
jupyter notebook notebooks/
```

### Ejecución Automática

Cada notebook es autocontenible y:
- Instala automáticamente las dependencias faltantes
- Configura rutas relativas
- Puede ejecutarse independientemente

## Notebooks

### 01. Carga y Validación de Datos
- Carga del dataset principal y diccionario de variables
- Validación de calidad de datos
- Análisis de valores nulos y duplicados
- Limpieza inicial

### 02. Análisis Exploratorio (EDA)
- Visualizaciones interactivas con Plotly
- Distribuciones, correlaciones y patrones
- Análisis geográfico y temporal
- Detección de outliers

### 03. Experimentos de Imputación
- Evaluación de estrategias de imputación
- Comparación: media, mediana, moda, KNN, Iterative
- Selección del mejor método por tipo de variable

### 04. Pipelines de Preprocesamiento
- Construcción de pipelines con ColumnTransformer
- Codificación (OneHot, Label Encoding)
- Escalado y normalización
- Reducción de dimensionalidad (PCA)

### 05. Modelos Baseline
- Entrenamiento de 5 modelos base
- Evaluación sin balanceo
- Comparación de métricas

### 06. Balanceo con SMOTEENN
- Aplicación de SMOTEENN (bajo, medio, alto)
- Reentrenamiento de modelos
- Análisis de mejoras

### 07. Ajuste de Hiperparámetros
- GridSearch/RandomSearch
- Selección del mejor modelo
- Ranking final

### 08. Exportación y Demo
- Exportación de resultados
- Aplicación Streamlit interactiva

## Modelos Utilizados

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier
4. LightGBM Classifier
5. K-Nearest Neighbors

## Métricas de Evaluación

- F1-Score
- Precision
- Recall
- ROC-AUC
- Matriz de Confusión

## Características Clave

- **Reproducibilidad Total**: Rutas relativas, instalación automática
- **Documentación Completa**: Cada notebook con análisis y conclusiones
- **Visualizaciones Interactivas**: Plotly para EDA
- **Manejo de Desbalanceo**: SMOTEENN en múltiples niveles
- **Pipeline Robusto**: ColumnTransformer con PCA
- **Demo Interactiva**: Aplicación Streamlit funcional

## Tecnologías

- **Python**: Pandas, NumPy, Scikit-learn
- **ML**: XGBoost, LightGBM, Imbalanced-learn
- **Visualización**: Plotly, Seaborn, Matplotlib
- **Notebooks**: Jupyter
- **Deployment**: Streamlit

## Notas Importantes

- Dataset: 42,644 registros, 37 variables
- Variable objetivo: GRUPO (4 clases)
- Año de datos: 2019
- Fuente: DANE (Colombia)
