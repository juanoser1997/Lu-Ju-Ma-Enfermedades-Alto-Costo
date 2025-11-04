"""
Aplicación Streamlit para Predicción de Enfermedades de Alto Costo
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Configurar rutas
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from preprocessing import PreprocessingPipeline
from evaluation import plot_confusion_matrix, plot_roc_curves

# Configuración de la página
st.set_page_config(
    page_title="Predicción Enfermedades Alto Costo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cargar modelo y componentes
@st.cache_resource
def load_model_components():
    """Carga modelo, preprocessor y encoder"""
    try:
        DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
        
        # Cargar modelo
        model_data = joblib.load(DATA_PROCESSED / 'best_model.joblib')
        model = model_data['model']
        metadata = model_data.get('metadata', {})
        
        # Cargar preprocessor
        preprocessor = PreprocessingPipeline.load(DATA_PROCESSED / 'preprocessor.joblib')
        
        # Cargar datos de test
        X_test = np.load(DATA_PROCESSED / 'X_test.npy')
        y_test = np.load(DATA_PROCESSED / 'y_test.npy')
        
        return model, preprocessor, metadata, X_test, y_test
    except Exception as e:
        st.error(f"Error al cargar componentes: {e}")
        return None, None, {}, None, None

# Header
st.markdown('<h1 class="main-header">🏥 Predicción de Enfermedades de Alto Costo</h1>', 
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    mode = st.radio(
        "Modo de operación:",
        ["📊 Evaluación del Modelo", "🔮 Predicción Individual"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📋 Información del Proyecto")
    st.markdown("""
    - **Dataset**: Mortalidad 2019 Colombia
    - **Clases**: CANCER, ER CRONICA, VIH, HEMOFILIA
    - **Técnicas**: SMOTEENN, PCA, Ensemble Methods
    """)

# Cargar modelo
model, preprocessor, metadata, X_test, y_test = load_model_components()

if model is None:
    st.error("⚠️ No se pudo cargar el modelo. Ejecuta primero los notebooks.")
    st.stop()

# MODO 1: Evaluación del Modelo
if mode == "📊 Evaluación del Modelo":
    st.header("📊 Evaluación del Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Modelo", metadata.get('model_name', 'N/A'))
    
    with col2:
        f1_score = metadata.get('f1_score', 0)
        st.metric("F1-Score", f"{f1_score:.4f}")
    
    with col3:
        # Calcular accuracy desde y_pred si no está en metadata
        from sklearn.metrics import accuracy_score
        if X_test is not None and y_test is not None:
            try:
                y_pred_temp = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred_temp)
            except:
                accuracy = metadata.get('accuracy', 0)
        else:
            accuracy = metadata.get('accuracy', 0)
        st.metric("Accuracy", f"{accuracy:.4f}")
    
    with col4:
        st.metric("Balanceo", "SMOTEENN")
    
    st.markdown("---")
    
    # Predicciones en test set
    with st.spinner("Generando predicciones..."):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Tabs para diferentes visualizaciones
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Métricas", "🎯 Matriz de Confusión", 
                                       "📊 Curvas ROC", "📋 Predicciones"])
    
    with tab1:
        st.subheader("Métricas de Rendimiento")
        
        from sklearn.metrics import (classification_report, accuracy_score, 
                                     f1_score, precision_score, recall_score)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        
        # Mostrar métricas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("F1-Score", f"{f1_weighted:.4f}")
        col3.metric("Precision", f"{precision_weighted:.4f}")
        col4.metric("Recall", f"{recall_weighted:.4f}")
        
        # Reporte de clasificación
        st.subheader("Reporte de Clasificación por Clase")
        
        class_names = preprocessor.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names, 
                                      output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report = df_report.round(4)
        
        st.dataframe(df_report.style.highlight_max(axis=0, color='lightgreen'), 
                    use_container_width=True)
    
    with tab2:
        st.subheader("Matriz de Confusión")
        
        class_names = preprocessor.label_encoder.classes_.tolist()
        fig_cm = plot_confusion_matrix(y_test, y_pred, labels=class_names,
                                      title="Matriz de Confusión - Test Set")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Calcular métricas por clase
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        st.markdown("### 📊 Análisis de Resultados")
        
        # Calcular accuracy por clase
        class_accuracies = []
        for i, class_name in enumerate(class_names):
            correct = cm[i, i]
            total = cm[i, :].sum()
            acc = (correct / total * 100) if total > 0 else 0
            class_accuracies.append((class_name, correct, total, acc))
        
        # Mostrar análisis
        for class_name, correct, total, acc in class_accuracies:
            st.markdown(f"**{class_name}**: {correct}/{total} predicciones correctas ({acc:.1f}%)")
        
        # Encontrar confusiones principales
        st.markdown("\n**Principales confusiones detectadas:**")
        confusions = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    confusions.append((class_names[i], class_names[j], cm[i, j]))
        
        confusions.sort(key=lambda x: x[2], reverse=True)
        for real, pred, count in confusions[:3]:
            st.markdown(f"- {count} casos de **{real}** clasificados como **{pred}**")
    
    with tab3:
        st.subheader("Curvas ROC")
        
        if y_pred_proba is not None:
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize
            
            class_names = preprocessor.label_encoder.classes_.tolist()
            n_classes = len(class_names)
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            
            fig_roc = plot_roc_curves(y_test, y_pred_proba, class_names=class_names,
                                     title="Curvas ROC por Clase")
            st.plotly_chart(fig_roc, use_container_width=True)
            
            st.markdown("### 📊 Análisis de Resultados por Clase")
            
            # Calcular AUC para cada clase
            auc_scores = []
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                auc_scores.append((class_names[i], roc_auc))
            
            # Ordenar por AUC
            auc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Mostrar análisis
            st.markdown("**Rendimiento por enfermedad (AUC):**")
            for class_name, auc_score in auc_scores:
                if auc_score >= 0.9:
                    performance = "Excelente"
                    color = "🟢"
                elif auc_score >= 0.8:
                    performance = "Muy bueno"
                    color = "🟡"
                elif auc_score >= 0.7:
                    performance = "Bueno"
                    color = "🟠"
                else:
                    performance = "Mejorable"
                    color = "🔴"
                
                st.markdown(f"{color} **{class_name}**: {auc_score:.3f} - {performance}")
            
            # Resumen general
            avg_auc = sum([score for _, score in auc_scores]) / len(auc_scores)
            st.markdown(f"\n**AUC promedio del modelo: {avg_auc:.3f}**")
        else:
            st.warning("El modelo no proporciona probabilidades.")
    
    with tab4:
        st.subheader("Muestra de Predicciones")
        
        # Crear DataFrame con predicciones
        class_names = preprocessor.label_encoder.classes_
        df_predictions = pd.DataFrame({
            'Valor Real': [class_names[i] for i in y_test],
            'Predicción': [class_names[i] for i in y_pred],
            'Correcto': y_test == y_pred
        })
        
        if y_pred_proba is not None:
            df_predictions['Confianza'] = (y_pred_proba.max(axis=1) * 100).round(2)
        
        # Filtros
        show_only_errors = st.checkbox("Mostrar solo errores")
        
        if show_only_errors:
            df_show = df_predictions[~df_predictions['Correcto']]
        else:
            df_show = df_predictions
        
        st.dataframe(df_show.head(100), use_container_width=True)
        
        # Estadísticas
        st.markdown("### 📊 Estadísticas")
        col1, col2, col3 = st.columns(3)
        
        total = len(df_predictions)
        correct = df_predictions['Correcto'].sum()
        incorrect = total - correct
        
        col1.metric("Total Predicciones", total)
        col2.metric("Correctas", f"{correct} ({correct/total*100:.1f}%)")
        col3.metric("Incorrectas", f"{incorrect} ({incorrect/total*100:.1f}%)")

# MODO 2: Predicción Individual
else:
    st.header("🔮 Predicción Individual")
    
    st.markdown("""
    Herramienta de predicción con **todas las variables** del modelo.
    """)
    
    # TODAS las variables del modelo
    ALL_MODEL_VARIABLES = {
        'COD_DPTO': {'min': 5, 'max': 99, 'default': 17, 'help': 'Código departamento residencia'},
        'COD_MUNIC': {'min': 1, 'max': 980, 'default': 500, 'help': 'Código municipio residencia'},
        'HORA': {'min': 0, 'max': 23, 'default': 12, 'help': 'Hora de defunción'},
        'MINUTOS': {'min': 0, 'max': 59, 'default': 30, 'help': 'Minutos de defunción'},
        'GRU_ED1': {'min': 0, 'max': 28, 'default': 16, 'help': 'Grupo edad detallado'},
        'CODPTORE': {'min': 5, 'max': 99, 'default': 17, 'help': 'Código departamento ocurrencia'},
        'CODMUNRE': {'min': 1, 'max': 980, 'default': 500, 'help': 'Código municipio ocurrencia'}
    }
    
    # Opciones para variables categóricas
    NOM_DEP_OPTIONS = ['CALDAS', 'ANTIOQUIA', 'BOGOTÁ, D.C.', 'VALLE DEL CAUCA', 'ATLÁNTICO', 
                       'CUNDINAMARCA', 'BOLÍVAR', 'SANTANDER', 'TOLIMA', 'CÓRDOBA']
    
    OCUPACION_OPTIONS = ['HOGAR', 'AGRICULTORES DE CULTIVOS TRANSITORIOS', 'PENSIONADO', 
                         'VENDEDORES AMBULANTES', 'ESTUDIANTE', 'SIN INFORMACION',
                         'CONDUCTORES DE TAXIS', 'ALBAÑILES  MAMPOSTEROS Y AFINES',
                         'PROFESORES DE EDUCACIÓN PRIMARIA', 'ASEADORES Y FUMIGADORES DE OFICINAS  HOTELES Y OTROS ESTABLECIMIENTOS']
    
    # Estado para valores aleatorios
    if 'random_values' not in st.session_state:
        st.session_state.random_values = {key: float(val['default']) for key, val in ALL_MODEL_VARIABLES.items()}
        st.session_state.random_values['Nom Dep'] = NOM_DEP_OPTIONS[0]
        st.session_state.random_values['OCUPACION'] = OCUPACION_OPTIONS[0]
    
    # Botón para generar valores aleatorios
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        if st.button("🎲 Valores Aleatorios", use_container_width=True, key='random_btn'):
            # Generar valores aleatorios para numéricas
            for key, val in ALL_MODEL_VARIABLES.items():
                st.session_state.random_values[key] = float(np.random.randint(val['min'], val['max'] + 1))
            
            # Generar valores aleatorios para categóricas
            st.session_state.random_values['Nom Dep'] = str(np.random.choice(NOM_DEP_OPTIONS))
            st.session_state.random_values['OCUPACION'] = str(np.random.choice(OCUPACION_OPTIONS))
            
            st.rerun()
    
    with col_btn2:
        if st.button("🔄 Resetear", use_container_width=True, key='reset_btn'):
            st.session_state.random_values = {key: float(val['default']) for key, val in ALL_MODEL_VARIABLES.items()}
            st.session_state.random_values['Nom Dep'] = NOM_DEP_OPTIONS[0]
            st.session_state.random_values['OCUPACION'] = OCUPACION_OPTIONS[0]
            st.rerun()
    
    st.subheader("📝 Variables del Modelo:")
    
    # Crear formulario
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        input_data = {}
        
        with col1:
            st.markdown("**Ubicación Residencia**")
            
            input_data['COD_DPTO'] = st.number_input(
                "Código Departamento Residencia",
                min_value=int(ALL_MODEL_VARIABLES['COD_DPTO']['min']),
                max_value=int(ALL_MODEL_VARIABLES['COD_DPTO']['max']),
                value=int(st.session_state.random_values['COD_DPTO']),
                help=ALL_MODEL_VARIABLES['COD_DPTO']['help'],
                key='input_COD_DPTO'
            )
            
            input_data['COD_MUNIC'] = st.number_input(
                "Código Municipio Residencia",
                min_value=int(ALL_MODEL_VARIABLES['COD_MUNIC']['min']),
                max_value=int(ALL_MODEL_VARIABLES['COD_MUNIC']['max']),
                value=int(st.session_state.random_values['COD_MUNIC']),
                help=ALL_MODEL_VARIABLES['COD_MUNIC']['help'],
                key='input_COD_MUNIC'
            )
            
            input_data['Nom Dep'] = st.selectbox(
                "Nombre Departamento",
                options=NOM_DEP_OPTIONS,
                index=NOM_DEP_OPTIONS.index(st.session_state.random_values['Nom Dep']),
                key='input_Nom_Dep'
            )
        
        with col2:
            st.markdown("**Ubicación Ocurrencia**")
            
            input_data['CODPTORE'] = st.number_input(
                "Código Departamento Ocurrencia",
                min_value=int(ALL_MODEL_VARIABLES['CODPTORE']['min']),
                max_value=int(ALL_MODEL_VARIABLES['CODPTORE']['max']),
                value=int(st.session_state.random_values['CODPTORE']),
                help=ALL_MODEL_VARIABLES['CODPTORE']['help'],
                key='input_CODPTORE'
            )
            
            input_data['CODMUNRE'] = st.number_input(
                "Código Municipio Ocurrencia",
                min_value=int(ALL_MODEL_VARIABLES['CODMUNRE']['min']),
                max_value=int(ALL_MODEL_VARIABLES['CODMUNRE']['max']),
                value=int(st.session_state.random_values['CODMUNRE']),
                help=ALL_MODEL_VARIABLES['CODMUNRE']['help'],
                key='input_CODMUNRE'
            )
            
            input_data['GRU_ED1'] = st.number_input(
                "Grupo de Edad",
                min_value=int(ALL_MODEL_VARIABLES['GRU_ED1']['min']),
                max_value=int(ALL_MODEL_VARIABLES['GRU_ED1']['max']),
                value=int(st.session_state.random_values['GRU_ED1']),
                help=ALL_MODEL_VARIABLES['GRU_ED1']['help'],
                key='input_GRU_ED1'
            )
        
        with col3:
            st.markdown("**Información Temporal y Ocupación**")
            
            input_data['HORA'] = st.number_input(
                "Hora de Defunción",
                min_value=int(ALL_MODEL_VARIABLES['HORA']['min']),
                max_value=int(ALL_MODEL_VARIABLES['HORA']['max']),
                value=int(st.session_state.random_values['HORA']),
                help=ALL_MODEL_VARIABLES['HORA']['help'],
                key='input_HORA'
            )
            
            input_data['MINUTOS'] = st.number_input(
                "Minutos de Defunción",
                min_value=int(ALL_MODEL_VARIABLES['MINUTOS']['min']),
                max_value=int(ALL_MODEL_VARIABLES['MINUTOS']['max']),
                value=int(st.session_state.random_values['MINUTOS']),
                help=ALL_MODEL_VARIABLES['MINUTOS']['help'],
                key='input_MINUTOS'
            )
            
            input_data['OCUPACION'] = st.selectbox(
                "Ocupación",
                options=OCUPACION_OPTIONS,
                index=OCUPACION_OPTIONS.index(st.session_state.random_values['OCUPACION']),
                key='input_OCUPACION'
            )
        
        submitted = st.form_submit_button("🔮 Realizar Predicción", type="primary", use_container_width=True)
        
        if submitted:
            try:
                # Crear DataFrame
                df_input = pd.DataFrame([input_data])
                
                # --- CONVERSIÓN A TIPOS EXACTOS DEL DATASET ORIGINAL ---
                
                # Las columnas numéricas del preprocessor esperan float64
                numeric_float_cols = ['COD_DPTO', 'COD_MUNIC', 'HORA', 'MINUTOS', 'GRU_ED1', 'CODPTORE', 'CODMUNRE']
                
                # Las columnas categóricas esperan string (object)
                categorical_cols = ['Nom Dep', 'OCUPACION']
                
                # Convertir numéricas a float64 (como en el dataset original)
                for col in numeric_float_cols:
                    if col in df_input.columns:
                        df_input[col] = df_input[col].astype(float)
                
                # Convertir categóricas a string
                for col in categorical_cols:
                    if col in df_input.columns:
                        df_input[col] = df_input[col].astype(str)
                
                # --- TRANSFORMAR DATOS ---
                with st.spinner("Procesando datos..."):
                    X_processed = preprocessor.transform(df_input)
                
                # Realizar predicción
                with st.spinner("Generando predicción..."):
                    prediction = model.predict(X_processed)[0]
                    prediction_proba = model.predict_proba(X_processed)[0] if hasattr(model, 'predict_proba') else None
                
                # Obtener nombre de la clase
                class_names = preprocessor.label_encoder.classes_
                predicted_class = class_names[prediction]
                
                st.markdown("---")
                st.success("✅ Predicción completada exitosamente")
                
                # Mostrar resultado
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.markdown("### 🎯 Resultado de la Predicción")
                    st.markdown(f"## **{predicted_class}**")
                    
                    if prediction_proba is not None:
                        confidence = prediction_proba[prediction] * 100
                        st.metric("Confianza", f"{confidence:.2f}%")
                        
                        # Mostrar probabilidades de todas las clases
                        st.markdown("#### Probabilidades por Clase:")
                        prob_df = pd.DataFrame({
                            'Clase': class_names,
                            'Probabilidad': [f"{p*100:.2f}%" for p in prediction_proba]
                        })
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                # Información adicional
                st.markdown("---")
                st.info("""
                **ℹ️ Nota Importante**: Esta predicción es generada por un modelo de Machine Learning 
                entrenado con datos históricos y debe ser utilizada como herramienta de apoyo. 
                No sustituye el criterio médico profesional.
                """)
                
            except Exception as e:
                st.error(f"❌ Error al realizar la predicción: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏥 Sistema de Predicción de Enfermedades de Alto Costo | 
    Desarrollado con Streamlit | 
    Datos: DANE Colombia 2019</p>
</div>
""", unsafe_allow_html=True)
