"""
Módulo para evaluación de modelos y cálculo de métricas.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, accuracy_score, 
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_proba: np.ndarray = None,
                     labels: List = None) -> Dict:
    """
    Calcula todas las métricas de evaluación.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Valores reales
    y_pred : np.ndarray
        Predicciones
    y_pred_proba : np.ndarray
        Probabilidades predichas (opcional)
    labels : List
        Etiquetas de clases
        
    Returns:
    --------
    Dict
        Diccionario con métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'recall_macro': recall_score(y_true, y_pred, average='macro')
    }
    
    # ROC-AUC para clasificación multiclase
    if y_pred_proba is not None:
        try:
            n_classes = y_pred_proba.shape[1]
            if n_classes > 2:
                # Binarizar para multiclase
                y_true_bin = label_binarize(y_true, classes=range(n_classes))
                metrics['roc_auc_weighted'] = roc_auc_score(y_true_bin, y_pred_proba, 
                                                           average='weighted', 
                                                           multi_class='ovr')
                metrics['roc_auc_macro'] = roc_auc_score(y_true_bin, y_pred_proba, 
                                                        average='macro', 
                                                        multi_class='ovr')
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except Exception as e:
            print(f" No se pudo calcular ROC-AUC: {e}")
            metrics['roc_auc_weighted'] = None
    
    return metrics


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                  target_names: List[str] = None) -> pd.DataFrame:
    """
    Genera reporte de clasificación como DataFrame.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Valores reales
    y_pred : np.ndarray
        Predicciones
    target_names : List[str]
        Nombres de las clases
        
    Returns:
    --------
    pd.DataFrame
        Reporte de clasificación
    """
    report_dict = classification_report(y_true, y_pred, 
                                       target_names=target_names, 
                                       output_dict=True)
    
    df_report = pd.DataFrame(report_dict).transpose()
    return df_report


def compare_models(results_dict: Dict, y_true: np.ndarray, 
                  metric: str = 'f1_weighted') -> pd.DataFrame:
    """
    Compara múltiples modelos por una métrica específica.
    
    Parameters:
    -----------
    results_dict : Dict
        Diccionario con resultados de modelos
    y_true : np.ndarray
        Valores reales
    metric : str
        Métrica para comparar
        
    Returns:
    --------
    pd.DataFrame
        Comparación de modelos ordenada
    """
    comparison_data = []
    
    for model_name, result in results_dict.items():
        if result.get('trained') and result['predictions'] is not None:
            metrics = calculate_metrics(y_true, result['predictions'], 
                                       result.get('probabilities'))
            
            row = {
                'Model': model_name,
                'F1_Weighted': metrics['f1_weighted'],
                'F1_Macro': metrics['f1_macro'],
                'Precision': metrics['precision_weighted'],
                'Recall': metrics['recall_weighted'],
                'Accuracy': metrics['accuracy']
            }
            
            if 'roc_auc_weighted' in metrics and metrics['roc_auc_weighted'] is not None:
                row['ROC_AUC'] = metrics['roc_auc_weighted']
            
            comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    if metric in df_comparison.columns:
        df_comparison = df_comparison.sort_values(metric, ascending=False)
    
    return df_comparison


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: List[str] = None, 
                         title: str = "Matriz de Confusión") -> go.Figure:
    """
    Crea gráfico de matriz de confusión con Plotly.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Valores reales
    y_pred : np.ndarray
        Predicciones
    labels : List[str]
        Nombres de las clases
    title : str
        Título del gráfico
        
    Returns:
    --------
    go.Figure
        Figura de Plotly
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [f"Clase {i}" for i in range(len(cm))]
    
    # Normalizar para porcentajes
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Crear texto con valores absolutos y porcentajes
    text = [[f"{cm[i][j]}<br>({cm_normalized[i][j]:.1f}%)" 
             for j in range(len(cm[0]))] 
            for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicción",
        yaxis_title="Valor Real",
        width=600,
        height=600
    )
    
    return fig


def plot_roc_curves(y_true: np.ndarray, y_pred_proba: np.ndarray,
                   class_names: List[str] = None,
                   title: str = "Curvas ROC") -> go.Figure:
    """
    Crea gráficos de curvas ROC para cada clase.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Valores reales
    y_pred_proba : np.ndarray
        Probabilidades predichas
    class_names : List[str]
        Nombres de las clases
    title : str
        Título del gráfico
        
    Returns:
    --------
    go.Figure
        Figura de Plotly
    """
    n_classes = y_pred_proba.shape[1]
    
    if class_names is None:
        class_names = [f"Clase {i}" for i in range(n_classes)]
    
    # Binarizar
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig = go.Figure()
    
    # Curva ROC para cada clase
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{class_names[i]} (AUC = {roc_auc:.3f})',
            line=dict(width=2)
        ))
    
    # Línea diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800,
        height=600,
        showlegend=True
    )
    
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame, 
                         metric: str = 'F1_Weighted',
                         title: str = "Comparación de Modelos") -> go.Figure:
    """
    Crea gráfico de barras comparando modelos.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame con comparación de modelos
    metric : str
        Métrica a graficar
    title : str
        Título del gráfico
        
    Returns:
    --------
    go.Figure
        Figura de Plotly
    """
    fig = px.bar(
        comparison_df,
        x='Model',
        y=metric,
        text=metric,
        title=title,
        color=metric,
        color_continuous_scale='Blues'
    )
    
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(
        xaxis_title="Modelo",
        yaxis_title=metric,
        showlegend=False,
        width=900,
        height=500
    )
    
    return fig


def plot_metrics_radar(metrics_dict: Dict[str, Dict], 
                      model_names: List[str] = None,
                      title: str = "Comparación de Métricas") -> go.Figure:
    """
    Crea gráfico de radar comparando múltiples métricas.
    
    Parameters:
    -----------
    metrics_dict : Dict[str, Dict]
        Diccionario con métricas por modelo
    model_names : List[str]
        Nombres de modelos a incluir
    title : str
        Título del gráfico
        
    Returns:
    --------
    go.Figure
        Figura de Plotly
    """
    if model_names is None:
        model_names = list(metrics_dict.keys())
    
    metrics_to_plot = ['f1_weighted', 'precision_weighted', 
                      'recall_weighted', 'accuracy']
    
    fig = go.Figure()
    
    for model_name in model_names:
        if model_name in metrics_dict:
            metrics = metrics_dict[model_name]
            values = [metrics.get(m, 0) for m in metrics_to_plot]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=[m.replace('_', ' ').title() for m in metrics_to_plot],
                fill='toself',
                name=model_name
            ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title,
        showlegend=True,
        width=700,
        height=700
    )
    
    return fig


def create_evaluation_report(model_name: str, y_true: np.ndarray, 
                            y_pred: np.ndarray, y_pred_proba: np.ndarray = None,
                            class_names: List[str] = None) -> Dict:
    """
    Crea un reporte completo de evaluación.
    
    Parameters:
    -----------
    model_name : str
        Nombre del modelo
    y_true : np.ndarray
        Valores reales
    y_pred : np.ndarray
        Predicciones
    y_pred_proba : np.ndarray
        Probabilidades (opcional)
    class_names : List[str]
        Nombres de las clases
        
    Returns:
    --------
    Dict
        Reporte completo
    """
    report = {
        'model_name': model_name,
        'metrics': calculate_metrics(y_true, y_pred, y_pred_proba),
        'classification_report': generate_classification_report(y_true, y_pred, class_names),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return report


def save_evaluation_results(results: Dict, output_dir: str):
    """
    Guarda resultados de evaluación en archivos.
    
    Parameters:
    -----------
    results : Dict
        Resultados a guardar
    output_dir : str
        Directorio de salida
    """
    import json
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Guardar métricas como JSON
    metrics_file = output_path / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(results.get('metrics', {}), f, indent=2)
    
    # Guardar reporte de clasificación como CSV
    if 'classification_report' in results:
        report_file = output_path / 'classification_report.csv'
        results['classification_report'].to_csv(report_file)
    
    print(f" Resultados guardados en: {output_dir}")
