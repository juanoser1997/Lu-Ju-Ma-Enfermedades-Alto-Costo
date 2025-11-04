"""
Módulo para exportación de resultados y predicciones.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def export_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                      y_pred_proba: np.ndarray = None,
                      output_path: str = None, format: str = 'csv') -> pd.DataFrame:
    """
    Exporta predicciones a archivo.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Valores reales
    y_pred : np.ndarray
        Predicciones
    y_pred_proba : np.ndarray
        Probabilidades (opcional)
    output_path : str
        Ruta de salida
    format : str
        Formato ('csv' o 'json')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con predicciones
    """
    df_predictions = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred
    })
    
    if y_pred_proba is not None:
        n_classes = y_pred_proba.shape[1]
        for i in range(n_classes):
            df_predictions[f'prob_class_{i}'] = y_pred_proba[:, i]
        
        df_predictions['max_probability'] = y_pred_proba.max(axis=1)
        df_predictions['prediction_confidence'] = (y_pred_proba.max(axis=1) * 100).round(2)
    
    df_predictions['correct_prediction'] = (df_predictions['true_label'] == 
                                            df_predictions['predicted_label'])
    
    if output_path:
        if format == 'csv':
            df_predictions.to_csv(output_path, index=False)
        elif format == 'json':
            df_predictions.to_json(output_path, orient='records', indent=2)
        
        print(f" Predicciones exportadas a: {output_path}")
    
    return df_predictions


def export_metrics(metrics_dict: Dict, output_path: str, 
                  format: str = 'json') -> None:
    """
    Exporta métricas a archivo.
    
    Parameters:
    -----------
    metrics_dict : Dict
        Diccionario de métricas
    output_path : str
        Ruta de salida
    format : str
        Formato ('json' o 'csv')
    """
    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    elif format == 'csv':
        df_metrics = pd.DataFrame([metrics_dict])
        df_metrics.to_csv(output_path, index=False)
    
    print(f" Métricas exportadas a: {output_path}")


def export_comparison_table(comparison_df: pd.DataFrame, 
                           output_path: str) -> None:
    """
    Exporta tabla de comparación de modelos.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame con comparación
    output_path : str
        Ruta de salida
    """
    comparison_df.to_csv(output_path, index=False)
    print(f" Comparación exportada a: {output_path}")


def create_project_summary(results: Dict, output_dir: str) -> Dict:
    """
    Crea resumen completo del proyecto.
    
    Parameters:
    -----------
    results : Dict
        Resultados del proyecto
    output_dir : str
        Directorio de salida
        
    Returns:
    --------
    Dict
        Resumen del proyecto
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'project_name': 'Predicción de Enfermedades de Alto Costo',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': results.get('dataset_info', {}),
        'preprocessing_info': results.get('preprocessing_info', {}),
        'best_model': results.get('best_model', {}),
        'performance_metrics': results.get('metrics', {}),
        'training_details': results.get('training_details', {})
    }
    
    # Guardar como JSON
    summary_file = output_path / 'project_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Crear también un README con el resumen
    readme_file = output_path / 'RESULTS_SUMMARY.md'
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write("# Resumen de Resultados del Proyecto\n\n")
        f.write(f"**Fecha:** {summary['date']}\n\n")
        f.write("## Información del Dataset\n\n")
        
        dataset_info = summary.get('dataset_info', {})
        if dataset_info:
            f.write(f"- **Registros:** {dataset_info.get('n_samples', 'N/A')}\n")
            f.write(f"- **Features:** {dataset_info.get('n_features', 'N/A')}\n")
            f.write(f"- **Clases:** {dataset_info.get('n_classes', 'N/A')}\n\n")
        
        f.write("## Mejor Modelo\n\n")
        best_model = summary.get('best_model', {})
        if best_model:
            f.write(f"- **Modelo:** {best_model.get('name', 'N/A')}\n")
            f.write(f"- **F1-Score:** {best_model.get('f1_score', 'N/A')}\n")
            f.write(f"- **Accuracy:** {best_model.get('accuracy', 'N/A')}\n\n")
        
        f.write("## Métricas de Rendimiento\n\n")
        metrics = summary.get('performance_metrics', {})
        if metrics:
            for metric, value in metrics.items():
                f.write(f"- **{metric}:** {value}\n")
    
    print(f" Resumen del proyecto guardado en: {output_dir}")
    
    return summary


def export_model_for_deployment(model, preprocessor, label_encoder,
                               metadata: Dict, output_dir: str) -> None:
    """
    Exporta modelo y componentes necesarios para deployment.
    
    Parameters:
    -----------
    model : sklearn estimator
        Modelo entrenado
    preprocessor : Pipeline
        Pipeline de preprocesamiento
    label_encoder : LabelEncoder
        Encoder de labels
    metadata : Dict
        Metadatos del modelo
    output_dir : str
        Directorio de salida
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Guardar modelo
    model_file = output_path / 'model.joblib'
    joblib.dump(model, model_file)
    
    # Guardar preprocessor
    preprocessor_file = output_path / 'preprocessor.joblib'
    joblib.dump(preprocessor, preprocessor_file)
    
    # Guardar label encoder
    encoder_file = output_path / 'label_encoder.joblib'
    joblib.dump(label_encoder, encoder_file)
    
    # Guardar metadata
    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Crear archivo de configuración
    config = {
        'model_file': 'model.joblib',
        'preprocessor_file': 'preprocessor.joblib',
        'label_encoder_file': 'label_encoder.joblib',
        'metadata_file': 'metadata.json',
        'date_exported': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    config_file = output_path / 'deployment_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f" Modelo y componentes exportados para deployment en: {output_dir}")


def load_deployment_package(deployment_dir: str) -> Dict:
    """
    Carga un paquete de deployment completo.
    
    Parameters:
    -----------
    deployment_dir : str
        Directorio con el paquete
        
    Returns:
    --------
    Dict
        Diccionario con modelo y componentes
    """
    deployment_path = Path(deployment_dir)
    
    # Cargar configuración
    config_file = deployment_path / 'deployment_config.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Cargar componentes
    package = {
        'model': joblib.load(deployment_path / config['model_file']),
        'preprocessor': joblib.load(deployment_path / config['preprocessor_file']),
        'label_encoder': joblib.load(deployment_path / config['label_encoder_file']),
        'config': config
    }
    
    # Cargar metadata si existe
    metadata_file = deployment_path / config['metadata_file']
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            package['metadata'] = json.load(f)
    
    print(f"✓ Paquete de deployment cargado desde: {deployment_dir}")
    
    return package


def generate_predictions_report(df_predictions: pd.DataFrame, 
                               class_names: list = None) -> str:
    """
    Genera un reporte textual de las predicciones.
    
    Parameters:
    -----------
    df_predictions : pd.DataFrame
        DataFrame con predicciones
    class_names : list
        Nombres de las clases
        
    Returns:
    --------
    str
        Reporte en formato texto
    """
    total = len(df_predictions)
    correct = df_predictions['correct_prediction'].sum()
    accuracy = (correct / total) * 100
    
    report = []
    report.append("=" * 60)
    report.append("REPORTE DE PREDICCIONES")
    report.append("=" * 60)
    report.append(f"\nTotal de predicciones: {total}")
    report.append(f"Predicciones correctas: {correct}")
    report.append(f"Predicciones incorrectas: {total - correct}")
    report.append(f"Exactitud: {accuracy:.2f}%")
    
    if 'prediction_confidence' in df_predictions.columns:
        avg_confidence = df_predictions['prediction_confidence'].mean()
        report.append(f"Confianza promedio: {avg_confidence:.2f}%")
    
    report.append("\n" + "=" * 60)
    report.append("Distribución de predicciones:")
    report.append("=" * 60)
    
    pred_dist = df_predictions['predicted_label'].value_counts()
    for label, count in pred_dist.items():
        pct = (count / total) * 100
        label_name = class_names[label] if class_names else f"Clase {label}"
        report.append(f"{label_name}: {count} ({pct:.2f}%)")
    
    return "\n".join(report)
