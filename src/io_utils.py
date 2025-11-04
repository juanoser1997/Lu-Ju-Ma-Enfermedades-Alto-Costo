"""
Módulo de utilidades para carga y manejo de datos.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Carga la hoja 'BD FINAL' del archivo Excel.
    
    Parameters:
    -----------
    file_path : str
        Ruta al archivo Excel
        
    Returns:
    --------
    pd.DataFrame
        Dataset principal
    """
    try:
        df = pd.read_excel(file_path, sheet_name='BD FINAL')
        print(f" Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
        return df
    except Exception as e:
        raise IOError(f"Error al cargar el dataset: {e}")


def load_dictionary(file_path: str) -> pd.DataFrame:
    """
    Carga y procesa la hoja 'DICCIONARIO' del archivo Excel.
    
    Parameters:
    -----------
    file_path : str
        Ruta al archivo Excel
        
    Returns:
    --------
    pd.DataFrame
        Diccionario de variables procesado
    """
    try:
        df_dict = pd.read_excel(file_path, sheet_name='DICCIONARIO')
        
        # El diccionario tiene un formato especial, necesitamos procesarlo
        # Extraer la información útil de las filas que contienen variables
        dict_data = []
        
        for idx, row in df_dict.iterrows():
            content = str(row.iloc[0])
            if content.startswith('V') and len(content.split()) >= 2:
                parts = content.split(maxsplit=2)
                if len(parts) >= 3:
                    var_id = parts[0]
                    var_name = parts[1]
                    var_desc = parts[2] if len(parts) > 2 else ''
                    dict_data.append({
                        'ID': var_id,
                        'VARIABLE': var_name,
                        'DESCRIPCION': var_desc
                    })
        
        df_dict_clean = pd.DataFrame(dict_data)
        print(f" Diccionario cargado: {len(df_dict_clean)} variables")
        return df_dict_clean
    
    except Exception as e:
        print(f" Error al cargar diccionario: {e}")
        return pd.DataFrame(columns=['ID', 'VARIABLE', 'DESCRIPCION'])


def get_variable_description(var_name: str, dictionary: pd.DataFrame) -> str:
    """
    Obtiene la descripción de una variable desde el diccionario.
    
    Parameters:
    -----------
    var_name : str
        Nombre de la variable
    dictionary : pd.DataFrame
        Diccionario de variables
        
    Returns:
    --------
    str
        Descripción de la variable
    """
    mask = dictionary['VARIABLE'].str.upper() == var_name.upper()
    if mask.any():
        return dictionary.loc[mask, 'DESCRIPCION'].iloc[0]
    return "Descripción no disponible"


def save_processed_data(df: pd.DataFrame, output_path: str, 
                       format: str = 'parquet') -> None:
    """
    Guarda datos procesados en formato parquet o CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a guardar
    output_path : str
        Ruta de salida
    format : str
        Formato ('parquet' o 'csv')
    """
    try:
        if format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'csv':
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        print(f" Datos guardados en: {output_path}")
    except Exception as e:
        raise IOError(f"Error al guardar datos: {e}")


def check_data_quality(df: pd.DataFrame, target_col: str = 'GRUPO') -> Dict:
    """
    Realiza verificaciones de calidad de datos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a verificar
    target_col : str
        Nombre de la columna objetivo
        
    Returns:
    --------
    Dict
        Diccionario con resultados de las verificaciones
    """
    report = {
        'shape': df.shape,
        'duplicates': df.duplicated().sum(),
        'total_nulls': df.isnull().sum().sum(),
        'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    
    # Información del target
    if target_col in df.columns:
        report['target_distribution'] = df[target_col].value_counts().to_dict()
        report['target_nulls'] = df[target_col].isnull().sum()
    
    return report


def identify_column_types(df: pd.DataFrame) -> Dict[str, list]:
    """
    Identifica tipos de columnas automáticamente.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a analizar
        
    Returns:
    --------
    Dict[str, list]
        Diccionario con listas de columnas por tipo
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Identificar numéricas que son categóricas (pocas categorías únicas)
    pseudo_categorical = []
    for col in numeric_cols:
        if df[col].nunique() < 20:  # Umbral para considerar categórica
            pseudo_categorical.append(col)
    
    return {
        'numeric': [c for c in numeric_cols if c not in pseudo_categorical],
        'categorical': categorical_cols,
        'pseudo_categorical': pseudo_categorical
    }


def remove_high_null_columns(df: pd.DataFrame, threshold: float = 0.5) -> Tuple[pd.DataFrame, list]:
    """
    Elimina columnas con un porcentaje alto de valores nulos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset original
    threshold : float
        Umbral de porcentaje de nulos (0.5 = 50%)
        
    Returns:
    --------
    Tuple[pd.DataFrame, list]
        Dataset sin columnas de alto nulo y lista de columnas eliminadas
    """
    null_pct = df.isnull().sum() / len(df)
    cols_to_drop = null_pct[null_pct > threshold].index.tolist()
    
    if cols_to_drop:
        print(f" Eliminando {len(cols_to_drop)} columnas con >{threshold*100}% nulos:")
        for col in cols_to_drop:
            print(f"  - {col}: {null_pct[col]*100:.1f}% nulos")
    
    df_clean = df.drop(columns=cols_to_drop)
    return df_clean, cols_to_drop


def detect_data_leakage_columns(df: pd.DataFrame, target_col: str = 'GRUPO') -> list:
    """
    Detecta columnas que pueden causar data leakage.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset a analizar
    target_col : str
        Nombre de la columna objetivo
        
    Returns:
    --------
    list
        Lista de columnas sospechosas
    """
    suspicious_cols = []
    
    # Patrones de columnas sospechosas
    leakage_patterns = [
        'CUENTA', 'NOMBRE', 'CIE', 'CAUSA', 'C_BAS', 'CAU_', 
        'CONS_', 'IDPROFCER', 'ASIS_MED'
    ]
    
    for col in df.columns:
        if col == target_col:
            continue
        
        col_upper = col.upper()
        
        # Verificar patrones
        for pattern in leakage_patterns:
            if pattern in col_upper:
                suspicious_cols.append(col)
                break
        
        # Verificar correlación perfecta o casi perfecta con el target
        if col in df.select_dtypes(include=[np.number]).columns:
            try:
                # Codificar temporalmente el target si es categórico
                if df[target_col].dtype == 'object':
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    target_encoded = le.fit_transform(df[target_col])
                else:
                    target_encoded = df[target_col]
                
                corr = np.corrcoef(df[col].fillna(0), target_encoded)[0, 1]
                if abs(corr) > 0.95:
                    if col not in suspicious_cols:
                        suspicious_cols.append(col)
            except:
                pass
    
    return list(set(suspicious_cols))
