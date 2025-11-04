"""
Módulo para entrenamiento y gestión de modelos.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from typing import Dict, List, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')


def get_baseline_models() -> Dict:
    """
    Retorna diccionario con los 5 modelos base configurados.
    
    Returns:
    --------
    Dict
        Diccionario de modelos
    """
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        )
    }
    
    return models


def train_baseline_models(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Entrena todos los modelos baseline.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Features de entrenamiento
    y_train : np.ndarray
        Target de entrenamiento
    X_test : np.ndarray
        Features de prueba
    y_test : np.ndarray
        Target de prueba
        
    Returns:
    --------
    Dict
        Diccionario con modelos entrenados y predicciones
    """
    models = get_baseline_models()
    results = {}
    
    print(" Entrenando modelos baseline...\n")
    
    for name, model in models.items():
        print(f"  Entrenando {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'trained': True
            }
            print(f"    ✓ Completado")
        
        except Exception as e:
            print(f"     Error: {e}")
            results[name] = {
                'model': None,
                'predictions': None,
                'probabilities': None,
                'trained': False,
                'error': str(e)
            }
    
    return results


def retrain_with_balanced_data(X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               level_name: str) -> Dict:
    """
    Reentrena modelos con datos balanceados.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Features de entrenamiento balanceadas
    y_train : np.ndarray
        Target de entrenamiento balanceado
    X_test : np.ndarray
        Features de prueba
    y_test : np.ndarray
        Target de prueba
    level_name : str
        Nombre del nivel de balanceo
        
    Returns:
    --------
    Dict
        Diccionario con modelos reentrenados
    """
    models = get_baseline_models()
    results = {}
    
    print(f"\n Reentrenando modelos con balanceo: {level_name}\n")
    
    for name, model in models.items():
        print(f"  Entrenando {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'trained': True,
                'level': level_name
            }
            print(f"    ✓ Completado")
        
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            results[name] = {
                'model': None,
                'predictions': None,
                'probabilities': None,
                'trained': False,
                'level': level_name,
                'error': str(e)
            }
    
    return results


def get_param_grids() -> Dict:
    """
    Retorna grids de hiperparámetros para cada modelo.
    
    Returns:
    --------
    Dict
        Diccionario de grids
    """
    param_grids = {
        'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        },
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'num_leaves': [31, 63, 127],
            'subsample': [0.8, 1.0]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }
    
    return param_grids


def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                        model_name: str, search_type: str = 'random',
                        cv: int = 3, n_iter: int = 20) -> Dict:
    """
    Ajusta hiperparámetros de un modelo.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Features de entrenamiento
    y_train : np.ndarray
        Target de entrenamiento
    model_name : str
        Nombre del modelo
    search_type : str
        Tipo de búsqueda ('grid' o 'random')
    cv : int
        Número de folds para cross-validation
    n_iter : int
        Número de iteraciones para RandomizedSearch
        
    Returns:
    --------
    Dict
        Resultados del ajuste
    """
    models = get_baseline_models()
    param_grids = get_param_grids()
    
    if model_name not in models:
        raise ValueError(f"Modelo {model_name} no reconocido")
    
    model = models[model_name]
    param_grid = param_grids[model_name]
    
    print(f"\n🔧 Ajustando hiperparámetros: {model_name}")
    print(f"   Método: {search_type.upper()}, CV={cv}")
    
    try:
        if search_type == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring='f1_weighted',
                n_jobs=-1, verbose=0
            )
        else:  # random
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv, 
                scoring='f1_weighted', random_state=42, n_jobs=-1, verbose=0
            )
        
        search.fit(X_train, y_train)
        
        result = {
            'model_name': model_name,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_model': search.best_estimator_,
            'cv_results': search.cv_results_,
            'success': True
        }
        
        print(f"   ✓ Mejor score: {search.best_score_:.4f}")
        print(f"   ✓ Mejores parámetros: {search.best_params_}")
        
        return result
    
    except Exception as e:
        print(f"    Error: {e}")
        return {
            'model_name': model_name,
            'best_params': None,
            'best_score': None,
            'best_model': None,
            'success': False,
            'error': str(e)
        }


def tune_all_models(X_train: np.ndarray, y_train: np.ndarray,
                   search_type: str = 'random', cv: int = 3) -> Dict:
    """
    Ajusta hiperparámetros de todos los modelos.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Features de entrenamiento
    y_train : np.ndarray
        Target de entrenamiento
    search_type : str
        Tipo de búsqueda
    cv : int
        Número de folds
        
    Returns:
    --------
    Dict
        Resultados de todos los modelos
    """
    models = get_baseline_models()
    results = {}
    
    print(" Ajustando hiperparámetros de todos los modelos...\n")
    
    for model_name in models.keys():
        result = tune_hyperparameters(X_train, y_train, model_name, 
                                     search_type, cv)
        results[model_name] = result
    
    return results


def save_model(model, filepath: str, metadata: Dict = None):
    """
    Guarda un modelo entrenado.
    
    Parameters:
    -----------
    model : sklearn estimator
        Modelo a guardar
    filepath : str
        Ruta de salida
    metadata : Dict
        Metadatos adicionales
    """
    data = {
        'model': model,
        'metadata': metadata or {}
    }
    
    joblib.dump(data, filepath)
    print(f" Modelo guardado en: {filepath}")


def load_model(filepath: str) -> Dict:
    """
    Carga un modelo guardado.
    
    Parameters:
    -----------
    filepath : str
        Ruta del modelo
        
    Returns:
    --------
    Dict
        Diccionario con modelo y metadatos
    """
    data = joblib.load(filepath)
    print(f" Modelo cargado desde: {filepath}")
    return data


def cross_validate_model(model, X: np.ndarray, y: np.ndarray, 
                        cv: int = 5) -> Dict:
    """
    Realiza validación cruzada de un modelo.
    
    Parameters:
    -----------
    model : sklearn estimator
        Modelo a evaluar
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    cv : int
        Número de folds
        
    Returns:
    --------
    Dict
        Resultados de la validación
    """
    scoring = ['f1_weighted', 'precision_weighted', 'recall_weighted', 'accuracy']
    
    results = {}
    for score in scoring:
        scores = cross_val_score(model, X, y, cv=cv, scoring=score, n_jobs=-1)
        results[score] = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    return results
