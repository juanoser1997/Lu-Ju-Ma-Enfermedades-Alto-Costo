"""
Módulo para balanceo de clases con SMOTEENN.
"""

import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from collections import Counter
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class SMOTEENNBalancer:
    """
    Clase para aplicar SMOTEENN en diferentes niveles de intensidad.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Inicializa el balanceador.
        
        Parameters:
        -----------
        random_state : int
            Semilla aleatoria
        """
        self.random_state = random_state
        self.original_distribution = None
        self.balanced_distributions = {}
    
    def get_class_distribution(self, y: np.ndarray) -> Dict:
        """
        Obtiene la distribución de clases.
        
        Parameters:
        -----------
        y : np.ndarray
            Target
            
        Returns:
        --------
        Dict
            Distribución de clases
        """
        counter = Counter(y)
        total = sum(counter.values())
        
        distribution = {
            'counts': dict(counter),
            'percentages': {k: v/total*100 for k, v in counter.items()},
            'total': total
        }
        
        return distribution
    
    def calculate_sampling_strategy(self, y: np.ndarray, 
                                   level: str = 'low') -> Dict:
        """
        Calcula la estrategia de sampling según el nivel.
        
        Parameters:
        -----------
        y : np.ndarray
            Target
        level : str
            Nivel de balanceo ('low', 'medium', 'high')
            
        Returns:
        --------
        Dict
            Estrategia de sampling
        """
        counter = Counter(y)
        majority_class = max(counter, key=counter.get)
        majority_count = counter[majority_class]
        
        sampling_strategy = {}
        
        for class_label, count in counter.items():
            if class_label == majority_class:
                continue
            
            if level == 'low':
                # Incrementar hasta 10% de la clase mayoritaria
                target_count = int(majority_count * 0.10)
            elif level == 'medium':
                # Incrementar hasta 20% de la clase mayoritaria
                target_count = int(majority_count * 0.20)
            elif level == 'high':
                # Incrementar hasta 80% de la clase mayoritaria (más balanceado)
                target_count = int(majority_count * 0.80)
            else:
                target_count = count
            
            # No reducir clases, solo incrementar
            sampling_strategy[class_label] = max(count, target_count)
        
        return sampling_strategy
    
    def apply_smoteenn(self, X: np.ndarray, y: np.ndarray, 
                      level: str = 'medium') -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica SMOTEENN con el nivel especificado.
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            Target
        level : str
            Nivel de balanceo ('low', 'medium', 'high')
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            X e y balanceados
        """
        print(f"\n Aplicando SMOTEENN - Nivel: {level.upper()}")
        
        # Distribución original
        self.original_distribution = self.get_class_distribution(y)
        print(f"\n Distribución original:")
        for label, pct in self.original_distribution['percentages'].items():
            count = self.original_distribution['counts'][label]
            print(f"  Clase {label}: {count} ({pct:.2f}%)")
        
        # Calcular estrategia
        sampling_strategy = self.calculate_sampling_strategy(y, level)
        
        try:
            # Configurar SMOTEENN
            smote = SMOTE(sampling_strategy=sampling_strategy, 
                         random_state=self.random_state, k_neighbors=5)
            enn = EditedNearestNeighbours(n_neighbors=3)
            
            smoteenn = SMOTEENN(smote=smote, enn=enn, 
                               random_state=self.random_state)
            
            # Aplicar
            X_resampled, y_resampled = smoteenn.fit_resample(X, y)
            
            # Distribución balanceada
            balanced_dist = self.get_class_distribution(y_resampled)
            self.balanced_distributions[level] = balanced_dist
            
            print(f"\n Distribución balanceada:")
            for label, pct in balanced_dist['percentages'].items():
                count = balanced_dist['counts'][label]
                original_count = self.original_distribution['counts'].get(label, 0)
                change = count - original_count
                print(f"  Clase {label}: {count} ({pct:.2f}%) [{change:+d}]")
            
            print(f"\n Total: {self.original_distribution['total']} → "
                  f"{balanced_dist['total']} muestras")
            
            return X_resampled, y_resampled
        
        except Exception as e:
            print(f" Error aplicando SMOTEENN: {e}")
            print("  Retornando datos originales...")
            return X, y
    
    def apply_multiple_levels(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Aplica SMOTEENN en los tres niveles.
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            Target
            
        Returns:
        --------
        Dict
            Diccionario con datos balanceados por nivel
        """
        results = {}
        
        for level in ['low', 'medium', 'high']:
            X_balanced, y_balanced = self.apply_smoteenn(X, y, level)
            results[level] = {
                'X': X_balanced,
                'y': y_balanced,
                'distribution': self.balanced_distributions.get(level)
            }
        
        return results
    
    def get_summary(self) -> pd.DataFrame:
        """
        Obtiene un resumen de las distribuciones.
        
        Returns:
        --------
        pd.DataFrame
            Resumen comparativo
        """
        if not self.balanced_distributions:
            return pd.DataFrame()
        
        summary_data = []
        
        # Original
        for label, count in self.original_distribution['counts'].items():
            summary_data.append({
                'Level': 'Original',
                'Class': label,
                'Count': count,
                'Percentage': self.original_distribution['percentages'][label]
            })
        
        # Balanceadas
        for level, dist in self.balanced_distributions.items():
            for label, count in dist['counts'].items():
                summary_data.append({
                    'Level': level.capitalize(),
                    'Class': label,
                    'Count': count,
                    'Percentage': dist['percentages'][label]
                })
        
        df_summary = pd.DataFrame(summary_data)
        return df_summary


def compare_balancing_impact(X_original: np.ndarray, y_original: np.ndarray,
                            X_balanced: np.ndarray, y_balanced: np.ndarray,
                            model, level_name: str) -> Dict:
    """
    Compara el impacto del balanceo en el rendimiento del modelo.
    
    Parameters:
    -----------
    X_original : np.ndarray
        Features originales
    y_original : np.ndarray
        Target original
    X_balanced : np.ndarray
        Features balanceadas
    y_balanced : np.ndarray
        Target balanceado
    model : sklearn estimator
        Modelo a evaluar
    level_name : str
        Nombre del nivel de balanceo
        
    Returns:
    --------
    Dict
        Comparación de métricas
    """
    from sklearn.model_selection import cross_val_score
    
    print(f"\n Comparando impacto del balanceo: {level_name}")
    
    # Evaluar con datos originales
    scores_original = cross_val_score(model, X_original, y_original, 
                                     cv=3, scoring='f1_weighted', n_jobs=-1)
    
    # Evaluar con datos balanceados
    scores_balanced = cross_val_score(model, X_balanced, y_balanced, 
                                     cv=3, scoring='f1_weighted', n_jobs=-1)
    
    comparison = {
        'level': level_name,
        'original_f1_mean': scores_original.mean(),
        'original_f1_std': scores_original.std(),
        'balanced_f1_mean': scores_balanced.mean(),
        'balanced_f1_std': scores_balanced.std(),
        'improvement': scores_balanced.mean() - scores_original.mean(),
        'improvement_pct': (scores_balanced.mean() - scores_original.mean()) / scores_original.mean() * 100
    }
    
    print(f"  Original:   F1={scores_original.mean():.4f} (±{scores_original.std():.4f})")
    print(f"  Balanceado: F1={scores_balanced.mean():.4f} (±{scores_balanced.std():.4f})")
    print(f"  Mejora:     {comparison['improvement']:+.4f} ({comparison['improvement_pct']:+.2f}%)")
    
    return comparison


def get_optimal_k_neighbors(X: np.ndarray, y: np.ndarray) -> int:
    """
    Determina el número óptimo de vecinos para SMOTE.
    
    Parameters:
    -----------
    X : np.ndarray
        Features
    y : np.ndarray
        Target
        
    Returns:
    --------
    int
        Número óptimo de vecinos
    """
    counter = Counter(y)
    min_samples = min(counter.values())
    
    # k_neighbors debe ser menor que el número de muestras de la clase minoritaria
    optimal_k = min(5, min_samples - 1)
    optimal_k = max(1, optimal_k)  # Al menos 1
    
    return optimal_k
