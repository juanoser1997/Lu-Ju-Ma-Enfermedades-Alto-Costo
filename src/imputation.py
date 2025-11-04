"""
Módulo para experimentos de imputación de valores faltantes.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ImputationExperiment:
    """
    Clase para experimentar con diferentes estrategias de imputación.
    """
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, numeric_cols: List[str], 
                 categorical_cols: List[str]):
        """
        Inicializa el experimento de imputación.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        numeric_cols : List[str]
            Lista de columnas numéricas
        categorical_cols : List[str]
            Lista de columnas categóricas
        """
        self.X = X.copy()
        self.y = y.copy()
        self.numeric_cols = [c for c in numeric_cols if c in X.columns]
        self.categorical_cols = [c for c in categorical_cols if c in X.columns]
        self.results = {}
    
    def impute_mean(self) -> pd.DataFrame:
        """Imputa con la media para numéricas."""
        X_imp = self.X.copy()
        if self.numeric_cols:
            imputer = SimpleImputer(strategy='mean')
            X_imp[self.numeric_cols] = imputer.fit_transform(X_imp[self.numeric_cols])
        return X_imp
    
    def impute_median(self) -> pd.DataFrame:
        """Imputa con la mediana para numéricas."""
        X_imp = self.X.copy()
        if self.numeric_cols:
            imputer = SimpleImputer(strategy='median')
            X_imp[self.numeric_cols] = imputer.fit_transform(X_imp[self.numeric_cols])
        return X_imp
    
    def impute_mode(self) -> pd.DataFrame:
        """Imputa con la moda para categóricas."""
        X_imp = self.X.copy()
        if self.categorical_cols:
            imputer = SimpleImputer(strategy='most_frequent')
            X_imp[self.categorical_cols] = imputer.fit_transform(X_imp[self.categorical_cols])
        return X_imp
    
    def impute_knn(self, n_neighbors: int = 5) -> pd.DataFrame:
        """
        Imputa con KNN.
        
        Parameters:
        -----------
        n_neighbors : int
            Número de vecinos
        """
        X_imp = self.X.copy()
        
        # KNN solo funciona con numéricas
        if self.numeric_cols:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            X_imp[self.numeric_cols] = imputer.fit_transform(X_imp[self.numeric_cols])
        
        # Imputar categóricas con moda
        if self.categorical_cols:
            for col in self.categorical_cols:
                X_imp[col].fillna(X_imp[col].mode()[0] if not X_imp[col].mode().empty else 'UNKNOWN', 
                                 inplace=True)
        
        return X_imp
    
    def impute_iterative(self, max_iter: int = 10) -> pd.DataFrame:
        """
        Imputa con IterativeImputer (MICE).
        
        Parameters:
        -----------
        max_iter : int
            Número máximo de iteraciones
        """
        X_imp = self.X.copy()
        
        if self.numeric_cols:
            imputer = IterativeImputer(max_iter=max_iter, random_state=42, 
                                      verbose=0)
            X_imp[self.numeric_cols] = imputer.fit_transform(X_imp[self.numeric_cols])
        
        # Imputar categóricas con moda
        if self.categorical_cols:
            for col in self.categorical_cols:
                X_imp[col].fillna(X_imp[col].mode()[0] if not X_imp[col].mode().empty else 'UNKNOWN', 
                                 inplace=True)
        
        return X_imp
    
    def evaluate_imputation(self, X_imputed: pd.DataFrame, 
                          method_name: str) -> Dict:
        """
        Evalúa la calidad de la imputación usando cross-validation.
        
        Parameters:
        -----------
        X_imputed : pd.DataFrame
            Dataset imputado
        method_name : str
            Nombre del método
            
        Returns:
        --------
        Dict
            Métricas de evaluación
        """
        try:
            # Codificar categóricas para el modelo de evaluación
            from sklearn.preprocessing import LabelEncoder
            X_encoded = X_imputed.copy()
            
            label_encoders = {}
            for col in self.categorical_cols:
                if col in X_encoded.columns:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                    label_encoders[col] = le
            
            # Modelo simple para evaluar
            model = RandomForestClassifier(n_estimators=50, max_depth=5, 
                                          random_state=42, n_jobs=-1)
            
            # Cross-validation
            scores = cross_val_score(model, X_encoded, self.y, 
                                   cv=3, scoring='f1_weighted', n_jobs=-1)
            
            result = {
                'method': method_name,
                'mean_f1': scores.mean(),
                'std_f1': scores.std(),
                'remaining_nulls': X_imputed.isnull().sum().sum()
            }
            
            print(f"  {method_name}: F1={scores.mean():.4f} (±{scores.std():.4f}), "
                  f"Nulos restantes={result['remaining_nulls']}")
            
            return result
        
        except Exception as e:
            print(f"   Error evaluando {method_name}: {e}")
            return {
                'method': method_name,
                'mean_f1': 0.0,
                'std_f1': 0.0,
                'remaining_nulls': X_imputed.isnull().sum().sum()
            }
    
    def run_all_experiments(self) -> pd.DataFrame:
        """
        Ejecuta todos los experimentos de imputación.
        
        Returns:
        --------
        pd.DataFrame
            Resultados comparativos
        """
        print(" Experimentando con estrategias de imputación...\n")
        
        results = []
        
        # 1. Media (solo numéricas)
        print("1. Imputación con Media...")
        X_mean = self.impute_mean()
        X_mean_complete = X_mean.copy()
        for col in self.categorical_cols:
            X_mean_complete[col].fillna(X_mean_complete[col].mode()[0] 
                                       if not X_mean_complete[col].mode().empty else 'UNKNOWN', 
                                       inplace=True)
        results.append(self.evaluate_imputation(X_mean_complete, 'Media'))
        
        # 2. Mediana (solo numéricas)
        print("2. Imputación con Mediana...")
        X_median = self.impute_median()
        X_median_complete = X_median.copy()
        for col in self.categorical_cols:
            X_median_complete[col].fillna(X_median_complete[col].mode()[0] 
                                         if not X_median_complete[col].mode().empty else 'UNKNOWN', 
                                         inplace=True)
        results.append(self.evaluate_imputation(X_median_complete, 'Mediana'))
        
        # 3. Moda (categóricas)
        print("3. Imputación con Moda...")
        X_mode = self.X.copy()
        for col in self.numeric_cols:
            X_mode[col].fillna(X_mode[col].median(), inplace=True)
        for col in self.categorical_cols:
            X_mode[col].fillna(X_mode[col].mode()[0] 
                              if not X_mode[col].mode().empty else 'UNKNOWN', 
                              inplace=True)
        results.append(self.evaluate_imputation(X_mode, 'Moda'))
        
        # 4. KNN
        print("4. Imputación con KNN...")
        X_knn = self.impute_knn(n_neighbors=5)
        results.append(self.evaluate_imputation(X_knn, 'KNN'))
        
        # 5. Iterative (MICE)
        print("5. Imputación con Iterative (MICE)...")
        X_iter = self.impute_iterative(max_iter=10)
        results.append(self.evaluate_imputation(X_iter, 'Iterative'))
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('mean_f1', ascending=False)
        
        print(f"\n{'='*60}")
        print(" Resumen de Resultados:")
        print(results_df.to_string(index=False))
        
        self.results = results_df
        return results_df
    
    def get_best_strategy(self) -> str:
        """Retorna el nombre de la mejor estrategia."""
        if self.results is not None and not self.results.empty:
            return self.results.iloc[0]['method']
        return 'Mediana'


def apply_best_imputation(X: pd.DataFrame, numeric_cols: List[str], 
                         categorical_cols: List[str], 
                         strategy: str = 'median') -> pd.DataFrame:
    """
    Aplica la mejor estrategia de imputación seleccionada.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Dataset con valores faltantes
    numeric_cols : List[str]
        Columnas numéricas
    categorical_cols : List[str]
        Columnas categóricas
    strategy : str
        Estrategia a aplicar ('mean', 'median', 'knn', 'iterative')
        
    Returns:
    --------
    pd.DataFrame
        Dataset imputado
    """
    X_imp = X.copy()
    
    print(f" Aplicando imputación: {strategy}")
    
    # Numéricas
    if numeric_cols:
        numeric_cols_in_X = [c for c in numeric_cols if c in X.columns]
        
        if strategy in ['mean', 'median']:
            imputer = SimpleImputer(strategy=strategy)
            X_imp[numeric_cols_in_X] = imputer.fit_transform(X_imp[numeric_cols_in_X])
        
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            X_imp[numeric_cols_in_X] = imputer.fit_transform(X_imp[numeric_cols_in_X])
        
        elif strategy == 'iterative':
            imputer = IterativeImputer(max_iter=10, random_state=42, verbose=0)
            X_imp[numeric_cols_in_X] = imputer.fit_transform(X_imp[numeric_cols_in_X])
    
    # Categóricas (siempre con moda)
    if categorical_cols:
        categorical_cols_in_X = [c for c in categorical_cols if c in X.columns]
        for col in categorical_cols_in_X:
            mode_val = X_imp[col].mode()[0] if not X_imp[col].mode().empty else 'UNKNOWN'
            X_imp[col].fillna(mode_val, inplace=True)
    
    print(f"✓ Imputación completada. Nulos restantes: {X_imp.isnull().sum().sum()}")
    
    return X_imp
