"""
Módulo para construcción de pipelines de preprocesamiento.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, List
import joblib


class PreprocessingPipeline:
    """
    Clase para construcción de pipelines de preprocesamiento robustos.
    """
    
    def __init__(self, numeric_cols: List[str], categorical_cols: List[str],
                 scaler_type: str = 'standard', use_pca: bool = True, 
                 pca_variance: float = 0.95):
        """
        Inicializa el pipeline de preprocesamiento.
        
        Parameters:
        -----------
        numeric_cols : List[str]
            Columnas numéricas
        categorical_cols : List[str]
            Columnas categóricas
        scaler_type : str
            Tipo de escalador ('standard' o 'minmax')
        use_pca : bool
            Si se aplica PCA
        pca_variance : float
            Varianza explicada para PCA
        """
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.scaler_type = scaler_type
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.pipeline = None
        self.label_encoder = None
        self.feature_names = None
        
    def build_pipeline(self) -> Pipeline:
        """
        Construye el pipeline completo de preprocesamiento.
        
        Returns:
        --------
        Pipeline
            Pipeline de scikit-learn
        """
        # Transformaciones para numéricas
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler() if self.scaler_type == 'standard' else MinMaxScaler())
        ])
        
        # Transformaciones para categóricas
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ],
            remainder='drop'
        )
        
        # Pipeline completo
        if self.use_pca:
            self.pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('pca', PCA(n_components=self.pca_variance, random_state=42))
            ])
        else:
            self.pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor)
            ])
        
        return self.pipeline
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ajusta y transforma los datos.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target (opcional)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            X transformado y y codificado
        """
        if self.pipeline is None:
            self.build_pipeline()
        
        X_transformed = self.pipeline.fit_transform(X)
        
        # Guardar nombres de features
        self._extract_feature_names()
        
        # Codificar target si es categórico
        y_encoded = None
        if y is not None:
            if y.dtype == 'object':
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = y.values
        
        return X_transformed, y_encoded
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforma nuevos datos.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
            
        Returns:
        --------
        np.ndarray
            X transformado
        """
        if self.pipeline is None:
            raise ValueError("Pipeline no ha sido ajustado. Llama a fit_transform primero.")
        
        return self.pipeline.transform(X)
    
    def _extract_feature_names(self):
        """Extrae nombres de features después del preprocesamiento."""
        try:
            # Obtener nombres de features numéricas
            num_features = self.numeric_cols.copy()
            
            # Obtener nombres de features categóricas (OneHot)
            cat_features = []
            if self.categorical_cols:
                onehot = self.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
                cat_features = onehot.get_feature_names_out(self.categorical_cols).tolist()
            
            self.feature_names = num_features + cat_features
            
            # Si se usó PCA, los nombres son componentes principales
            if self.use_pca:
                n_components = self.pipeline.named_steps['pca'].n_components_
                self.feature_names = [f'PC{i+1}' for i in range(n_components)]
        
        except Exception as e:
            print(f" No se pudieron extraer nombres de features: {e}")
            self.feature_names = None
    
    def get_feature_names(self) -> List[str]:
        """Retorna nombres de las features transformadas."""
        return self.feature_names if self.feature_names else []
    
    def get_pca_info(self) -> Dict:
        """
        Obtiene información sobre PCA.
        
        Returns:
        --------
        Dict
            Información de PCA
        """
        if not self.use_pca or self.pipeline is None:
            return {}
        
        pca = self.pipeline.named_steps.get('pca')
        if pca is None:
            return {}
        
        return {
            'n_components': pca.n_components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'total_variance_explained': pca.explained_variance_ratio_.sum()
        }
    
    def save(self, filepath: str):
        """Guarda el pipeline."""
        joblib.dump({
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'feature_names': self.feature_names
        }, filepath)
        print(f"✓ Pipeline guardado en: {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Carga un pipeline guardado."""
        data = joblib.load(filepath)
        obj = cls(
            numeric_cols=data['numeric_cols'],
            categorical_cols=data['categorical_cols']
        )
        obj.pipeline = data['pipeline']
        obj.label_encoder = data['label_encoder']
        obj.feature_names = data['feature_names']
        print(f"✓ Pipeline cargado desde: {filepath}")
        return obj


def compare_pca_components(X: pd.DataFrame, numeric_cols: List[str], 
                          categorical_cols: List[str], y: pd.Series,
                          variance_levels: List[float] = [0.90, 0.95, 0.99]) -> pd.DataFrame:
    """
    Compara diferentes niveles de componentes principales.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    numeric_cols : List[str]
        Columnas numéricas
    categorical_cols : List[str]
        Columnas categóricas
    y : pd.Series
        Target
    variance_levels : List[float]
        Niveles de varianza a probar
        
    Returns:
    --------
    pd.DataFrame
        Comparación de resultados
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import time
    
    results = []
    
    for var_level in variance_levels:
        print(f"\n Probando PCA con {var_level*100}% varianza explicada...")
        
        # Crear pipeline
        prep = PreprocessingPipeline(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            use_pca=True,
            pca_variance=var_level
        )
        
        # Transformar datos
        start_time = time.time()
        X_transformed, y_encoded = prep.fit_transform(X, y)
        transform_time = time.time() - start_time
        
        # Obtener info de PCA
        pca_info = prep.get_pca_info()
        
        # Evaluar con modelo rápido
        model = RandomForestClassifier(n_estimators=50, max_depth=5, 
                                      random_state=42, n_jobs=-1)
        
        start_time = time.time()
        scores = cross_val_score(model, X_transformed, y_encoded, 
                                cv=3, scoring='f1_weighted', n_jobs=-1)
        eval_time = time.time() - start_time
        
        results.append({
            'variance_threshold': var_level,
            'n_components': pca_info.get('n_components', 0),
            'actual_variance': pca_info.get('total_variance_explained', 0),
            'f1_score': scores.mean(),
            'f1_std': scores.std(),
            'transform_time': transform_time,
            'eval_time': eval_time
        })
        
        print(f"  Componentes: {pca_info.get('n_components', 0)}")
        print(f"  Varianza real: {pca_info.get('total_variance_explained', 0)*100:.2f}%")
        print(f"  F1-Score: {scores.mean():.4f} (±{scores.std():.4f})")
    
    results_df = pd.DataFrame(results)
    return results_df


def encode_target(y: pd.Series, encoder: LabelEncoder = None) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Codifica la variable objetivo.
    
    Parameters:
    -----------
    y : pd.Series
        Variable objetivo
    encoder : LabelEncoder
        Encoder existente (opcional)
        
    Returns:
    --------
    Tuple[np.ndarray, LabelEncoder]
        Target codificado y encoder
    """
    if encoder is None:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
    else:
        y_encoded = encoder.transform(y)
    
    return y_encoded, encoder


def decode_target(y_encoded: np.ndarray, encoder: LabelEncoder) -> np.ndarray:
    """
    Decodifica la variable objetivo.
    
    Parameters:
    -----------
    y_encoded : np.ndarray
        Target codificado
    encoder : LabelEncoder
        Encoder utilizado
        
    Returns:
    --------
    np.ndarray
        Target decodificado
    """
    return encoder.inverse_transform(y_encoded)
