import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_breast_cancer_data():
    try:
        data_dir = os.path.join(os.path.dirname(__file__))
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found at: {data_dir}")
            
        seer_file = os.path.join(data_dir, 'seer_data.csv')
        if not os.path.exists(seer_file):
            raise FileNotFoundError(f"SEER dataset file not found at: {seer_file}")
        
        print(f"Loading dataset from: {seer_file}")
        df = pd.read_csv(seer_file)
        
        df['target'] = (df['Status'] == 'Dead').astype(int)
        df = df.drop(['Status'], axis=1)
        
        X = df.drop(['target'], axis=1)
        y = df['target']
        
        if X.isnull().any().any():
            raise ValueError("Dataset contains missing values. Please handle missing values before proceeding.")
        if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError("All features must be numeric. Please ensure all feature columns contain numeric values.")
            
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        assert len(X_scaled) == len(y), f"X and y have different lengths: X={len(X_scaled)}, y={len(y)}"
        
        feature_info = {
            'feature_names': X.columns.tolist(),
            'scaler': scaler
        }
        
        print(f"Successfully loaded dataset with {len(X_scaled)} samples and {len(X.columns)} features")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        return X_scaled, y, feature_info
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
