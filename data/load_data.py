from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_breast_cancer_data():
    try:
        dataset = fetch_ucirepo(id=17)
        X = dataset.data.features
        y = dataset.data.targets.iloc[:, 0]
        y = y.map({'M': 1, 'B': 0})
        if X.isnull().any().any():
            raise ValueError("Dataset contains missing values")
        if not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError("All features must be numeric")
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
        return X_scaled, y, feature_info
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
