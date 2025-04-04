from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
import numpy as np
import joblib
import os
from datetime import datetime
from data.load_data import load_breast_cancer_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, confusion_matrix, roc_auc_score

def custom_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    weighted_error = (25 * fn + fp) / (25 * (fn + tp) + (tn + fp))
    return 1 - weighted_error

def train_and_save_model():
    print("Loading and preprocessing data...")
    X, y, feature_info = load_breast_cancer_data()
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Calculating sample weights...")
    sample_weights = np.ones(len(y_train))
    high_risk_indices = y_train == 1
    sample_weights[high_risk_indices] = 25
    print("\nClass weighting:")
    print("Low Risk (0): 1.0")
    print("High Risk (1): 25.0")
    
    hyperparams = {
        'learning_rate': 0.01,
        'max_bins': 128,
        'max_interaction_bins': 16,
        'interactions': 5,
        'outer_bags': 4,
        'inner_bags': 0,
        'validation_size': 0.2,
        'min_samples_leaf': 5,
        'max_rounds': 1000,
        'early_stopping_rounds': 50,
        'n_jobs': -1
    }
    
    print("Initializing model with optimized hyperparameters...")
    ebm = ExplainableBoostingClassifier(
        random_state=42,
        feature_names=feature_info['feature_names'],
        **hyperparams
    )
    
    print("\nPerforming 10-fold cross-validation with essential metrics...")
    scoring = {
        'custom_score': make_scorer(custom_score),
        'recall': 'recall',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = cross_validate(
        ebm,
        X_train,
        y_train,
        cv=10,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
        fit_params={'sample_weight': sample_weights}
    )
    
    print("\nCross-validation results:")
    for metric in scoring.keys():
        mean_score = cv_results[f'test_{metric}'].mean()
        std_score = cv_results[f'test_{metric}'].std()
        print(f"{metric}: {mean_score:.3f} (+/- {std_score * 2:.3f})")
    
    print("\nPerforming final model training...")
    ebm.fit(X_train, y_train, sample_weight=sample_weights)
    
    print("\nEvaluating on test set...")
    y_pred = ebm.predict(X_test)
    y_pred_proba = ebm.predict_proba(X_test)[:, 1]
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print("\nTest set performance:")
    for metric, score in test_metrics.items():
        print(f"{metric}: {score:.3f}")
    
    print("\nDetailed error analysis:")
    print(f"False Negatives (missed high risk): {fn}")
    print(f"False Positives (false alarms): {fp}")
    print(f"True Negatives (correct low risk): {tn}")
    print(f"True Positives (correct high risk): {tp}")
    
    if fn > 0:
        print("\nAnalyzing false negative cases:")
        fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
        fn_features = X_test.iloc[fn_indices]
        print("\nFeature values for false negative cases:")
        print(fn_features)
        print("\nPrediction probabilities for false negative cases:")
        for idx in fn_indices:
            print(f"Case {idx}: {y_pred_proba[idx]:.4f}")
    
    print("\nSaving model...")
    model_dir = "models/saved"
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"ebm_model_{timestamp}.pkl")
    
    model_data = {
        'model': ebm,
        'X_test': X_test,
        'y_test': y_test,
        'feature_info': feature_info,
        'training_metrics': {
            'cv_results': cv_results,
            'test_metrics': test_metrics,
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        },
        'hyperparameters': hyperparams,
        'sample_weights': {
            'low_risk': 1.0,
            'high_risk': 25.0
        }
    }
    
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to: {model_path}")
    return ebm, X_test, y_test, feature_info
