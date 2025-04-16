from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
import numpy as np
import joblib
import os
from datetime import datetime
from data.load_data import load_breast_cancer_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

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
    malignant_indices = y_train == 1
    sample_weights[malignant_indices] = 25
    print("\nClass weighting:")
    print("Benign (0): 1.0")
    print("Malignant (1): 25.0")
    param_grid = {
        'learning_rate': [0.05],
        'max_bins': [128, 256],
        'max_interaction_bins': [16, 32],
        'interactions': [5, 10],
        'outer_bags': [4, 8],
        'inner_bags': [0],
        'validation_size': [0.2]
    }
    print("Initializing base model...")
    base_ebm = ExplainableBoostingClassifier(
        random_state=42,
        feature_names=feature_info['feature_names'],
        interactions=5,
        n_jobs=-1,
        min_samples_leaf=5,
        max_rounds=5000,
        early_stopping_rounds=100
    )
    custom_scorer = make_scorer(custom_score)
    print("\nPerforming grid search with 10-fold cross-validation...")
    grid_search = GridSearchCV(
        base_ebm,
        param_grid,
        cv=10,
        scoring={
            'custom_score': custom_scorer,
            'recall': 'recall',
            'precision': 'precision',
            'accuracy': 'accuracy'
        },
        refit='custom_score',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    print("\nTraining completed. Getting best model...")
    ebm = grid_search.best_estimator_
    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print("\nCalculating cross-validation scores...")
    cv_scores = cross_val_score(ebm, X_train, y_train, cv=10, scoring='recall')
    print(f"\nCross-validation recall scores: {cv_scores}")
    print(f"Average CV recall: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print("\nPerforming final model training...")
    ebm.fit(X_train, y_train, sample_weight=sample_weights)
    print("\nEvaluating on test set...")
    y_pred = ebm.predict(X_test)
    y_pred_proba = ebm.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("\nTest set performance:")
    print(f"Accuracy: {test_accuracy:.3f}")
    print(f"Precision: {test_precision:.3f}")
    print(f"Recall: {test_recall:.3f}")
    print(f"F1 Score: {test_f1:.3f}")
    print("\nDetailed error analysis:")
    print(f"False Negatives (missed malignant): {fn}")
    print(f"False Positives (false alarms): {fp}")
    print(f"True Negatives (correct benign): {tn}")
    print(f"True Positives (correct malignant): {tp}")
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
            'cv_scores': cv_scores,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        },
        'best_params': grid_search.best_params_,
        'sample_weights': {
            'benign': 1.0,
            'malignant': 25.0
        }
    }
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to: {model_path}")
    return ebm, X_test, y_test, feature_info
