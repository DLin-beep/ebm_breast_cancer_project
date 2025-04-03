def evaluate_model(model, X_test, y_test, feature_info):
    """
    Evaluate the model with a strong focus on false negative rates and sensitivity.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate weighted error (matching training penalty)
    weighted_error = (25 * fn + fp) / (25 * (fn + tp) + (tn + fp))
    
    # Calculate metrics
    sensitivity = recall_score(y_test, y_pred)
    specificity = tn / (tn + fp)
    ppv = precision_score(y_test, y_pred)
    npv = tn / (tn + fn)
    
    # Calculate ROC and PR curves
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    # Create evaluation results dictionary
    evaluation_results = {
        'confusion_matrix': cm,
        'metrics': {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'weighted_error': weighted_error,
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'auc_pr': average_precision_score(y_test, y_pred_proba)
        },
        'curves': {
            'roc': {'fpr': fpr, 'tpr': tpr},
            'pr': {'precision': precision, 'recall': recall}
        },
        'predictions': {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    }
    
    # Print detailed evaluation results
    print("\nModel Evaluation Results:")
    print("-" * 50)
    print(f"Sensitivity (True Positive Rate): {sensitivity:.3f}")
    print(f"Specificity (True Negative Rate): {specificity:.3f}")
    print(f"Positive Predictive Value (Precision): {ppv:.3f}")
    print(f"Negative Predictive Value: {npv:.3f}")
    print(f"Weighted Error (25x FN penalty): {weighted_error:.3f}")
    print(f"ROC AUC: {evaluation_results['metrics']['auc_roc']:.3f}")
    print(f"PR AUC: {evaluation_results['metrics']['auc_pr']:.3f}")
    
    print("\nConfusion Matrix:")
    print("-" * 50)
    print("                 Predicted Negative  Predicted Positive")
    print(f"Actual Negative       {tn:8d}           {fp:8d}")
    print(f"Actual Positive       {fn:8d}           {tp:8d}")
    
    print("\nDetailed False Negative Analysis:")
    print("-" * 50)
    fn_rate = fn / (fn + tp)
    print(f"False Negative Rate: {fn_rate:.3f}")
    print(f"Number of False Negatives: {fn}")
    print(f"Missed Malignant Cases: {fn}/{fn + tp} ({fn_rate*100:.1f}%)")
    
    # Create and save evaluation plots
    [... keep existing plotting code ...]
    
    return evaluation_results 