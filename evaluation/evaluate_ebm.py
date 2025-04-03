from interpret import show
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd

def create_feature_interaction_plot(ebm, feature1_name, feature2_name, feature1_idx, feature2_idx):
    global_explanation = ebm.explain_global()
    if hasattr(global_explanation, 'interaction_data'):
        interaction_data = global_explanation.interaction_data()
        interaction_key = f"{feature1_idx} {feature2_idx}"
        if interaction_key in interaction_data:
            data = interaction_data[interaction_key]
            scores = data['scores']
            x_values = data['left_names']
            y_values = data['right_names']
            fig = go.Figure(data=go.Heatmap(
                z=scores,
                x=x_values,
                y=y_values,
                colorscale='RdBu'
            ))
            fig.update_layout(
                title=f'Interaction between {feature1_name} and {feature2_name}',
                xaxis_title=feature1_name,
                yaxis_title=feature2_name,
                width=600,
                height=600
            )
            return fig
    return None

def create_feature_effect_plot(ebm, feature_name, feature_idx):
    global_explanation = ebm.explain_global()
    try:
        additive_terms = ebm.term_scores_[feature_idx]
        bins = ebm.preprocessor_.col_bin_edges_[feature_idx]
        x_values = []
        for i in range(len(bins) - 1):
            x_values.append((bins[i] + bins[i + 1]) / 2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values,
            y=additive_terms,
            mode='lines+markers',
            name='Feature Effect'
        ))
        fig.update_layout(
            title=f'Effect of {feature_name} on Prediction',
            xaxis_title=feature_name,
            yaxis_title='Impact on Prediction (log-odds)',
            showlegend=False,
            width=600,
            height=400
        )
        return fig
    except (IndexError, AttributeError):
        return None

def evaluate_model(ebm, X, y):
    assert len(X) == len(y), f"X and y have different lengths: X={len(X)}, y={len(y)}"
    y_pred = ebm.predict(X)
    y_pred_proba = ebm.predict_proba(X)[:, 1]
    assert len(y_pred) == len(y), f"Predictions and true labels have different lengths: pred={len(y_pred)}, true={len(y)}"
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    try:
        feature_names = ebm.feature_names
        term_scores = ebm.term_scores_
        if isinstance(term_scores, list):
            term_scores = np.array([np.mean(score) for score in term_scores])
        if len(feature_names) != len(term_scores):
            from sklearn.inspection import permutation_importance
            importance_scores = permutation_importance(
                ebm, X, y, n_repeats=10, random_state=42
            ).importances_mean
        else:
            importance_scores = np.abs(term_scores)
    except (AttributeError, ValueError):
        from sklearn.inspection import permutation_importance
        importance_scores = permutation_importance(
            ebm, X, y, n_repeats=10, random_state=42
        ).importances_mean
        feature_names = X.columns.tolist()
    evaluation_results = {
        'confusion_matrix': cm,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'false_negatives': fn,
            'false_positives': fp
        },
        'curves': {
            'roc': {'fpr': fpr, 'tpr': tpr},
            'pr': {'precision': precision_curve, 'recall': recall_curve}
        },
        'feature_names': feature_names,
        'importance_scores': importance_scores
    }
    print("\nModel Evaluation Results:")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"PR AUC: {pr_auc:.3f}")
    print("\nConfusion Matrix:")
    print("-" * 50)
    print("                 Predicted Negative  Predicted Positive")
    print(f"Actual Negative       {tn:8d}           {fp:8d}")
    print(f"Actual Positive       {fn:8d}           {tp:8d}")
    print("\nDetailed False Negative Analysis:")
    print("-" * 50)
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f"False Negative Rate: {fn_rate:.3f}")
    print(f"Number of False Negatives: {fn}")
    print(f"Missed Malignant Cases: {fn}/{fn + tp} ({fn_rate*100:.1f}%)")
    if fn > 0:
        print("\nAnalyzing False Negative Cases:")
        fn_indices = np.where((y == 1) & (y_pred == 0))[0]
        fn_features = X.iloc[fn_indices]
        print("\nFeature values for false negative cases:")
        print(fn_features)
    return evaluation_results
