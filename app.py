import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime
from data.load_data import load_breast_cancer_data
from evaluation.evaluate_ebm import evaluate_model

def load_latest_model():
    model_dir = "models/saved"
    if not os.path.exists(model_dir):
        return None
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not model_files:
        return None
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    return os.path.join(model_dir, latest_model)

def display_metrics(evaluation_results):
    st.header("Model Performance Metrics")
    metrics = evaluation_results['metrics']
    cm = evaluation_results['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    sensitivity = metrics['recall']
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Primary Metrics")
        st.metric("Sensitivity (TPR)", f"{sensitivity:.3f}")
        st.metric("Specificity (TNR)", f"{specificity:.3f}")
        st.metric("Precision (PPV)", f"{metrics['precision']:.3f}")
        st.metric("F1 Score", f"{metrics['f1']:.3f}")
    with col2:
        st.subheader("Additional Metrics")
        st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        st.metric("PR AUC", f"{metrics['pr_auc']:.3f}")
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.metric("False Negatives", f"{metrics['false_negatives']}")
    st.subheader("False Negative Analysis")
    st.write("This section focuses on missed malignant cases, which are critical in medical diagnosis.")
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    fn_metrics = {
        "False Negative Rate": f"{fn_rate:.3f}",
        "Missed Malignant Cases": f"{fn} out of {fn + tp}",
        "Miss Rate": f"{fn_rate*100:.1f}%"
    }
    st.table(pd.DataFrame([fn_metrics]).T.rename(columns={0: "Value"}))
    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(
        cm,
        index=['Actual Negative', 'Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )
    st.table(cm_df)

def main():
    st.set_page_config(
        page_title="Breast Cancer Risk Assessment Tool",
        page_icon="🎗️",
        layout="wide"
    )
    with st.sidebar:
        st.title("Model Information")
        model_path = load_latest_model()
        if model_path is None:
            st.error("No trained model found. Please run main.py first.")
            st.stop()
        try:
            model_data = joblib.load(model_path)
            ebm = model_data['model']
            X_test = model_data['X_test']
            y_test = model_data['y_test']
            feature_info = model_data['feature_info']
            training_metrics = model_data['training_metrics']
            st.write("### Model Performance Summary")
            st.write("This model has been trained to identify potential malignant breast tumors based on fine needle aspiration (FNA) characteristics.")
            st.write("#### Key Performance Indicators")
            st.write(f"**Accuracy**: {training_metrics['test_accuracy']:.1%}")
            st.write(f"**Sensitivity (True Positive Rate)**: {training_metrics['test_recall']:.1%}")
            cm = training_metrics['confusion_matrix']
            tn = cm['tn']
            fp = cm['fp']
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            st.write(f"**Specificity (True Negative Rate)**: {specificity:.1%}")
            st.write("#### Model Details")
            st.write(f"Last Updated: {datetime.fromtimestamp(os.path.getctime(model_path)).strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    st.title("🎗️ Breast Cancer Risk Assessment Tool")
    st.write("## Welcome to the Breast Cancer Risk Assessment Tool")
    st.write("This tool uses advanced machine learning to analyze fine needle aspiration (FNA) characteristics and assess the risk of malignancy in breast tumors.")
    st.write("### How to Use This Tool:")
    st.write("1. Navigate to the 'Make Prediction' tab")
    st.write("2. Adjust the sliders to match your patient's FNA characteristics")
    st.write("3. Click 'Predict' to get the risk assessment")
    st.write("### Understanding the Results:")
    st.write("- **Benign**: Low risk of malignancy")
    st.write("- **Malignant**: High risk of malignancy, requires further investigation")
    st.write("*Note: This tool is designed to assist in clinical decision-making but should not replace professional medical judgment.*")
    evaluation_results = evaluate_model(ebm, X_test, y_test)
    tab_names = ["Performance Overview", "Feature Analysis", "Make Prediction"]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        st.header("Model Performance Overview")
        st.subheader("Clinical Performance Metrics")
        st.write("These metrics help evaluate how well the model performs in clinical terms:")
        st.write("- **Sensitivity**: Ability to correctly identify malignant cases")
        st.write("- **Specificity**: Ability to correctly identify benign cases")
        st.write("- **Precision**: Proportion of positive predictions that are correct")
        st.write("- **Recall**: Proportion of actual malignant cases that are correctly identified")
        metrics = evaluation_results['metrics']
        cm = evaluation_results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        sensitivity = metrics['recall']
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sensitivity (True Positive Rate)", f"{sensitivity:.1%}")
            st.metric("Specificity (True Negative Rate)", f"{specificity:.1%}")
        with col2:
            st.metric("Precision (Positive Predictive Value)", f"{metrics['precision']:.1%}")
            st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        st.subheader("Clinical Decision Matrix")
        st.write("This matrix shows how the model's predictions compare to actual diagnoses:")
        st.write("- **True Negatives**: Correctly identified benign cases")
        st.write("- **False Positives**: Benign cases incorrectly flagged as malignant")
        st.write("- **False Negatives**: Malignant cases incorrectly identified as benign")
        st.write("- **True Positives**: Correctly identified malignant cases")
        cm_df = pd.DataFrame(
            cm,
            index=['Actual Benign', 'Actual Malignant'],
            columns=['Predicted Benign', 'Predicted Malignant']
        )
        st.table(cm_df)
        st.subheader("Critical Error Analysis")
        st.write("This section focuses on missed malignant cases, which are of particular concern in clinical practice.")
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        fn_metrics = {
            "Missed Malignant Cases": f"{fn} out of {fn + tp}",
            "Miss Rate": f"{fn_rate*100:.1f}%",
            "False Negative Rate": f"{fn_rate:.1%}"
        }
        st.table(pd.DataFrame([fn_metrics]).T.rename(columns={0: "Value"}))
    with tabs[1]:
        st.header("Feature Analysis")
        st.write("This section shows which characteristics are most important in determining malignancy risk.")
        feature_names = evaluation_results['feature_names']
        importance_scores = evaluation_results['importance_scores']
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=True)
        st.subheader("Feature Importance Ranking")
        st.write("The following chart shows which characteristics have the strongest influence on the model's predictions.")
        st.write("Higher values indicate stronger influence on the malignancy prediction.")
        fig = go.Figure([go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h'
        )])
        fig.update_layout(
            title="Feature Importance in Malignancy Prediction",
            xaxis_title="Influence on Prediction",
            yaxis_title="FNA Characteristics",
            height=600
        )
        st.plotly_chart(fig)
        st.subheader("Characteristic Distributions")
        st.write("These distributions show how the values of key characteristics differ between benign and malignant cases.")
        st.write("This can help identify typical ranges for each condition.")
        top_features = importance_df['Feature'].tail(5).tolist()
        for feature in top_features:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=X_test[feature][y_test == 0],
                name='Benign',
                opacity=0.75
            ))
            fig.add_trace(go.Histogram(
                x=X_test[feature][y_test == 1],
                name='Malignant',
                opacity=0.75
            ))
            fig.update_layout(
                title=f"Distribution of {feature}",
                xaxis_title=feature,
                yaxis_title="Number of Cases",
                barmode='overlay'
            )
            st.plotly_chart(fig)
    with tabs[2]:
        st.header("Make a Prediction")
        st.write("### Patient Assessment")
        st.write("Adjust the sliders below to match your patient's FNA characteristics.")
        st.write("The model will provide a risk assessment based on these values.")
        col1, col2 = st.columns(2)
        user_input = {}
        feature_ranges = {}
        X, _, _ = load_breast_cancer_data()
        for col in X.columns:
            feature_ranges[col] = {
                'min': float(X[col].min()),
                'max': float(X[col].max()),
                'mean': float(X[col].mean())
            }
        features = list(X.columns)
        mid = len(features) // 2
        with col1:
            for col in features[:mid]:
                user_input[col] = st.slider(
                    col,
                    feature_ranges[col]['min'],
                    feature_ranges[col]['max'],
                    feature_ranges[col]['mean'],
                    help=f"Range: [{feature_ranges[col]['min']:.2f}, {feature_ranges[col]['max']:.2f}]"
                )
        with col2:
            for col in features[mid:]:
                user_input[col] = st.slider(
                    col,
                    feature_ranges[col]['min'],
                    feature_ranges[col]['max'],
                    feature_ranges[col]['mean'],
                    help=f"Range: [{feature_ranges[col]['min']:.2f}, {feature_ranges[col]['max']:.2f}]"
                )
        if st.button("Assess Risk", type="primary"):
            X_new = pd.DataFrame([user_input])
            prediction = ebm.predict(X_new)[0]
            prediction_proba = ebm.predict_proba(X_new)[0]
            st.subheader("Risk Assessment Result")
            if prediction == 1:
                st.error(f"🔴 **High Risk of Malignancy** (Probability: {prediction_proba[1] * 100:.1f}%)")
                st.write("**Clinical Recommendation:**")
                st.write("- Consider further diagnostic testing")
                st.write("- Schedule follow-up consultation")
                st.write("- Document findings in patient record")
            else:
                st.success(f"🟢 **Low Risk of Malignancy** (Probability: {prediction_proba[0] * 100:.1f}%)")
                st.write("**Clinical Recommendation:**")
                st.write("- Continue routine monitoring")
                st.write("- Document findings in patient record")
                st.write("- Schedule regular follow-up as per protocol")
            st.subheader("Key Factors in This Assessment")
            st.write("The following characteristics had the strongest influence on this risk assessment:")
            local_explanation = ebm.explain_local(X_new)
            scores = local_explanation.scores[0]
            names = local_explanation.feature_names
            fig = go.Figure([go.Bar(x=scores, y=names, orientation='h')])
            fig.update_layout(
                title="Influence of Each Characteristic",
                xaxis_title="Impact on Prediction",
                yaxis_title="FNA Characteristics"
            )
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
