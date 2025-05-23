# Breast Cancer Risk Assessment Tool

A machine learning-based tool for assessing breast cancer risk using Explainable Boosting Machine (EBM).

## Overview

This project implements a breast cancer risk assessment tool using Explainable Boosting Machine (EBM), a glass-box machine learning model that provides both accurate predictions and interpretable results. The tool is designed to assist medical professionals in making informed decisions about breast cancer diagnosis.

## Features

- Interactive web interface using Streamlit
- Explainable machine learning model
- Detailed performance metrics and visualizations
- Feature importance analysis
- Real-time risk assessment

## Preview

Here's a visual tour of the tool's key features:

### Dashboard Overview
![Main Interface](pictures/screenshots/main_interface.png.png)
The main dashboard provides an intuitive interface for medical professionals to input and analyze patient data.

### Model Performance Visualization
![Performance Metrics](pictures/screenshots/performance_metrics.png.png)
Comprehensive visualization of the model's clinical performance metrics and decision matrix.

### Feature Analysis Dashboard
![Feature Analysis](pictures/screenshots/feature_analysis.png.png)
Interactive visualization of feature importance and their impact on predictions, helping understand which characteristics are most crucial for diagnosis.

*Note: The tool provides real-time analysis and explanations for each prediction, ensuring transparency in the decision-making process.*

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DLin-beep/ebm_breast_cancer_project.git
cd ebm_breast_cancer_project
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python main.py
```

2. Run the web application:
```bash
streamlit run app.py
```

## Project Structure

```
ebm_breast_cancer_project/
├── app.py                 # Streamlit web application
├── main.py               # Main script for training and evaluation
├── data/
│   ├── load_data.py      # Data loading and preprocessing
│   └── seer_data.csv     # SEER dataset
├── models/
│   ├── train_ebm.py      # Model training implementation
│   └── saved/            # Trained model storage
└── evaluation/
    └── evaluate_ebm.py   # Model evaluation utilities
```

## Model Performance

The model has been optimized for clinical use with the following characteristics:

- **High Sensitivity**: Prioritizes identifying malignant cases
- **Balanced Specificity**: Maintains reasonable accuracy for benign cases
- **Interpretable Predictions**: Provides feature importance and local explanations
- **Clinical Recommendations**: Includes actionable insights for medical professionals

### Limitations

- This tool is designed for screening purposes only and should not replace professional medical diagnosis
- The model's performance may vary with different populations or data distributions
- The tool requires standardized measurements for optimal performance
- False negatives, while minimized, are still possible and require clinical judgment

## Dataset Credits
 
 This project uses the Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI Machine Learning Repository:
 
 - **Dataset ID**: 17
 - **Source**: UCI Machine Learning Repository
 - **Citation**: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
 - **Original Dataset**: Breast Cancer Wisconsin (Diagnostic) Data Set
 - **Contributors**: Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian

## License

MIT License

Copyright (c) 2025 Derek Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
