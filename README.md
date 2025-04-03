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

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd breast-cancer-risk-assessment
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
breast-cancer-risk-assessment/
├── app.py                 # Streamlit web application
├── main.py               # Main script for training and evaluation
├── data/
│   └── load_data.py      # Data loading and preprocessing
├── models/
│   ├── train_ebm.py      # Model training implementation
│   └── saved/            # Trained model storage
└── evaluation/
    └── evaluate_ebm.py   # Model evaluation utilities
```
