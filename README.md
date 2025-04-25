# Synthetic Data Generation and ML Model Training

A comprehensive Streamlit application for generating synthetic data, training machine learning models, and educational visualization of algorithm performance.

## Live Demo

**[Try the application online!](https://projectsyntheticdatageneration.streamlit.app/)**

## Overview

This application provides an end-to-end platform for:
1. Generating customizable synthetic datasets
2. Training and evaluating multiple machine learning classifiers 
3. Visualizing model performance and data characteristics
4. Learning about different ML algorithms through interactive education
5. Implementing and testing trained models

## Features

### Main App (`App.py`)
- Synthetic data generation with customizable feature distributions
- Support for multiple classifier algorithms with automatic preprocessing
- Real-time visualization of model performance metrics
- Model comparison and selection
- Dataset exploration and visualization tools
- Model saving and exporting functionality

### Algorithm Education (`pages/02_Algorithm_Education.py`)
- Detailed explanations of various ML classification algorithms
- Interactive demonstrations with customizable parameters
- Mathematical foundations and implementation details
- Algorithm strengths, limitations, and use cases
- Performance visualization across different data distributions

### Model Implementation (`pages/03_Model_Implementation.py`) 
- Upload and use previously trained models
- Real-time prediction with custom input values
- Model and scaler integration

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/synthetic_data_generation.git
cd synthetic_data_generation

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run App.py
```

## Requirements

- Python 3.7+
- streamlit>=1.28.0
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.2.0
- plotly>=5.13.0
- seaborn>=0.12.0
- matplotlib>=3.7.0
- joblib>=1.2.0

## Usage

### Generating Synthetic Data
1. Define features and their distributions
2. Configure class characteristics
3. Set sample size and other generation parameters
4. Generate and explore your synthetic dataset

### Training Models
1. Select classifier algorithms to evaluate
2. Configure training parameters (test split, etc.)
3. Train models and view performance metrics
4. Compare model results through interactive visualizations

### Educational Resources
1. Navigate to the Algorithm Education page
2. Select an algorithm to learn about
3. Interact with the demo to see how parameters affect performance
4. Examine mathematical foundations and implementation details

### Model Implementation
1. Upload previously saved model and scaler files
2. Input feature values or generate random test values
3. Make predictions and view results

## Project Structure

```
synthetic_data_generation/
├── App.py                  # Main application
├── models/                 # Directory for saved models
├── pages/                  # Additional application pages
│   ├── 02_Algorithm_Education.py    # Educational content about ML algorithms
│   └── 03_Model_implementation.py   # Model deployment and usage interface
├── temp_uploads/           # Temporary directory for file uploads
└── requirements.txt        # Project dependencies
```