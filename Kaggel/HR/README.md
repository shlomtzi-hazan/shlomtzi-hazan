# Heart Rate Condition Prediction

This repository contains a Python implementation for analyzing and predicting heart rate conditions using machine learning techniques. The code leverages datasets containing time-domain, frequency-domain, and non-linear features, as well as powerful classifiers such as Random Forest and Gradient Boosting.

## Overview

The objective of this project is to classify heart rate conditions into three categories: `interruption`, `no stress`, and `time pressure`. The implementation involves:
- Data preprocessing: loading, merging, and handling missing values.
- Exploratory Data Analysis (EDA): visualizing feature correlations and feature importance.
- Model training and evaluation: using Random Forest and Gradient Boosting classifiers.
- Feature importance analysis and model performance metrics.
- Visualization of confusion matrices and ROC curves.

## Data

The datasets used in this project are located at [Kaggle - Heart Rate Prediction Dataset](https://www.kaggle.com/datasets/saurav9786/heart-rate-prediction). Download the data and place the following files in your working directory:
- `time_domain_features_train.csv`
- `frequency_domain_features_train.csv`
- `heart_rate_non_linear_features_train.csv`
- `time_domain_features_test.csv`
- `frequency_domain_features_test.csv`
- `heart_rate_non_linear_features_test.csv`

## Instructions

### 1. Setup

1. Install required Python libraries:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib
   ```
2. Update the `folder` variable in the script with the location of your dataset files.

### 2. Run the Code

Execute the script to:
- Load and preprocess the data.
- Train and evaluate the models.
- Visualize feature correlations, feature importance, and performance metrics.

### 3. Results

#### Random Forest:
- **Cross-Validation Accuracy**: 0.96 (+/- 0.00)
- **Validation Accuracy**: 0.96
- **Test Accuracy**: 0.97

#### Gradient Boosting:
- **Validation Accuracy**: 0.91
- **Test Accuracy**: 0.92

Classification reports and confusion matrices provide detailed performance for each class. For instance, the Random Forest model achieves high precision and recall across all categories, with an overall test accuracy of 97%.

## Key Features

- **Feature Importance**: Both mutual information and model-specific methods (Random Forest and Gradient Boosting) are used to identify influential features.
- **ROC Curves**: Per-class ROC curves are plotted to analyze the discriminative power of the Gradient Boosting model.

## Visualization Highlights

1. **Feature Correlation Heatmap**: Displays the relationships between features.
2. **Feature Importance Charts**: Highlights which features contribute most to model predictions.
3. **Confusion Matrices**: Visualize the performance of models in terms of true and predicted labels.

## How to Interpret the Results

- Random Forest provides better accuracy and overall performance compared to Gradient Boosting.
- The `no stress` condition is predicted with the highest precision and recall, while `time pressure` is more challenging due to overlap with other categories.

## Recommendations

- Hyperparameters for the models are pre-tuned but should be re-optimized for specific datasets to prevent overfitting or underfitting.
- Consider adding more features or experimenting with other classifiers to further improve performance.
