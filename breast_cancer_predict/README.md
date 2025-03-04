# Breast Cancer Prediction using Machine Learning

## Introduction

This project focuses on predicting whether a breast cancer tumor is benign or malignant based on patient exam data. The goal is to leverage machine learning techniques to build a predictive model that can assist in early diagnosis and treatment planning. The project is implemented in a Jupyter Notebook and uses Python along with popular libraries such as Pandas, NumPy, Scikit-learn, and Matplotlib for data analysis and visualization.

## Project Overview

The project is divided into several key steps:

1. **Importing Libraries and Dataset**: The necessary Python libraries are imported, and the breast cancer dataset is loaded using Scikit-learn's `load_breast_cancer` function.

2. **Data Preparation and Preprocessing**: The dataset is preprocessed to ensure it is clean and ready for analysis. This includes loading the dataset, exploring its structure, and adding a target column for diagnosis (benign or malignant).

3. **Data Visualization**: Key features of the dataset are visualized using histograms to understand the distribution of different features across benign and malignant tumors.

4. **Model Training and Evaluation**: The dataset is split into training and testing sets. A Random Forest Classifier is used to train the model, and its performance is evaluated using accuracy, classification report, and confusion matrix.

5. **Hyperparameter Tuning**: GridSearchCV is employed to find the best hyperparameters for the Random Forest model, optimizing its performance.

## Key Features

- **Data Exploration**: The dataset is thoroughly explored to understand the distribution and characteristics of the features.
- **Visualization**: Histograms are used to visualize the distribution of key features such as mean radius, mean texture, and mean area.
- **Machine Learning Model**: A Random Forest Classifier is trained and evaluated for its ability to predict breast cancer diagnosis.
- **Hyperparameter Optimization**: GridSearchCV is used to fine-tune the model's hyperparameters, ensuring optimal performance.

## Requirements

To run this project, you will need the following Python libraries:

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
