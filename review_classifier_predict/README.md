# Review Sentiment Classifier using Machine Learning

## Introduction

This project focuses on predicting whether a food/product/movie review is either positive or negative. The model is trained on the Sentiment Labelled Sentences Dataset from UCI Machine Learning Repository. The project is implemented in a Jupyter Notebook and uses Python along with popular libraries such as Pandas, Scikit-learn, WordCloud and Matplotlib for data analysis and visualization.

## Project Overview

The project is divided into several key steps:

1. **Importing Libraries and Dataset**: The necessary Python libraries are imported, and the labelled reviews dataset using the import files from google colab.

2. **Data Preparation and Preprocessing**: The dataset is preprocessed to ensure it is clean and ready for the vectorizer.

3. **Data Visualization**: Using the WordCloud library, we can see the words that appear the on the positive and negative reviews, and remove those that are not relevant for the analysis.

4. **Model Testing**: Four different types of model is tested to see what is the best fit for the exercise, for this i chose the Logistic Regression

5. **Hyperparameter Tuning**: GridSearchCV is employed to find the best hyperparameters for the Logistic Regression model, optimizing its performance.

6. **Model Training and Evaluation**: The dataset is split into training and testing sets. A Logistic Regression is used to train the model, and its performance is evaluated using classification report and confusion matrix.

7. **Deployment**: I used Joblib to save both the vectorizer and the model, and made the deployment using Gradle in the other Jupyter Notebook.

## Requirements

To run this project, you will need the following Python libraries:

- Pandas
- WordCloud
- Matplotlib
- Scikit-learn
- Spacy

You can install these libraries using pip:

```bash
