# Diabetes Prediction Machine Learning Model

This repository hosts a machine learning model for predicting the likelihood of diabetes in individuals based on key health metrics. The model, trained on the provided dataset named **dataset.cnv**, demonstrates an accuracy of 78.664% in predicting diabetes.

## Overview

Diabetes is a prevalent health concern, and early prediction can play a pivotal role in effective management and prevention. This machine learning model leverages a dataset with 768 entries and 9 columns, including features such as pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, age, and the outcome (0 for non-diabetic, 1 for diabetic).

## Features

- **Machine Learning Algorithm**: The model employs a state-of-the-art machine learning algorithm fine-tuned for diabetes prediction.

- **Accuracy**: Achieving an accuracy of 78.664% on the test dataset, this model provides reliable predictions.

- **Input Features**: The model considers a comprehensive set of health metrics, ensuring a robust analysis for accurate diabetes predictions.

## Data

### Columns

1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. **BloodPressure**: Diastolic blood pressure (mm Hg)
4. **SkinThickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (mu U/ml)
6. **BMI**: Body mass index (weight in kg/(height in m)^2)
7. **DiabetesPedigreeFunction**: Diabetes pedigree function
8. **Age**: Age in years
9. **Outcome**: Class variable (0 if non-diabetic, 1 if diabetic)

### Usage

1. **Download the Dataset**: Access the dataset which is uploaded above.

2. **File Format**: The dataset is provided in the ".cnv" format, facilitating seamless integration for training and evaluation.

3. **Data Exploration**: Perform an exploratory data analysis to understand feature distributions before utilizing the dataset for model training.

### Dataset Structure

- **dataset.cnv**: The main dataset file with 768 rows and 9 columns.

### How to Use

1. **Training**: Utilize the provided Jupyter notebook (*diabetes_prediction_model.ipynb*) or script to train the model on your dataset.

2. **Prediction**: Leverage the trained model for diabetes prediction by providing relevant input features.

### How to Contribute

Contributions are welcome! Whether you're enhancing the model, adding features, or improving documentation, follow the standard GitHub workflow. Fork the repository, create a branch, make changes, and submit a pull request.

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
