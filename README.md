# Admission Prediction Model

This repository contains the implementation of a machine learning model that predicts student admission chances based on various factors like GRE score, TOEFL score, university rating, and other related metrics.

## Project Overview

This project uses a **Neural Network** built with **TensorFlow** and **Keras** to predict the chances of admission to a university. The model is trained on a dataset that includes various features related to student applications. The goal is to predict the **probability of admission** (a continuous value) for a given student based on their scores and background.

## Requirements

Before running this code, make sure to install the necessary Python libraries:

pip install numpy pandas matplotlib tensorflow scikit-learn pickle

## Dataset

The dataset used in this project is **Admission_Predict_Ver1.1.csv**, which contains the following columns:

- **GRE Score**: The GRE score of the student.
- **TOEFL Score**: The TOEFL score of the student.
- **University Rating**: Rating of the university (1 to 5).
- **SOP**: Statement of Purpose strength (1 to 5).
- **LOR**: Letter of Recommendation strength (1 to 5).
- **CGPA**: Cumulative Grade Point Average.
- **Research**: Whether the student has done research (1 for yes, 0 for no).

## Code Explanation

### 1. Data Preprocessing
- The dataset is read using Pandas, and the first column (**Serial No.**) is dropped as it's not needed for prediction.
- Features (x) are separated from the target (y), where `x` represents the input features and `y` is the **Chance of Admit** column.
- The dataset is split into training and test sets (80% for training and 20% for testing).

### 2. Feature Scaling
- The input features are scaled using **MinMaxScaler** to normalize the data between 0 and 1, which helps improve the model's convergence during training.

### 3. Neural Network Model
A **Sequential Neural Network** model is created using Keras, with:
- **1st layer**: 16 neurons with ReLU activation.
- **2nd layer**: 8 neurons with ReLU activation.
- **Output layer**: A single neuron with a linear activation function for predicting the continuous admission probability.

The model is compiled using the **Adam optimizer** and the **mean squared error** loss function.

### 4. Model Training
- The model is trained for 100 epochs with a **validation split of 0.2** to track performance on unseen data.
- After training, the model is saved as `admission_model.h5`, and the scaler is saved as `scaler.pkl` for future use.

### 5. Model Evaluation
- Predictions are made on the test set, and the **R² score** (coefficient of determination) is calculated to evaluate the model's performance.

### 6. Training History Plot
- A plot is generated to visualize the training and validation loss over epochs, showing the model's learning progress.

## How to Use

### 1. Train the Model
- Ensure that you have the required libraries installed.
- Place the dataset (**Admission_Predict_Ver1.1.csv**) in the same directory.
- Run the script, and it will train the model and save the trained model and scaler for future use.

### 2. Making Predictions
You can use the saved `admission_model.h5` and `scaler.pkl` files to make predictions on new data.

## Files in This Repository

- **Admission_Predict_Ver1.1.csv**: The dataset used for training.
- **admission_model.h5**: The trained neural network model.
- **scaler.pkl**: The MinMaxScaler used for feature scaling.
- **model.py**: The Python script containing the model training code.
- **requirements.txt**: A text file listing the required Python packages.
