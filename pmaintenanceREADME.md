# Predictive Maintenance using Machine Learning

## Overview

This project focuses on applying machine learning techniques to predict two key aspects of machine health in an industrial setting: the likelihood of machine failure (a classification task) and the extent of tool wear (a regression task). The goal is to enable proactive maintenance strategies, minimize operational disruptions, and optimize resource utilization. This project utilizes the "Predictive Maintenance" dataset from Kaggle.

## Problem Statement

This project addresses two main problems:

* **Machine Failure Prediction (Classification):** To develop a model that can accurately predict whether a machine will fail based on sensor data and operational conditions.
* **Tool Wear Prediction (Regression):** To develop a model that can accurately predict the remaining tool wear based on operational parameters.

## Dataset

The "Predictive Maintenance" dataset, available on Kaggle, was used for this project. It contains sensor readings, operational parameters (temperature, rotational speed, torque), product type, failure type indicators, tool wear in minutes, and a binary target variable indicating machine failure.

## Methodology

The following steps were undertaken in this project:

1.  **Data Collection and Exploration:** Loading the dataset using Pandas and performing initial checks for missing values and data characteristics.
2.  **Exploratory Data Analysis (EDA):** Visualizing data distributions, relationships between features, and the target variable using histograms, boxplots, scatter plots, correlation matrices, and grouped boxplots.
3.  **Feature Engineering:** Creating new features such as temperature difference, squared terms for torque and rotational speed, torque/speed ratio, lagged features, and rolling statistics to potentially improve model performance.
4.  **Data Preprocessing:**
    * One-hot encoding the categorical 'Type' column.
    * Scaling numerical features using `StandardScaler`.
5.  **Model Building and Evaluation (Classification - Machine Failure):**
    * Training and evaluating initial Logistic Regression and XGBoost models.
    * Creating and evaluating a hard voting ensemble of these initial models.
    * Performing hyperparameter tuning for Logistic Regression and XGBoost using `RandomizedSearchCV`.
    * Training and evaluating the tuned Logistic Regression and XGBoost models.
    * Creating and evaluating a final hard voting ensemble using the tuned models.
    * Analyzing feature importance from the best classification model.
6.  **Model Building and Evaluation (Regression - Tool Wear):**
    * Training and evaluating a Random Forest Regressor and a LightGBM Regressor.
    * Performing hyperparameter tuning for both regression models using `RandomizedSearchCV`.
    * Training and evaluating the tuned Random Forest and LightGBM Regressors.
    * Creating and evaluating a weighted average ensemble of the tuned regression models.
    * Analyzing feature importance from the best regression model.
7.  **Deployment:** The trained predictive maintenance models were deployed as an interactive web application using the Streamlit library, allowing users to input manufacturing data and receive predictions for tool wear and machine failure.

## Results

The project yielded the following key results:

* **Machine Failure Classification:** The final hard voting ensemble of the tuned Logistic Regression and XGBoost models achieved high performance in predicting machine failure, as indicated by accuracy, precision, recall, and F1-score. The confusion matrix and ROC curve provided further insights into the model's performance.
* **Tool Wear Regression:** The tuned LightGBM Regressor demonstrated the best performance in predicting tool wear, with low Mean Squared Error (MSE) and Mean Absolute Error (MAE), and a high R-squared value. The predicted vs. actual values and residual plots illustrated the model's predictive capability.

## Deployment

The trained predictive maintenance models were deployed as a user-friendly web application using Streamlit. Users can input manufacturing parameters through a sidebar, and the application provides real-time predictions for both tool wear and the likelihood of machine failure.

## Libraries Used

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn (sklearn)
* xgboost
* lightgbm
* joblib
* streamlit

## Usage

To run the Streamlit application for deployment:

1.  Ensure you have all the required libraries installed (`pip install -r requirements.txt`).
2.  Save the trained model files (`preprocessor_regression.joblib`, `best_lgbm_regressor.joblib`, `preprocessor_classification.joblib`, `best_hybrid_classifier.joblib`) in the same directory as the Streamlit script.
3.  Run the Streamlit application from your terminal using the command: `streamlit run your_streamlit_app_name.py` (replace `your_streamlit_app_name.py` with the name of your Streamlit script).

## Model Files

The following trained model and preprocessor files are included:

* `preprocessor_regression.joblib`: StandardScaler fitted for the regression task.
* `best_lgbm_regressor.joblib`: The best trained LightGBM regression model.
* `preprocessor_classification.joblib`: ColumnTransformer fitted for the classification task.
* `best_hybrid_classifier.joblib`: The best trained hybrid classification model.

## Future Work

Future work could involve:

* Exploring more advanced time-series models.
* Incorporating sensor fusion techniques.
* Integrating real-time data streams.
* Developing a more sophisticated deployment strategy.
* Exploring the synergy between the predictive maintenance insights and the automated quality inspection capabilities of the "Baby Project."

## Author

Judy Wairimu Njuku
judynjuku7@gmail.com