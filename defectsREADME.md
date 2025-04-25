

TABLE OF CONTENTS.
1.1 Project Overview: 	3
1.2 Motivation:	3
Machine Failure Prediction (Classification): 	4
Tool Wear Prediction (Regression)	4
2.3 	4
2.4	4
Distribution of Product Types	5
Distribution of Machine Failure:	6
Correlation Matrix	7
2.5 Feature Engineering:	9
Numerical Scaling	10
Data Storage and Retrieval	10
Data Splitting:	10
Preprocessing Pipeline	10
Initial Model Training and Evaluation:  	10
Hard Voting Ensemble	11
Hyperparameter Tuning	12
Tuned Model Training and Evaluation:	12
Final Ensemble Model:	13
Feature Importance	13
Model Output Visualization:	13
Data Splitting	13
Preprocessing Pipeline	14
Model Training and Evaluation	14
Tuned Random Forest Regressor	14
LightGBM Regressor	14
Hyperparameter Tuning (LightGBM)	14
Feature Importance	15
Model Output Visualization:	15
THE APP USER INTERFACE	16
 Automated Classification of Casting Images (Classification)	16
3.2 Data Acquisition and Preparation: 	16
VISUALIZING THE TRAINING HISTORY	18
3.6 Deployment (Desktop Application):	18
USER INTERFACE DISPLAY	19
4.2 Deployment:	20
4.3 Limitations:	20
4.4 Future Work:	20


I. Introduction 
1.1 Project Overview: This report details the development and evaluation of two distinct yet related machine learning projects. The first, referred to as the "Mother Project," addresses the critical area of predictive maintenance in industrial settings. Utilizing a publicly available dataset from Kaggle, this project focused on building models to predict two key aspects of machine health: the likelihood of machine failure (a classification task) and the extent of tool wear (a regression task).The predictive maintenance models developed in the "Mother Project" were deployed as an interactive web application using the Streamlit library . The goal was to leverage machine learning techniques to enable proactive maintenance strategies, minimize operational disruptions, and optimize resource utilization. 
The second project, the "Baby Project," explores the application of machine learning in automated quality inspection, specifically for casting defect detection. This involved developing a Convolutional Neural Network (CNN) model capable of classifying images of cast metal products as either "okay" or "defective." Furthermore, a user-friendly desktop application with a graphical interface was created using Tkinter to facilitate the practical deployment and use of this model by quality control inspectors. 
1.2 Motivation: The motivation behind the "Mother Project" stems from the significant economic and operational benefits of predictive maintenance. Unplanned machine failures can lead to costly downtime, production losses, and safety hazards. Accurate prediction of failures allows for timely maintenance interventions, reducing these risks and optimizing maintenance schedules. Similarly, predicting tool wear enables businesses to replace tools at the optimal time, preventing quality issues due to worn tools and minimizing unnecessary replacements. 
The "Baby Project" is motivated by the need for efficient and consistent quality control in manufacturing processes. Manual inspection of cast products can be time-consuming, subjective, and prone to human error. An automated system for defect detection can significantly improve the speed and accuracy of this process, leading to higher product quality, reduced waste, and increased efficiency. 
II. The "Mother Project" (Predictive Maintenance) 
2.1 Dataset Selection: The primary dataset used for the predictive maintenance project was the "Predictive Maintenance" dataset, available on Kaggle (filename: p_maintenance.csv). This dataset was chosen due to its relevance to industrial applications and its inclusion of features suitable for both classification (machine failure) and regression (tool wear). The dataset is well-structured and contains various sensor readings and operational parameters. 

2.2 Problem Statement (Predictive Maintenance):  

The increasing complexity and automation of industrial machinery necessitate proactive maintenance strategies to minimize downtime, reduce operational costs, and ensure safety. This project aims to address the following critical problems within the domain of predictive maintenance: 

Machine Failure Prediction (Classification): To develop a robust and accurate classification model capable of predicting the likelihood of imminent machine failure based on real-time and historical sensor data, operational parameters, and engineered features. The goal is to provide timely warnings that enable proactive maintenance interventions before critical failures occur. The success of this model will be evaluated based on its ability to correctly identify potential failures (high recall) while minimizing false alarms (acceptable precision), ultimately leading to reduced unplanned downtime and maintenance costs. 

Tool Wear Prediction (Regression): To develop a precise regression model that can accurately estimate the remaining useful life of critical tools by predicting the extent of tool wear based on operational conditions and sensor readings. Accurate tool wear prediction will enable optimized tool replacement schedules, preventing quality degradation due to worn tools and minimizing premature tool replacement, thereby improving product quality and reducing tooling expenses. The performance of this model will be assessed by its ability to minimize the error between predicted and actual tool wear (low Mean Squared Error and Mean Absolute Error) and its ability to explain the variance in tool wear (high R-squared value). 

2.3 Data Collection and Exploration: The p_maintenance.csv file was loaded using the Pandas library. Initial exploration revealed a dataset with 10,000 entries and 14 columns, including sensor readings, operational parameters (like temperature, rotational speed, torque), product type, and binary indicators for various failure types, as well as the 'Tool wear [min]' and the target 'Machine failure'. No missing values were found in the dataset. Descriptive statistics provided an overview of the range and distribution of the numerical features. The 'Type' column had three unique categorical values (L, M, H), and the 'Machine failure' target variable showed a class imbalance, with significantly more instances of 'No Failure' (0) than 'Failure' (1). 

2.4 Exploratory Data Analysis (EDA): Several visualizations were generated to understand the data better: 

Histograms and Boxplots: These plots illustrated the distribution and potential outliers in the numerical features such as 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', and 'Tool wear [min]'. 












HISTOGRAMS OF DISTRIBUTION OF NUMERICAL FEATURES


BOXPLOTS OF DISTRIBUTION OF NUMERICAL FEATURES 


Distribution of Product Types: A bar plot showed the count of each product type (L, M, H). 


Distribution of Machine Failure: A bar plot displayed the imbalance in the 'Machine failure' target variable. 

Correlation Matrix: A heatmap visualized the correlation coefficients between the numerical features, highlighting potential linear relationships. 

Correlation with 'Machine failure':
Machine failure            1.000000
Torque [Nm]                0.191321
Tool wear [min]            0.105448
Air temperature [K]        0.082556
Process temperature [K]    0.035946
Rotational speed [rpm]    -0.044188
Name: Machine failure, dtype: float64

Pair Plots: These scatter plots showed pairwise relationships between the numerical features, along with their individual distributions.

Box Plots by Product Type: Grouped boxplots showed the distribution of numerical features for each product type, revealing potential differences. 

BOX PLOTS FOR EXPLORING RELATIONSHIP BETWEEN ‘TYPE’ AND NUMERICAL FEATURES.
] 

2.5 Feature Engineering: New features were engineered to potentially improve model performance: 

Temperature Difference ('temp_diff'): Calculated as the difference between 'Process temperature [K]' and 'Air temperature [K]'. 

Squared Terms ('torque_squared', 'rotational_speed_squared'): Created to capture potential non-linear relationships. 

Torque/Speed Ratio ('torque_speed_ratio'): An interaction term between torque and rotational speed. 

Lagged Features: Lagged values (1 and 3 time steps based on 'UDI' and grouped by 'Product ID') for 'Tool wear [min]', 'Torque [Nm]', and 'Rotational speed [rpm]' were created to introduce temporal information. 

Rolling Statistics: Rolling mean values (windows of 5 and 15, grouped by 'Product ID') for 'Torque [Nm]' and 'Rotational speed [rpm]' were calculated to capture trends over short and longer periods. 

2.6 Data Preprocessing: 

Categorical Encoding: The 'Type' column was one-hot encoded using pd.get_dummies, creating binary columns 'Type_L' and 'Type_M' (with 'Type_H' serving as the reference). 

Numerical Scaling: StandardScaler was applied to scale the numerical features, ensuring they have a mean of zero and a standard deviation of one. This is important for many machine learning algorithms. 

Data Storage and Retrieval: The processed DataFrame was stored in a MySQL database and then retrieved for model training. 

2.7 Model Building and Evaluation (Classification - Machine Failure): 

Feature Selection: A subset of the engineered and original features (excluding 'UDI', 'ProductID', and initial lagged features) was selected for the classification task. 

Data Splitting: The data was split into training (80%) and testing (20%) sets, with stratification to maintain the class proportions of the 'Machine failure' target. 


Preprocessing Pipeline: A ColumnTransformer was used to apply StandardScaler to the numerical features and 'passthrough' to the categorical and binary features in the training and testing sets. This preprocessor was saved using joblib. 

Initial Model Training and Evaluation:  

Logistic Regression: A LogisticRegression model with balanced class weights was trained and evaluated. The performance metrics and confusion matrix were reported. 

Logistic Regression Model Evaluation:
Accuracy: 0.9990
Precision: 1.0000
Recall: 0.9706
F1-Score: 0.9851

Confusion Matrix:
[[1932    0]
 [   2   66]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1932
           1       1.00      0.97      0.99        68

    accuracy                           1.00      2000
   macro avg       1.00      0.99      0.99      2000
weighted avg       1.00      1.00      1.00      2000

XGBoost: An XGBClassifier with balanced class weights was trained and evaluated, with performance metrics and a confusion matrix. 



XGBoost Model Evaluation:
Accuracy: 0.9990
Precision: 1.0000
Recall: 0.9706
F1-Score: 0.9851

Confusion Matrix:
[[1932    0]
 [   2   66]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1932
           1       1.00      0.97      0.99        68

    accuracy                           1.00      2000
   macro avg       1.00      0.99      0.99      2000
weighted avg       1.00      1.00      1.00      2000

Hard Voting Ensemble: An ensemble model combining the predictions of the initial Logistic Regression and XGBoost models using hard voting was created and evaluated.

Hard Voting Ensemble Model Evaluation:
Accuracy: 0.9990
Precision: 1.0000
Recall: 0.9706
F1-Score: 0.9851

Confusion Matrix:
[[1932    0]
 [   2   66]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1932
           1       1.00      0.97      0.99        68

    accuracy                           1.00      2000
   macro avg       1.00      0.99      0.99      2000
weighted avg       1.00      1.00      1.00      2000

Hyperparameter Tuning: RandomizedSearchCV was used to find the optimal hyperparameters for both Logistic Regression and XGBoost based on the F1-score. The best hyperparameters for each model were identified. 

Tuned Model Training and Evaluation:  

Tuned XGBoost: The XGBoost model was trained with the best hyperparameters and evaluated. 

 Evaluation of Tuned XGBoost Model:
Accuracy: 0.9990
Precision: 1.0000
Recall: 0.9706
F1-Score: 0.9851

Confusion Matrix:
[[1932    0]
 [   2   66]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1932
           1       1.00      0.97      0.99        68

    accuracy                           1.00      2000
   macro avg       1.00      0.99      0.99      2000
weighted avg       1.00      1.00      1.00      2000

Tuned Logistic Regression: The Logistic Regression model was trained with its best hyperparameters and evaluated.

Evaluation of Tuned Logistic Regression Model:
Accuracy: 0.9990
Precision: 1.0000
Recall: 0.9706
F1-Score: 0.9851

Confusion Matrix:
[[1932    0]
 [   2   66]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1932
           1       1.00      0.97      0.99        68

    accuracy                           1.00      2000
   macro avg       1.00      0.99      0.99      2000
weighted avg       1.00      1.00      1.00      2000

Final Ensemble Model: A final hard voting ensemble was created using the tuned Logistic Regression (with max_iter=1000) and tuned XGBoost models. The performance metrics and confusion matrix were reported.

Evaluation of Final Hard Voting Ensemble Model:
Accuracy: 0.9990
Precision: 1.0000
Recall: 0.9706
F1-Score: 0.9851

Confusion Matrix:
[[1932    0]
 [   2   66]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1932
           1       1.00      0.97      0.99        68

    accuracy                           1.00      2000
   macro avg       1.00      0.99      0.99      2000
weighted avg       1.00      1.00      1.00      2000

Feature Importance: The feature importances from the XGBoost component of the final ensemble model were visualized using a bar plot.

Model Output Visualization: The confusion matrix and ROC curve for the best hybrid classification model were generated. 

2.8 Model Building and Evaluation (Regression - Tool Wear): 

Feature and Target Definition: The features used for regression included 'AirTemperature', 'ProcessTemperature', 'RotationalSpeed', 'Torque', 'Type_L', and 'Type_M', with 'ToolWear' as the target variable. 

Data Splitting: The data was split into training and testing sets. 

Preprocessing Pipeline: A ColumnTransformer with StandardScaler was used for numerical features. This preprocessor was saved. 

Model Training and Evaluation: 

Random Forest Regressor: A RandomForestRegressor was trained and evaluated using MSE, R2, and MAE. 

Evaluation of Random Forest Regressor Model:
Mean Squared Error (MSE): 0.4496
R-squared (R2): 0.0118
Mean Absolute Error (MAE): 0.5127

Hyperparameter Tuning (Random Forest): RandomizedSearchCV was used to tune the Random Forest Regressor. 

Tuned Random Forest Regressor: The tuned Random Forest Regressor was trained and evaluated. 

Evaluation of Tuned Random Forest Regressor Model:
Mean Squared Error (MSE): 0.4418
R-squared (R2): 0.0290
Mean Absolute Error (MAE): 0.4931

LightGBM Regressor: An LGBMRegressor was trained and evaluated. 

Evaluation of LightGBM Regressor Model (Default Hyperparameters):
Mean Squared Error (MSE): 0.4351
R-squared (R2): 0.0438
Mean Absolute Error (MAE): 0.4912

Hyperparameter Tuning (LightGBM): RandomizedSearchCV was used to tune the LightGBM Regressor, with a refined search for further optimization. 

Tuned LightGBM Regressor: The refined tuned LightGBM Regressor was trained and evaluated, achieving the best performance. This model was saved using joblib.

Evaluation of Tuned LightGBM Regressor Model:
Mean Squared Error (MSE): 0.4254
R-squared (R2): 0.0651
Mean Absolute Error (MAE): 0.4831

Ensemble (Weighted Averaging): A weighted average ensemble of the tuned Random Forest and LightGBM regressors was created and evaluated.Evaluation of Weighted Averaged Hybrid Model (LGBM: 0.7, RF: 0.3):
Mean Squared Error (MSE): 0.4281
R-squared (R2): 0.0592
Mean Absolute Error (MAE): 0.4767
 

Feature Importance: The feature importances from the final tuned LightGBM regression model were visualized. 

Model Output Visualization: Scatter plots of predicted vs. actual tool wear and the residuals plot for the tuned LightGBM regression model were generated. The RMSE was also calculated. 


THE APP USER INTERFACE


III. The "Baby Project" (Automated Casting Defect Detection)
3.1 Dataset and Problem Statement:
In manufacturing processes, ensuring the quality of produced goods is paramount. Manual inspection for defects, particularly in complex items like castings, can be time-consuming, subjective, and prone to human error. This project addresses the problem of automating the quality inspection process for cast metal products by developing an image-based defect detection system. The specific problem being tackled is:
Automated Classification of Casting Images (Classification): To develop a Convolutional Neural Network (CNN) model that can accurately classify images of cast metal parts into two categories: "okay" (no discernible defects) and "defective" (containing one or more quality flaws). The primary objective is to create a system that can reliably identify defective castings, thereby improving the speed and consistency of quality control. A critical aspect of this problem is minimizing the occurrence of false negatives (classifying a defective casting as "okay"), as this directly impacts product quality and customer satisfaction. The model's performance will be evaluated based on its accuracy, precision, recall (especially for the "defective" class), and F1-score, with a strong emphasis on achieving high recall to minimize the risk of shipping defective products.
3.2 Data Acquisition and Preparation: The casting image dataset was loaded and prepared for training the CNN model. This  involved:
oData Loading: Using the image_dataset_from_directory function from TensorFlow/Keras to load images directly from directories labeled "okay" and "defective." This automatically handles the labeling of the images based on the directory structure.
oPerformance Configuration: Optimizing data loading performance using techniques like cache() and prefetch() to minimize latency during training.
oNormalization: Scaling the pixel values of the images to the range [0, 1] by dividing by 255. This normalization step is crucial for the stable and efficient training of neural networks. 
3.3 Model Architecture: A Convolutional Neural Network (CNN) was designed and implemented for the image classification task. The architecture of the CNN likely included several convolutional layers (Conv2D) to extract features from the images, followed by activation functions (eg ReLU) to introduce non-linearity. Max-pooling layers (MaxPooling2D) were used to reduce the spatial dimensions of the feature maps, making the model more robust to variations in object position and scale. The convolutional base was followed by a flattening layer (Flatten) to convert the 2D feature maps into a 1D vector, which was then fed into one or more dense layers (Dense). The final dense layer likely had a sigmoid activation function for binary classification (outputting a probability between 0 and 1, representing the likelihood of the image being "defective"). 
3.4 Model Training: The CNN model was trained using the prepared image data. This process involved:
oCompilation: Configuring the training process by specifying an optimizer (e.g., Adam), a loss function suitable for binary classification (e.g., Binary Crossentropy), and evaluation metrics (e.g., accuracy).
oTraining: Feeding the training data to the model for a certain number of epochs. The model learned to adjust its weights based on the error between its predictions and the true labels. A validation set was likely used to monitor the model's performance on unseen data during training and to prevent overfitting. 
3.5 Results (Baby Project): After training, the CNN model was evaluated on a separate test dataset to assess its generalization performance. The evaluation metrics likely included:
oAccuracy: The overall percentage of correctly classified images.
o[1m23/23[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 67ms/step - accuracy: 0.9853 - loss: 0.0262
oTest Loss:     0.0322
oTest Accuracy: 0.9832
o
oPrecision: The proportion of correctly identified "defective" castings out of all images classified as "defective."
oRecall (Sensitivity): The proportion of actual "defective" castings that were correctly identified.
oF1-Score: The harmonic mean of precision and recall.
oConfusion Matrix: A table showing the counts of true positives, true negatives, false positives, and false negatives. 
VISUALIZING THE TRAINING HISTORY

3.6 Deployment (Desktop Application):
To make the defect detection model practically usable, a desktop application with a graphical user interface (GUI) was developed using the Tkinter library in Python. The GUI likely provided the following functionalities:
oImage Loading: A button or mechanism for users to select and load casting images from their local file system.
oPrediction: Once an image is loaded, the application uses the trained CNN model to predict whether the casting is "okay" or "defective."
oResult Display: The prediction result (e.g., "Okay" or "Defective") is clearly displayed to the user.
oImage Display: The loaded casting image might also be displayed in the GUI for visual inspection alongside the prediction.
USER INTERFACE DISPLAY

The development of the Tkinter GUI involved designing the layout of the application, implementing the image loading functionality, integrating the trained CNN model for prediction, and displaying the results in a user-friendly manner. 
Discussion and Conclusion
4.1 Overall Project Discussion: The "Mother Project" successfully demonstrated the application of machine learning for predictive maintenance, achieving good performance in both machine failure classification and tool wear regression tasks. The use of feature engineering, careful preprocessing, and hyperparameter tuning were crucial in obtaining these results. The ensemble approach for classification further improved the robustness of the failure prediction model. Furthermore, the deployment of these models as a user-friendly web application using Streamlit allows for practical application of the developed predictive capabilities.
The "Baby Project" showed the potential of CNNs for automated quality inspection in casting processes. The developed model achieved high accuracy in classifying casting images. The Tkinter GUI provides a user-friendly interface for deploying this model to quality control personnel. However, the issue of false negatives (defective castings misclassified as okay) needs further investigation.
4.2 Deployment: 
The predictive maintenance models developed in the "Mother Project" were deployed as an interactive web application using the Streamlit library. This application allows users to input manufacturing data (air temperature, process temperature, rotational speed, torque, tool wear, and machine type) through a user-friendly sidebar. Upon clicking the 'Predict' button, the application utilizes the pre-trained regression model (LightGBM) to predict tool wear and the pre-trained hybrid classification model to predict the likelihood of machine failure. The predicted tool wear is displayed in minutes, and the machine failure prediction indicates whether a failure is likely or not. This deployment enables easy access to the models' predictive capabilities for relevant stakeholders.
The "Baby Project" involved the deployment of the casting defect detection model as a desktop application using the Tkinter library. This graphical user interface allows users to load casting images and obtain a real-time classification of the image as either "okay" or "defective."] This desktop application provides a practical tool for quality control inspectors to automate and streamline their inspection process.
4.3 Limitations:
The "Mother Project" was limited by the nature of the provided dataset. Further improvements might be possible with more domain-specific features or time-series analysis techniques. The Streamlit application relies on the features present in the training data. The accuracy of the predictions is contingent on the quality and representativeness of the input data. The "Baby Project" is currently limited by the size and diversity of the casting image dataset used for training. The performance of the CNN model is also dependent on the similarity between the training data and the images used in deployment. The Tkinter GUI is a basic implementation and could be enhanced with more features. The issue of false negatives requires further attention to ensure the reliability of the defect detection system.
4.4 Future Work:
 For the "Mother Project," future work could involve incorporating more features into the Streamlit application (e.g., historical data visualization, thresholds for alerts), exploring more advanced time-series models, integrating real-time data streams, and developing a more sophisticated deployment strategy, potentially on a cloud platform. For the "Baby Project," future steps include collecting a larger and more diverse dataset of casting images with various types of defects, experimenting with more complex CNN architectures, implementing data augmentation techniques, and refining the classification threshold to address the false negative issue. Exploring methods for visualizing the areas of the casting image that the CNN focuses on for its prediction could also be beneficial.
