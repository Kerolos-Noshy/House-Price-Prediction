# Housing-Price-Prediction
## Overview
This repository contains code for a housing price prediction project. The goal of this project is to predict housing prices based on various features of the house. This is a competition on Kaggle, Here is the link to this competition [Click here](https://www.kaggle.com/competitions/home-data-for-ml-course)

## Code Overview
The Jupyter Notebook consists of the following data science techniques:

### Data Preprocessing
Data preprocessing is a critical step in preparing the data for model training. In this project, we performed the following data preprocessing techniques:

- Handling Missing Values: We addressed missing values in the dataset by adopting two strategies. Columns with a high percentage of missing values (more than 90%) were dropped, and for other columns, we filled in missing values using appropriate strategies for numerical and categorical features.

- Encoding Categorical Features: Categorical features were encoded using Label Encoding to convert them into numerical representations that can be fed into machine learning algorithms.

### Exploratory Data Analysis (EDA)
EDA is a crucial step to gain insights into the data and understand its underlying patterns. In this project, we employed the following techniques:

- Visualization of Distributions: We used histograms and box plots to visualize the distributions of numerical features, allowing us to observe the spread and central tendencies of the data.

![distributions](https://github.com/Kerolos-Noshy/House-Price-Prediction/assets/101178275/3969687b-858d-409f-ac6d-c6d40a3bd735)

- Relationship Exploration: We used scatter plots and regplots to explore the relationships between numerical features and the target variable (SalePrice). This analysis helped us understand how different features are related to the target and identify potential correlations.

![regplot](https://github.com/Kerolos-Noshy/House-Price-Prediction/assets/101178275/9e004517-8e55-464a-b821-af35c7b3a6ce)

### Data Transformation
We applied log transformation to right-skewed numerical features to make their distributions more symmetric. This transformation is often beneficial for regression models, as it can lead to improved model performance by ensuring that the data follows a more normal distribution.

### Feature Engineering
Feature engineering involves creating new features from existing ones to enhance model performance. In this project, we performed the following feature engineering techniques:

- New Feature Creation: We created two new features: "TotalSF" represented the total square footage of the house by combining "1stFlrSF" and "2ndFlrSF", and "Last_Remod" indicated the number of years since the last remodeling.

### Data Splitting and Scaling
We split the dataset into training and testing sets using the train_test_split function from scikit-learn. This technique is essential for training the models on a subset of the data and then evaluating their performance on unseen data.

For numerical features, we applied standard scaling (z-scaling) using StandardScaler from scikit-learn. Scaling ensures that all numerical features are on a similar scale, preventing any single feature from dominating the model training process.

### Model Building and Evaluation
In this project, we built various regression models to predict housing prices based on the provided features. The models used include:

- Linear Regression
- Gradient Boosting
- Random Forest
- Extra Trees
- Support Vector Regression (SVR)

Each model was fitted with preprocessed data and evaluated using two key metrics:

- R-squared Score: It measures the goodness of fit of the model to the data, indicating how well the model explains the variance in the target variable.
- Mean Squared Error (MSE): It quantifies the accuracy of the model's predictions by calculating the average squared difference between predicted and actual values.

![R2](https://github.com/Kerolos-Noshy/House-Price-Prediction/assets/101178275/d53dc8a8-ce05-4b7a-b2ce-709c81de1d95)
- The Support Vector Regression (SVR) model achieved the highest R-squared score, indicating the best fit to the data.
  
![MSE](https://github.com/Kerolos-Noshy/House-Price-Prediction/assets/101178275/9482ced5-3739-4405-8772-e464ddc73bea)
- The SVR model also has the lowest Mean Squared Error, implying better prediction accuracy.



  
