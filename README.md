# ML - Telco Customer Churn

## Project Overview

This project aims to predict customer churn in the telecommunications industry using various machine learning models. The process includes data preprocessing, feature engineering, model training, and evaluation to achieve high prediction accuracy.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries and Tools](#libraries-and-tools)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Modeling](#modeling)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Conclusion](#conclusion)

## Introduction

This project leverages machine learning to predict customer churn in the telecommunications industry. By analyzing the given dataset, we apply various models to determine which features most significantly impact customer retention.

## Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn**: Machine learning modeling and evaluation
- **XGBoost**: Gradient boosting algorithms
- **Imbalanced-learn**: Handling imbalanced datasets with techniques like SMOTE

## Data Preprocessing

1. **Data Loading**:
   - Loaded the dataset using `pd.read_csv()`.

2. **Data Cleaning**:
   - Dropped irrelevant column: `customerID`.
   - Ensured there were no missing values.

3. **Categorical Encoding**:
   - Applied `LabelEncoder` and `OneHotEncoder` for categorical features.

## Feature Engineering

1. **Class Balancing**:
   - Used SMOTE to handle class imbalance in the target variable.

2. **Correlation Analysis**:
   - Visualized correlations between features using a heatmap.

3. **Distribution Analysis**:
   - Created distribution plots and box plots to understand feature distributions and identify outliers.

## Modeling

1. **Logistic Regression**:
   - Implemented a logistic regression model with `max_iter=100000` and `C=5`.

2. **AdaBoost Classifier**:
   - Built an AdaBoost model with a base decision tree classifier and `n_estimators=2000`.

3. **XGBoost Classifier**:
   - Applied XGBoost for gradient boosting with `n_estimators=100` and `max_depth=1`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/telco-customer-churn.git
   ```

2. Navigate to the project directory:
   ```bash
   cd telco-customer-churn
   ```


## Usage

1. **Prepare Data**:
   - Ensure the dataset is available at the specified path.

2. **Train Models**:
   - Run the provided script to train models and evaluate performance.

3. **Predict Outcomes**:
   - Use the trained models to predict customer churn on new data.

## Conclusion

This project demonstrates the use of various machine learning models to predict customer churn in the telecommunications industry. The models were evaluated and tuned to achieve high accuracy, providing valuable insights into the factors influencing churn rates.

---

### Sample Code (for reference)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

# Reading Data
data = pd.read_csv(r"D:\Courses language programming\5_Machine Learning\Dataset For Machine Learning\Telco-Customer-Churn\Telco_Customer_Churn.csv")

# Data Preprocessing
data = data.drop(columns="customerID", axis=1)
for col in data.columns:
    print(col, data[col].unique())
    print("-" * 100)
data.isnull().sum()

# Data Cleaning
# Data is clean and has no null values

# Group by Internet Service Type
group_dsl = data[data["InternetService"] == "DSL"]
group_viper = data[data["InternetService"] == "Fiber optic"]
group_no = data[data["InternetService"] == "No"]

# Group by Payment Method
group_ele = data[data["PaymentMethod"] == "Electronic check"]
group_mail = data[data["PaymentMethod"] == "Mailed check"]
group_bank = data[data["PaymentMethod"] == "Bank transfer (automatic)"]
group_credit = data[data["PaymentMethod"] == "Credit card (automatic)"]

# Encoding categorical features
obj_col = data.select_dtypes(include=["object"])
for col in obj_col.columns:
    if col == "InternetService":
        x = data[col].unique()
        y = data[col].values
        new_col = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        for i, j in enumerate(x):
            data[x] = new_col[i]
        data.drop(columns=col, axis=1, inplace=True)
    else:
        data[col] = LabelEncoder().fit_transform(data[col])

# Data Visualization
plt.figure(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, square=True, fmt="0.3f")
plt.show()

for col in data.columns[:-3]:
    plt.figure(figsize=(5, 5))
    sns.displot(data[col])
    plt.show()

plt.figure(figsize=(30, 15))
plt.boxplot(data)
plt.show()

# Feature Engineering
# Oversampling using SMOTE
print(data[data["Churn"] == 0].shape, data[data["Churn"] == 1].shape)
x_input = data.drop(columns="Churn", axis=1)
y_output = data["Churn"]
new_x, new_y = SMOTE().fit_resample(x_input, y_output)
data = pd.DataFrame(pd.concat([new_x, new_y], axis=1))
print(data[data["Churn"] == 0].shape, data[data["Churn"] == 1].shape)

# Splitting Data & Building Model
x_input = data.drop(columns="Churn", axis=1)
y_output = data["Churn"]
x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, train_size=0.7, random_state=42)
y_output.value_counts()

# Model 1: Logistic Regression
model_reg = LogisticRegression(max_iter=100000, C=5)
model_reg.fit(x_train, y_train)
print(f"The Accuracy Training Data is {model_reg.score(x_train, y_train)}")
print(f"The Accuracy Testing Data is {model_reg.score(x_test, y_test)}")

# Model 2: AdaBoost Classifier
model_AD = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=200, min_samples_split=10, min_samples_leaf=10, max_features=5),
                              n_estimators=2000, learning_rate=0.0000001)
model_AD.fit(x_train, y_train)
print(f"The predict Score Train is ==> {model_AD.score(x_train, y_train)}")
print("%----------------------------------------------------------%")


print(f"The predict Score Test is ==> {model_AD.score(x_test, y_test)}")

# Model 3: XGBoost Classifier
model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=1)
model_xgb.fit(x_train, y_train)
print(f"The predict Score Train is ==> {model_xgb.score(x_train, y_train)}")
print("%----------------------------------------------------------%")
print(f"The predict Score Test is ==> {model_xgb.score(x_test, y_test)}")
```
