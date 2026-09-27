# Titanic Data Analysis and Machine Learning

## Overview

This project analyzes the Titanic passenger dataset using Python and applies data analytics and machine learning techniques to understand passenger survival patterns and build predictive models.

The project includes three main components:

1. **Data Wrangling & Visualization**
2. **CRISP-DM Modeling with Logistic Regression**
3. **K-Nearest Neighbors (KNN) Classification**

The analysis covers data cleaning, missing-value handling, categorical encoding, outlier treatment, feature scaling, descriptive statistics, data visualization, model development, and model evaluation.

---

## Objectives

The main objectives of this project were to:

* Clean and prepare the Titanic dataset for analysis.
* Identify and handle missing values.
* Encode categorical variables.
* Detect and handle outliers using the IQR method.
* Scale selected numerical variables.
* Explore the dataset using descriptive statistics and visualizations.
* Predict passenger survival using Logistic Regression.
* Build a K-Nearest Neighbors classification model.
* Evaluate model performance using multiple classification metrics.
* Apply the CRISP-DM framework to the modeling process.
* Interpret model predictions and their potential practical applications.

---

## Dataset

The Titanic dataset contains information about passengers aboard the RMS Titanic, including demographic and passenger-related variables.

Key variables include:

| Variable      | Description                       |
| ------------- | --------------------------------- |
| `PassengerId` | Unique passenger identifier       |
| `Survived`    | Survival status                   |
| `Pclass`      | Passenger class                   |
| `Name`        | Passenger name                    |
| `Sex`         | Passenger sex                     |
| `Age`         | Passenger age                     |
| `SibSp`       | Number of siblings/spouses aboard |
| `Parch`       | Number of parents/children aboard |
| `Ticket`      | Ticket number                     |
| `Fare`        | Passenger fare                    |
| `Cabin`       | Cabin information                 |
| `Embarked`    | Port of embarkation               |

The target variable for the classification models is:

`Survived`

where:

* `0` = Did not survive
* `1` = Survived

---

# Part 1: Data Wrangling & Visualization

The first part of the project focuses on preparing the Titanic dataset and exploring its characteristics.

### Data Preparation

The following preprocessing techniques were applied:

* Loaded the Titanic dataset using Pandas.
* Inspected the dataset and its structure.
* Checked for missing values.
* Handled missing values using appropriate filling or removal methods.
* Encoded categorical variables such as `Sex` and `Embarked`.
* Identified and handled outliers in `Age` and `Fare`.
* Used the Interquartile Range (IQR) method to identify outliers.
* Scaled selected numerical variables using `StandardScaler`.
* Calculated descriptive statistics.

### Descriptive Statistics

The analysis examined:

* Mean
* Median
* Mode
* Standard deviation
* Minimum
* Maximum
* Quartiles

### Visualizations

The project includes the following visualizations:

1. **Age Histogram**
2. **Survival by Sex**
3. **Survival by Passenger Class**
4. **Survival Distribution**
5. **Correlation Heatmap**

These visualizations were used to identify patterns and relationships within the dataset.

### Clean Dataset

After preprocessing, the cleaned dataset was exported as:

```text
titanic_clean.csv
```

---

# Part 2: CRISP-DM and Logistic Regression

The second part of the project applies the **CRISP-DM framework** to a Titanic survival prediction problem.

## Business Understanding

The business question is:

> Can passenger information be used to predict whether a Titanic passenger survived?

A predictive model can demonstrate how historical passenger characteristics can be used for binary classification and can provide an example of how predictive analytics can support planning and decision-making.

## Data Understanding

The Titanic dataset was explored to understand:

* Dataset structure and size
* Passenger characteristics
* Target variable distribution
* Missing values
* Variable distributions
* Relationships between features and survival

Exploratory analysis helped identify patterns that could be useful for predicting survival.

## Data Preparation

Data preparation included:

* Removing or excluding unsuitable variables.
* Handling missing values.
* Encoding categorical variables.
* Preparing predictor and target variables.
* Scaling numerical variables where appropriate.
* Splitting the data into training and testing sets.

These steps prepared the dataset for machine learning.

## Modeling

The model used in this section is **Logistic Regression**.

Logistic Regression is appropriate for this task because the target variable is binary:

```text
0 = Did not survive
1 = Survived
```

The model was trained using the prepared Titanic data and used to generate predictions for unseen test data.

## Evaluation

The Logistic Regression model was evaluated using:

* Accuracy
* Precision
* Recall
* Confusion Matrix

### Accuracy

Accuracy represents the proportion of predictions that were classified correctly.

### Precision

Precision measures how many passengers predicted as survivors actually survived.

### Recall

Recall measures how many of the actual survivors were correctly identified by the model.

### Confusion Matrix

The confusion matrix shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

The relative number of false positives and false negatives provides additional information about the types of errors made by the model.

## Deployment

The trained model can be used to predict survival for new passenger records.

For example, a new passenger's characteristics such as age, sex, passenger class, and fare could be provided to the trained model to generate a predicted survival outcome.

---

# Part 3: K-Nearest Neighbors (KNN)

The third part of the project applies a **K-Nearest Neighbors (KNN)** classification model to the Titanic dataset.

The KNN model was developed using numerical and categorical passenger features.

## Model Development

The KNN analysis included:

* Preparing the Titanic dataset.
* Encoding categorical variables.
* Preparing numerical features.
* Scaling features.
* Testing different values of K.
* Selecting an appropriate K based on model performance.
* Training the KNN classifier.
* Generating predictions.

## Model Evaluation

The KNN model was evaluated using several techniques.

### Elbow Curve

The Elbow Curve was used to examine model performance across different K values.

The curve helps identify a suitable number of neighbors by showing how model error or performance changes as K increases.

### ROC Curve

The ROC Curve was used to evaluate the model's ability to distinguish between survivors and non-survivors.

The ROC AUC provides a measure of classification performance across different classification thresholds.

### Accuracy and Classification Report

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

These metrics provide different perspectives on classification performance.

### Decile-wise Lift Chart

The Lift Chart was used to examine how effectively the model identifies likely survivors across prediction-ranked groups.

Higher lift in the top deciles indicates that the model is concentrating positive predictions among passengers with higher predicted survival probabilities.

### Example Predictions

The model was also used to generate survival predictions for three new example passengers.

These predictions demonstrate how a trained classification model can be applied to previously unseen observations.

---

# Technologies and Tools

The project was completed using:

* **Python**
* **Google Colab**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

Key machine learning techniques include:

* Logistic Regression
* K-Nearest Neighbors
* Feature Scaling
* Categorical Encoding
* Classification Metrics

---

# Machine Learning Workflow

The overall workflow was:

```text
Raw Titanic Dataset
        ↓
Data Exploration
        ↓
Missing Value Handling
        ↓
Categorical Encoding
        ↓
Outlier Treatment
        ↓
Feature Scaling
        ↓
Exploratory Visualization
        ↓
Train/Test Split
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Predictions
```

---

# CRISP-DM Framework

The modeling process follows the six stages of CRISP-DM:

```text
Business Understanding
        ↓
Data Understanding
        ↓
Data Preparation
        ↓
Modeling
        ↓
Evaluation
        ↓
Deployment
```

This framework provides a structured approach for moving from a business problem to a data-driven predictive solution.

---

# Project Files

### Notebooks

The `notebooks` folder contains the Python/Google Colab notebooks used to complete the analysis.

### Data

The `data` folder contains the original and cleaned Titanic datasets.

### Reports

The `reports` folder contains the completed analysis report in PDF format.

### Images

The `images` folder contains selected charts and model evaluation visualizations generated during the analysis.

---

# Key Skills Demonstrated

This project demonstrates experience with:

* Data cleaning
* Data preprocessing
* Exploratory Data Analysis (EDA)
* Data visualization
* Statistical analysis
* Feature engineering
* Categorical encoding
* Outlier detection
* Feature scaling
* Classification
* Logistic Regression
* K-Nearest Neighbors
* Model evaluation
* Confusion matrices
* ROC/AUC analysis
* Lift analysis
* CRISP-DM
* Python
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
