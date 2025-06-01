# Classification-with-Logistic-Regression.

This repository demonstrates a complete pipeline for **binary classification using Logistic Regression** on the **Breast Cancer Wisconsin dataset**. It includes data preprocessing, model training, evaluation, threshold tuning, and sigmoid function explanation.

## 📌 Objective

To build and evaluate a binary classification model using Logistic Regression and understand the effect of threshold tuning and the role of the sigmoid function.

## 🛠️ Tools & Libraries

- Python
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)

##  Dataset

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** [Scikit-learn's built-in datasets](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- **Target:** `1 = Malignant`, `0 = Benign`

## 🚀 Steps Performed

1. **Load Dataset**  
   Using `load_breast_cancer()` from Scikit-learn.

2. **Train/Test Split & Standardization**  
   Split the data into training and test sets. Standardize features using `StandardScaler`.

3. **Model Training**  
   Fit a Logistic Regression model.

4. **Model Evaluation**  
   - Confusion Matrix  
   - Precision & Recall  
   - ROC-AUC Score  
   - ROC Curve Plot

5. **Threshold Tuning**  
   Changed the classification threshold from 0.5 to 0.6 to observe performance impact.

6. **Sigmoid Function Plot**  
   Visual explanation of how logistic regression outputs probabilities using the sigmoid function.

## Outputs

- Confusion Matrix
- Precision & Recall Scores
- ROC-AUC Score
- ROC Curve
- Sigmoid Function Graph
- Demonstration of custom threshold effect on classification


