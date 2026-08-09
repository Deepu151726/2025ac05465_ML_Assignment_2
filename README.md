# 🧬 Breast Cancer Classification ML 

## BITS M.Tech AIML — Machine Learning Assignment 2

This repository contains an end-to-end classification project using the **Breast Cancer Wisconsin (Diagnostic)** dataset.

The project covers the complete machine-learning workflow:

**Problem Definition → Data Understanding → EDA → Preprocessing → Model Training → Evaluation → Model Comparison → Error Analysis → Model Selection → Model Saving → Streamlit Deployment**

---

## 1. Problem Statement

The objective is to implement and compare multiple supervised classification algorithms on a public binary-classification dataset and evaluate their performance using standard classification metrics.

The **Breast Cancer Wisconsin (Diagnostic)** dataset is used to predict whether a tumour is **malignant or benign** from numerical diagnostic measurements.

Six classification models are implemented:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest
6. Gradient Boosting

Each model is evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 2. Dataset

### Breast Cancer Wisconsin (Diagnostic)

| Property | Value |
|---|---|
| Instances | 569 |
| Input features | 30 |
| Problem type | Binary classification |
| Target | Malignant / Benign |
| Feature type | Numerical |
| Target encoding | 0 = Malignant, 1 = Benign |

The project uses an **80/20 stratified train/test split** with `random_state = 42`.

---

## 3. Exploratory Data Analysis

The notebook includes:

- Dataset shape and structure
- Data types
- Missing-value check
- Duplicate check
- Descriptive statistics
- Target-class distribution
- Feature distributions
- Correlation heatmap
- Feature comparison by target class

These visualizations are used to understand the dataset before model development.

---

## 4. Machine Learning Models

### Logistic Regression
A linear classification model used as a strong baseline. Feature standardization is included.

### Decision Tree
A non-linear rule-based classifier capable of learning feature interactions.

### kNN
A distance-based classifier. Standardization is applied because the features have different numerical scales.

### Naive Bayes
A probabilistic classification baseline using Gaussian feature assumptions.

### Random Forest
An ensemble of decision trees designed to improve robustness and generalization.

### Gradient Boosting
An additive ensemble that sequentially improves weak learners.

---

## 5. Final Model Results

The following results were obtained from the final notebook execution.

| Rank | ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **Logistic Regression** | **0.9825** | **0.9954** | **0.9861** | **0.9861** | **0.9861** | **0.9623** |
| 2 | kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| 3 | Gradient Boosting | 0.9561 | 0.9911 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |
| 4 | Random Forest | 0.9561 | 0.9944 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| 5 | Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| 6 | Decision Tree | 0.9035 | 0.9373 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |

---

## 6. Final Model Selection

**Logistic Regression** was selected as the final model.

It achieved:

- **Accuracy:** 98.25%
- **AUC:** 0.9954
- **Precision:** 98.61%
- **Recall:** 98.61%
- **F1 Score:** 98.61%
- **MCC:** 0.9623

### Conclusion

Six classification models were evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and MCC. **Logistic Regression performed the best**, achieving the strongest overall and most consistent performance across the evaluation metrics. Therefore, Logistic Regression was selected as the final model for this project.

---

## 7. Model Artifacts

The trained models are saved using **Joblib**:

```text
model/
├── logistic_regression.joblib
├── decision_tree.joblib
├── knn.joblib
├── naive_bayes.joblib
├── random_forest.joblib
├── gradient_boosting.joblib
└── metrics.csv
```

The saved artifacts are loaded directly by `app.py`.

---

## 8. Streamlit Application

The Streamlit application is designed as an interactive ML evaluation dashboard.

### Features

#### 📊 Model Performance
Displays:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC

#### 🏆 Six-Model Benchmark
Runs all available saved models on the same test data and displays a ranked comparison.

#### 📂 CSV Upload
Users can upload a compatible test CSV through the sidebar.

If no file is uploaded, the application automatically uses the repository's `test_data.csv`.

#### 🔎 Confusion Matrix
Displays the confusion matrix together with:

- True Negatives
- False Positives
- False Negatives
- True Positives

#### 📈 ROC Curve
Displays the ROC curve and AUC of the selected model.

#### 📋 Classification Report
Displays class-level:

- Precision
- Recall
- F1 Score
- Support

#### 🔬 Prediction Explorer
Displays predictions, predicted class, benign probability and prediction confidence.

#### ⬇️ Prediction Export
Predictions can be downloaded as a CSV file.

---

## 9. Repository Structure

```text
ML_Assignment_2/
│
├── 2025ac05465_MachineLearning_Assignment.ipynb
├── app.py
├── requirements.txt
├── README.md
├── train_data.csv
├── test_data.csv
├── dataset_info.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    ├── gradient_boosting.joblib
    └── metrics.csv
```

---

## 10. Running the Application Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in a browser.

---

### Streamlit Application

`YOUR_STREAMLIT_APP_URL`

---

## 12. Requirements

The application uses:

- Python
- Streamlit
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Joblib

All dependencies are listed in `requirements.txt`.
