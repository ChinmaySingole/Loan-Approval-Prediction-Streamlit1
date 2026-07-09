


# 💳 Loan Approval Prediction using Machine Learning and Streamlit

A complete Machine Learning project that predicts whether a loan application is likely to be **Approved** or **Rejected** based on an applicant's financial and personal information.

The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, multiple machine learning models, hyperparameter tuning, model comparison, and deployment using **Streamlit**.

---
  
## 🌐 Live Demo

👉 [Try the App Here](https://smart-loan-approval-prediction1.streamlit.app/)

---

## 📌 Project Overview

Financial institutions receive thousands of loan applications every day. Evaluating every application manually is time-consuming and may lead to inconsistent decisions.

This project automates the loan approval process using Machine Learning by learning patterns from historical loan application data.

The final model predicts whether a loan should be approved or rejected based on applicant information such as:

- Age
- Education
- Income
- Employment Experience
- Home Ownership
- Loan Amount
- Loan Purpose
- Interest Rate
- Credit History
- Credit Score
- Previous Loan History

---

# 🎯 Objectives

- Build a reliable loan approval prediction system
- Perform complete Exploratory Data Analysis (EDA)
- Clean and preprocess raw data
- Train multiple ML models
- Compare model performance
- Perform Hyperparameter Tuning
- Deploy the best model using Streamlit

---

# 📊 Dataset

- **Dataset Size:** 45,000 loan applications
- **Target Variable:** Loan Status

Target Encoding

| Value | Meaning |
|--------|---------|
| 0 | Loan Rejected |
| 1 | Loan Approved |

---

# 📁 Project Structure

```
Loan-Approval-Prediction/
│
├── app.py                     # Streamlit Application
├── Mini_Project.ipynb         # Complete ML Notebook
├── loan_data.csv              # Dataset
│
├── best_model.pkl             # Trained LightGBM Model
├── scaler.pkl                 # Standard Scaler
├── label_encoders.pkl         # Label Encoders
├── feature_columns.pkl        # Feature Names
├── best_model_name.pkl
│
├── requirements.txt
├── README.md
│
└── Images/
      dashboard.png
```

---

# ⚙️ Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost

### Deployment

- Streamlit

---

# 🛠 Data Preprocessing

The following preprocessing steps were performed:

- Missing Value Analysis
- Duplicate Check
- Feature Engineering
- Log Transformation of Income
- Label Encoding
- Train-Test Split
- Feature Scaling (StandardScaler)

---

# 📈 Exploratory Data Analysis

Several visualizations were created to understand the dataset.

Examples include:

- Loan Status Distribution
- Age Distribution
- Income Distribution
- Credit Score vs Loan Status
- Credit History vs Loan Status
- Correlation Heatmap
- Boxplots
- Histograms

Key Findings

- Dataset is imbalanced.
- Higher income generally improves approval chances.
- Credit score influences loan decisions.
- Loan percentage has a strong impact.
- Credit history contributes to approval decisions.

---

# 🤖 Machine Learning Models

The following models were trained and compared.

| Model |
|---------|
| Logistic Regression |
| Decision Tree |
| Random Forest |
| XGBoost |
| LightGBM |
| CatBoost |

---

# 🔧 Hyperparameter Tuning

To improve performance, GridSearchCV was used.

Parameters tuned include:

- Number of Trees
- Maximum Depth
- Learning Rate
- Number of Leaves

The tuned models outperformed the default versions.

---

# 📊 Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

Since the dataset is imbalanced, **F1 Score** and **ROC-AUC** were considered the primary evaluation metrics.

---

# 🏆 Best Model

After comparing all models, the **Tuned LightGBM Classifier** achieved the best overall performance.

### Final Performance

| Metric | Score |
|----------|--------|
| Accuracy | 92.22% |
| Precision | 78.94% |
| Recall | 88.65% |
| F1 Score | 83.35% |
| ROC-AUC | 97.79% |

---

# 🚀 Streamlit Web Application

The project includes an interactive Streamlit web application.

### Features

- Applicant Information Form
- Real-time Loan Prediction
- Approval Probability
- Rejection Probability
- Applicant Summary
- Decision Signals
- Risk Analysis
- Improvement Suggestions
- Encoded Feature View



# 🧠 Machine Learning Workflow

```
   Dataset
      │
      ▼
Data Cleaning
      │
      ▼
     EDA
      │
      ▼
Feature Engineering
      │
      ▼
    Encoding
      │
      ▼
Train-Test Split
      │
      ▼
Feature Scaling
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Model Deployment
```

---

# ▶️ How to Run

Clone the repository

```bash
git clone https://github.com/ChinmaySingole/Loan-Approval-Prediction-Streamlit.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 📌 Future Improvements

- Use Cross Validation
- Add User Authentication
- Deploy on Streamlit Cloud
- Connect with SQL Database
- Add PDF Report Generation
- Improve UI with Charts
- Add REST API

---

# 👨‍💻 Author

**Chinmay Singole**

Data Analyst | Machine Learning Enthusiast

### Skills

- Python
- SQL
- Power BI
- Tableau
- Machine Learning
- Data Analysis
- Streamlit


---

# ⭐ If you found this project useful, consider giving it a Star.