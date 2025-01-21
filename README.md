# **Loan Classification Project**

## **Overview**

This project involves predicting loan approval status using machine learning techniques. The dataset contains information about applicants, loan details, and credit history, providing a comprehensive foundation for exploring and modeling loan approval patterns. Advanced techniques like hyperparameter tuning, feature analysis, and ensemble learning were used to achieve optimal results.

---

## **Features**

### **Key Contributions**
- **End-to-End Pipeline**:
  - Data cleaning, preprocessing, and exploration.
  - Implementation of multiple machine learning models.
  - Detailed performance analysis and visualization.
- **Advanced Techniques**:
  - Randomized hyperparameter tuning using `RandomizedSearchCV`.
  - Feature importance analysis to explain model predictions.
  - Comparative analysis across different algorithms.
- **Interpretability**:
  - Visualizations highlighting data trends and model performance.
  - Insights into factors influencing loan approvals.

---

## **Dataset Overview**

### **Columns in the Dataset**
- **Applicant Demographics**:
  - `person_age`, `person_gender`, `person_education`, `person_income`.
- **Loan Details**:
  - `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `loan_intent`.
- **Credit History**:
  - `cb_person_cred_hist_length`, `credit_score`, `previous_loan_defaults_on_file`.
- **Target Variable**:
  - `loan_status` (Approved/Rejected).

### **Dataset Size**
- **Rows**: 45,000+
- **Columns**: 14 (After cleaning the data using Pandas).

---

## **Technologies Used**

- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`
  - Model Tuning: `RandomizedSearchCV`

---

## **Machine Learning Models**

### **1. Decision Tree Classifier**
- **Methodology**:
  - Tuned hyperparameters like tree depth, leaf size, and splitting criteria.
  - Feature importance analysis for interpretability.
- **Performance**:
  - Achieved **91.69% accuracy** with high recall for approved loans.

### **2. AdaBoost Classifier**
- **Methodology**:
  - Boosted weak learners using optimized Decision Tree base estimators.
  - Tuned parameters: learning rate, number of estimators, and base depth.
- **Performance**:
  - Achieved **93.12% accuracy**, improving overall recall for rejected loans.

### **3. Support Vector Machine (SVM)**
- **Methodology**:
  - Scaled features for compatibility with SVM.
  - Optimized `C`, `gamma`, and kernel type using randomized search.
- **Performance**:
  - Balanced accuracy with reduced overfitting.

---

## **Workflow**

1. **Data Exploration**:
   - Analyzed relationships between applicant income, intent, and loan status.
   - Explored how demographic and credit history features affect approvals.
2. **Data Preprocessing**:
   - Handled missing data for critical features.
   - Encoded categorical variables into numerical format.
   - Log-transformed skewed features for stability.
3. **Model Implementation**:
   - Built and tuned Decision Tree, AdaBoost, and SVM models.
   - Conducted 5-fold cross-validation for robust evaluation.
4. **Feature Importance Analysis**:
   - Identified top predictors like income, credit score, and loan intent.
5. **Evaluation**:
   - Compared accuracy, precision, recall, and F1-scores across models.

---

## **Results**

| Model         | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| Decision Tree | 91.69%   | 94%       | 96%    | 95%      |
| AdaBoost      | 93.12%   | 94%       | 97%    | 96%      |
| SVM           | TBD      | TBD       | TBD    | TBD      |

---

## **Visualizations**

### **Key Insights**
- **Feature Importance**:
  - Decision Tree and AdaBoost identified key factors like income and credit history as the most significant predictors of loan approval.
- **Loan Status Analysis**:
  - Visualized distribution of income and loan intent by approval status.
- **Model Performance**:
  - Plotted precision-recall and ROC curves to assess model reliability.

---
