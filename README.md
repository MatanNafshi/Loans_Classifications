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
  ![image](https://github.com/user-attachments/assets/5029e2c4-a8da-4627-9ea8-7e8dcd3e1484)
  ![image](https://github.com/user-attachments/assets/b7e6c03d-c820-4dfd-964e-ede67a922ab4)


### **2. AdaBoost Classifier**
- **Methodology**:
  - Boosted weak learners using optimized Decision Tree base estimators.
  - Tuned parameters: learning rate, number of estimators, and base depth.
- **Performance**:
  - Achieved **93.12% accuracy**, improving overall recall for rejected loans.
  ![image](https://github.com/user-attachments/assets/ca2bbb7f-62a1-44ac-9cf9-75e0ddf1874b)
  ![image](https://github.com/user-attachments/assets/c5926e6a-c888-49e7-9e79-bfdba4ad8e1b)


### **3. Support Vector Machine (SVM)**
- **Methodology**:
  - Scaled features for compatibility with SVM.
  - Optimized `C`, `gamma`, and kernel type using randomized search.
- **Performance**:
  - Balanced accuracy with reduced overfitting.
  ![image](https://github.com/user-attachments/assets/5a4357fa-5049-4578-abaf-180badbbd716)
  ![image](https://github.com/user-attachments/assets/e69dca7b-5ccd-4147-94c1-e8ebfcc23eb3)

---

## **Workflow**

1. **Data Exploration**:
   - Analyzed relationships between applicant income, intent, and loan status.
   - Explored how demographic and credit history features affect approvals.

   ![image](https://github.com/user-attachments/assets/8d51dbf8-0e6d-43ce-80e1-148a31daccb2)

   ![image](https://github.com/user-attachments/assets/6e71ad45-3506-4f8d-8854-39c24a6e2014)


2. **Data Preprocessing**:
   - Handled missing data for critical features.
   - Encoded categorical variables into numerical format.
   - Log-transformed skewed features for stability.
     
  ![image](https://github.com/user-attachments/assets/25276e3f-5756-418b-9475-7812d7e96260)

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
