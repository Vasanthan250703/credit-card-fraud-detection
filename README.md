Credit Card Fraud Detection using Machine Learning

  This project focuses on detecting fraudulent transactions in credit card datasets using two supervised learning algorithms — Decision Tree and Support Vector Machine (SVM). The notebook includes data preprocessing, visualization, model training, evaluation, and export of the trained pipeline.

Objective:
  To build a classification model capable of identifying fraudulent credit card transactions from a highly imbalanced dataset using Decision Tree and Linear SVM classifiers.



Dataset: [creditcard.csv](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv)

Description: The dataset contains transactions made by credit cards in September 2013 by European cardholders. It consists of 31 features, including:

  Time, Amount, and 28 anonymized features (V1 to V28)

  Class is the target variable: 0 for legitimate, 1 for fraud

  The dataset is highly imbalanced with frauds representing only 0.172% of all transactions.


Steps Performed

  1.Data Loading and Exploration

  2.Loaded the dataset using pandas.

  3.Displayed dataset info, missing values, and class distribution.

  4.Class Imbalance Visualization

  5.Plotted a pie chart to show the imbalance between fraudulent and non-fraudulent transactions.

  6.Preprocessing

  7.Standardized features using StandardScaler

  8.Excluded the Time column from modeling

  9.Normalized features using L1 norm

  10.Train-Test Split

  11.Performed an 70-30 split of the dataset using train_test_split.

  12.Model Training
  

Decision Tree Classifier:

  Used DecisionTreeClassifier(max_depth=4, random_state=35)

  Handled imbalance with sample_weight using compute_sample_weight('balanced', y_train)
  

Linear SVM Classifier:

  Used LinearSVC(class_weight='balanced', loss='hinge', random_state=31, fit_intercept=False)
  

Model Evaluation:

  Used metrics such as Confusion Matrix, Classification Report, ROC-AUC Score

  Decision Tree had a lower ROC-AUC (~0.499)

  SVM performed well with ROC-AUC > 0.90
  

Model Visualization:

  Plotted the Decision Tree using graphviz and plot_tree

  Model Export

Saved the trained model using joblib as fraud_detection_pipeline.pkl



Requirements

Python ≥ 3.8

scikit-learn

matplotlib

pandas

numpy

joblib

graphviz

