# Data 100 Project: Spam vs. Ham Email Classification (Part 2)

## 📧 Introduction
This project builds on **Part 1**, where I explored email text data and developed a logistic regression model using basic feature engineering. In **Part 2**, I extend the work by training classifiers on a labeled dataset of emails and carefully evaluating their performance.  

The dataset includes:
- **Training set**: 8,348 labeled emails (ham = 0, spam = 1).  
- **Test set**: 1,000 unlabeled emails to be classified and submitted for evaluation.  

Each email includes:
- `id`: Unique identifier for the example.  
- `subject`: Subject line of the email.  
- `email`: Main body text.  
- `spam`: Label (1 = spam, 0 = ham).  

## Objectives
- Load, clean, and preprocess a text-based dataset of emails.  
- Engineer features from email subject lines and body text.  
- Train classifiers (e.g., logistic regression) to predict spam vs. ham.  
- Evaluate classifiers beyond raw accuracy using precision, recall, and false positive rate.  
- Reflect on real-world implications of email misclassification.  

## Classifier Evaluation
When evaluating classifiers, overall accuracy can be misleading. Two main types of errors are possible:  

- **False Positive (FP)**: A ham email flagged as spam (incorrectly filtered out).  
- **False Negative (FN)**: A spam email mislabeled as ham (slips into the inbox).  

These errors have different implications. For example, a false positive could cause someone to miss an important email, while a false negative allows spam into the inbox.  

To address this, we measure:  
- **Precision**: The proportion of predicted spam emails that are truly spam.  
- **Recall**: The proportion of actual spam emails that were correctly identified.  
- **False Positive Rate (FPR)**: The proportion of ham emails incorrectly flagged as spam.  

Balancing these metrics is crucial for building an effective spam filter.  

## Tools & Libraries
- Python 3  
- Jupyter Notebook  
- [`pandas`](https://pandas.pydata.org/) for data wrangling  
- [`numpy`](https://numpy.org/) for computations  
- [`scikit-learn`](https://scikit-learn.org/stable/) for feature extraction, modeling, and evaluation  

