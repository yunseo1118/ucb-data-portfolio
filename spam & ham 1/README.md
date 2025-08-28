# Data 100 Project: Spam vs. Ham Email Classification

## 📧 Introduction
In this project, I apply what I’ve learned in class to build a **binary classifier** that distinguishes between:  
- **Spam**: junk, commercial, or bulk emails  
- **Ham**: regular, non-spam emails  

The project is split into two parts:  
- **Part 1**: Initial data exploration, feature engineering with text, and fitting a logistic regression model.  
- **Part 2**: Building a full spam/ham classifier, refining features, and testing alternative models.  

## Learning Goals
After completing this project, I will feel comfortable with:  
- Feature engineering techniques for text data.  
- Using the [`scikit-learn`](https://scikit-learn.org/stable/) library to preprocess data and fit models.  
- Validating model performance and addressing overfitting.  

## Objectives
- Clean and prepare raw email text data.  
- Engineer features suitable for classification (bag-of-words, frequencies, etc.).  
- Fit and evaluate a **logistic regression** model.  
- Extend the model to build a robust spam/ham classifier.  
- Reflect on model accuracy and trade-offs between false positives and false negatives.  

## Tools & Libraries
- Python 3  
- Jupyter Notebook  
- [`pandas`](https://pandas.pydata.org/) for data handling  
- [`numpy`](https://numpy.org/) for numerical operations  
- [`scikit-learn`](https://scikit-learn.org/stable/) for feature extraction, modeling, and validation  
