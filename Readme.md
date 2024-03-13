# Email Spam Classifier

## Overview

This project aims to build a machine learning model to classify emails as either spam or not spam. The model is trained on a dataset containing labeled examples of emails.

## Dataset

The dataset used for training the model consists of email messages along with their labels indicating whether they are spam or not spam. The dataset was preprocessed to clean the data and extract relevant features.

### Data Preprocessing

- **Cleaning**: Initial data cleaning was performed to remove any irrelevant or redundant information from the dataset.
- **Text Preprocessing**: Text preprocessing steps included tokenization, removing stopwords, stemming, and extracting features like sentences, words, and characters from each message using NLTK.

### Handling Imbalanced Data

The dataset was found to be imbalanced, with approximately 87% of the samples labeled as not spam and the remaining 13% labeled as spam. Techniques such as resampling or adjusting class weights were explored to handle this imbalance.

### Vectorization

Text data was vectorized using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer, which converts text documents into numerical feature vectors.

## Models

Several machine learning models were trained and evaluated for their performance in classifying emails as spam or not spam. The following models were experimented with:

- Naive Bayes
- Multinomial Naive Bayes
- Gaussian Naive Bayes
- Random Forest
- Ensemble Learning (Voting Classifiers)

## Model Evaluation

Model performance was evaluated using metrics such as accuracy and precision. After experimentation, the Multinomial Naive Bayes model was selected as the best-performing model, achieving an accuracy of 98% and a precision score of 1.0.

## Conclusion

The Email Spam Classifier project demonstrates the effectiveness of machine learning techniques in classifying emails and highlights the importance of data preprocessing, feature engineering, and model selection in achieving accurate predictions.
