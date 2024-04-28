#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
from scipy.sparse import hstack


# In[2]:


df = pd.read_csv('preprocessed_twitterdata.csv')
df['cleaned_message'].fillna('', inplace=True)


# In[3]:


df['word_count'] = df['cleaned_message'].apply(lambda x: len(x.split()))
y = df['sentiment']
# Map class labels to start from 0 instead of -1
y_adjusted = y.map({-1: 0, 0: 1, 1: 2, 2: 3})


# In[4]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['cleaned_message', 'word_count']], y_adjusted, test_size=0.2, random_state=42)


# # Feature Engineering

# In[5]:


# Convert text data into TF-IDF vectors with max_features=10000
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['cleaned_message'])
X_test_tfidf = tfidf_vectorizer.transform(X_test['cleaned_message'])

# Convert TF-IDF matrices to DataFrames
X_train_tfidf_df = pd.DataFrame.sparse.from_spmatrix(X_train_tfidf, columns=tfidf_vectorizer.get_feature_names_out())
X_test_tfidf_df = pd.DataFrame.sparse.from_spmatrix(X_test_tfidf, columns=tfidf_vectorizer.get_feature_names_out())

# Concatenate TF-IDF vectors with additional features
X_train_final = pd.concat([X_train_tfidf_df, X_train[['word_count']].reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test_tfidf_df, X_test[['word_count']].reset_index(drop=True)], axis=1)


# In[6]:


# Initialize and train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train_final, y_train)


# In[7]:


# Make predictions on the test set
y_pred = model.predict(X_test_final)


# In[8]:


# Calculate accuracy 
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the XGBoost model with additional features: {accuracy:.2f}")


# In[9]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy Score Testing: {accuracy:.2f}")

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Generate classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)


# In[10]:


import numpy as np
import matplotlib.pyplot as plt

# Compute True Positive Rate (TPR) and False Positive Rate (FPR)
TP = np.diag(conf_matrix)
FP = np.sum(conf_matrix, axis=0) - TP
FN = np.sum(conf_matrix, axis=1) - TP
TN = np.sum(conf_matrix) - (TP + FP + FN)

TPR = TP / (TP + FN)
FPR = FP / (FP + TN)

# Plot ROC curve
plt.figure()
plt.plot(FPR, TPR, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random classifier
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Calculate AUC-ROC
AUC_ROC = np.trapz(TPR, FPR)
print(f"AUC-ROC: {AUC_ROC:.2f}")


# In[11]:


# Sort TPR and FPR arrays based on ascending order of FPR
sorted_indices = np.argsort(FPR)
TPR_sorted = TPR[sorted_indices]
FPR_sorted = FPR[sorted_indices]

# Plot ROC curve
plt.figure()
plt.plot(FPR_sorted, TPR_sorted, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random classifier
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Calculate AUC-ROC
AUC_ROC = np.trapz(TPR_sorted, FPR_sorted)
print(f"AUC-ROC: {AUC_ROC:.2f}")


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
            xticklabels=['-1', '0', '1', '2'], 
            yticklabels=['-1', '0', '1', '2'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# # Random OverSampling

# In[13]:


pip install imbalanced-learn


# In[14]:


from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# Initialize RandomOverSampler
ros = RandomOverSampler(random_state=42)

# Perform random oversampling on the training data
X_train_ros, y_train_ros = ros.fit_resample(X_train_final, y_train)

# class distribution after oversampling
print("Class distribution after random oversampling:", Counter(y_train_ros))


# In[16]:


from sklearn.metrics import classification_report, confusion_matrix

# Train the model on the oversampled data
model.fit(X_train_ros, y_train_ros)

# Make predictions on the test set
y_pred = model.predict(X_test_final)

# Calculate accuracy 
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after random oversampling: {accuracy:.2f}")

# Generate the classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report after random oversampling:")
print(class_report)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix after random oversampling:")
print(conf_matrix)


# # Hyperparameter tuning

# In[17]:


# Convert text data into TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['cleaned_message'])
X_test_tfidf = tfidf_vectorizer.transform(X_test['cleaned_message'])

# Concatenate TF-IDF vectors with additional features
X_train_final = hstack([X_train_tfidf, X_train[['word_count']]])
X_test_final = hstack([X_test_tfidf, X_test[['word_count']]])


# In[18]:


# Define the hyperparameters grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}


# In[20]:


# Perform Randomized Search CV
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train_final, y_train)


# In[21]:


# Get the best parameters and best estimator
best_params = random_search.best_params_
best_estimator = random_search.best_estimator_

# Make predictions on the test set using the best estimator
y_pred = best_estimator.predict(X_test_final)


# In[22]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the XGBoost model with additional features after hyperparameter tuning: {accuracy:.2f}")


# In[23]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score Testing: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
            xticklabels=['-1', '0', '1', '2'], 
            yticklabels=['-1', '0', '1', '2'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# # Word Cloud

# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image # for opening, manipulating, and saving many different image file
import warnings
import numpy as np
from ast import literal_eval
import random
import nltk
warnings.filterwarnings('ignore')


# In[26]:


get_ipython().system('pip install wordcloud')


# In[27]:


import wordcloud


# In[28]:


get_ipython().system('pip install gensim')
import gensim


# In[37]:


df = pd.read_csv('preprocessed_twitterdata.csv')


# In[38]:


df['cleaned_message'].fillna('', inplace=True)
message_string = ' '.join(df['cleaned_message'])


# In[39]:


from wordcloud import WordCloud


# In[40]:


#twitter_mask = np.array(Image.open('twitter.png'))

# Generate the WordCloud with the Twitter mask
#wordcloud = WordCloud(mask=twitter_mask, background_color="white").generate(message_string)
wordcloud = WordCloud(background_color="white").generate(message_string)

# Display the WordCloud
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

