# Cyber-Security
 Book-My-Show plans to introduce ads on their website, but they're concerned about user privacy. They want to assess if certain ad URLs might lead to phishing attacks, compromising security.

 # Code
 ```python
 import pandas as pd

# Load the dataset
dataset_path = r"C:\Users\prana\Desktop\AI-ML-Project-2-Cyber Security\dataset.csv"
data = pd.read_csv(dataset_path)

# Display basic information about the dataset
print(data.info())

# Display summary statistics
print(data.describe())

# Display the first few rows
print(data.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Create histograms for each feature
data.hist(bins=20, figsize=(15, 10))
plt.show()

# Create a heatmap of feature correlations
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Check for missing values in each column
missing_values = data.isnull().sum()
print(missing_values)


# Calculate correlations between features
corr_matrix = data.corr()

# Display the correlation matrix
print(corr_matrix)


# Choose a correlation threshold (e.g., 0.7)
correlation_threshold = 0.7

# Find highly correlated feature pairs
highly_correlated = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
            feature_i = corr_matrix.columns[i]
            feature_j = corr_matrix.columns[j]
            highly_correlated.add((feature_i, feature_j))

print("Highly Correlated Feature Pairs:")
for pair in highly_correlated:
    print(pair)


# Choose a correlation threshold (e.g., 0.7)
correlation_threshold = 0.7

# Find highly correlated feature pairs
highly_correlated = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
            feature_i = corr_matrix.columns[i]
            feature_j = corr_matrix.columns[j]
            highly_correlated.add((feature_i, feature_j))

# List of features to drop
features_to_drop = set()
for feature_i, feature_j in highly_correlated:
    # Drop one of the correlated features
    if feature_i not in features_to_drop:
        features_to_drop.add(feature_j)

# Remove correlated features from the dataset
data_dropped = data.drop(columns=features_to_drop)


# Replace 'Result' with your actual target column name
X = data.drop(columns=['Result'])
y = data['Result']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier  # You can choose any classifier from Scikit-Learn

# Initialize the classifier
classifier = RandomForestClassifier()  # Replace with your chosen classifier


from sklearn.ensemble import RandomForestClassifier  # You can choose any classifier from Scikit-Learn

# Initialize the classifier
classifier = RandomForestClassifier()  # Replace with your chosen classifier

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Predict probabilities for positive class (1)
y_probs = classifier.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

from sklearn.model_selection import cross_val_score

# Perform K-Fold cross-validation (e.g., 5-fold)
scores = cross_val_score(classifier, X, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())

# Example of hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define parameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)




import pandas as pd

# Load the dataset
dataset_path = r"C:\Users\prana\Desktop\AI-ML-Project-2-Cyber Security\dataset.csv"
data = pd.read_csv(dataset_path)

# Display basic information about the dataset
print(data.info())

# Display summary statistics
print(data.describe())

# Display the first few rows
print(data.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Create histograms for each feature
data.hist(bins=20, figsize=(15, 10))
plt.show()

# Create a heatmap of feature correlations
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Check for missing values in each column
missing_values = data.isnull().sum()
print(missing_values)


# Calculate correlations between features
corr_matrix = data.corr()

# Display the correlation matrix
print(corr_matrix)


# Choose a correlation threshold (e.g., 0.7)
correlation_threshold = 0.7

# Find highly correlated feature pairs
highly_correlated = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
            feature_i = corr_matrix.columns[i]
            feature_j = corr_matrix.columns[j]
            highly_correlated.add((feature_i, feature_j))

print("Highly Correlated Feature Pairs:")
for pair in highly_correlated:
    print(pair)


# Choose a correlation threshold (e.g., 0.7)
correlation_threshold = 0.7

# Find highly correlated feature pairs
highly_correlated = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
            feature_i = corr_matrix.columns[i]
            feature_j = corr_matrix.columns[j]
            highly_correlated.add((feature_i, feature_j))

# List of features to drop
features_to_drop = set()
for feature_i, feature_j in highly_correlated:
    # Drop one of the correlated features
    if feature_i not in features_to_drop:
        features_to_drop.add(feature_j)

# Remove correlated features from the dataset
data_dropped = data.drop(columns=features_to_drop)


# Replace 'Result' with your actual target column name
X = data.drop(columns=['Result'])
y = data['Result']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier  # You can choose any classifier from Scikit-Learn

# Initialize the classifier
classifier = RandomForestClassifier()  # Replace with your chosen classifier


from sklearn.ensemble import RandomForestClassifier  # You can choose any classifier from Scikit-Learn

# Initialize the classifier
classifier = RandomForestClassifier()  # Replace with your chosen classifier

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Predict probabilities for positive class (1)
y_probs = classifier.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

from sklearn.model_selection import cross_val_score

# Perform K-Fold cross-validation (e.g., 5-fold)
scores = cross_val_score(classifier, X, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())

# Example of hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define parameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)




