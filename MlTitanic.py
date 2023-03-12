# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
titanic = pd.read_csv('titanic.csv')

# Preprocess the data
titanic = titanic.drop(['Name', 'Ticket', 'Cabin'], axis=1) # drop unnecessary columns
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True) # fill missing values in Age column with mean
titanic['Embarked'].fillna('S', inplace=True) # fill missing values in Embarked column with most common value
le = LabelEncoder()
titanic['Sex'] = le.fit_transform(titanic['Sex']) # encode Sex column as numeric values
titanic['Embarked'] = le.fit_transform(titanic['Embarked']) # encode Embarked column as numeric values

# Select features and target variable
X = titanic.drop(['Survived'], axis=1)
y = titanic['Survived']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = clf.predict(X_val)

# Evaluate the model
acc = accuracy_score(y_val, y_pred)
print('Accuracy:', acc)
