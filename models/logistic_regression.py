import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from sklearn.linear_model import LogisticRegression
from evaluation.classification_metrics import evaluate_classification

# Load the preprocessed data
data = joblib.load('data/preprocessed_data.pkl')

X_train_resampled = data['X_train_resampled']
y_train_resampled = data['y_train_resampled']
X_test = data['X_test_scaled']
y_test = data['y_test']

model=LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(X_train_resampled,y_train_resampled)

#predictions
y_train_pred = model.predict(X_train_resampled)
y_test_pred = model.predict(X_test)

# Evaluation for training data
print(f"Logistic Regression")
print(f"Training Metrics:")
metrics = evaluate_classification(y_train_resampled, y_train_pred)  # Use train predictions here
for key,value in metrics.items():
    print(f"{key}:{value}")

# Evaluation for testing data
print(f"\nTesting Metrics:")
metrics = evaluate_classification(y_test, y_test_pred)  # Use test predictions here
for key,value in metrics.items():
    print(f"{key}:{value}")