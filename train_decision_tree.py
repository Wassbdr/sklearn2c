"""
Train a Decision Tree model for transpilation to C
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

# Create a simple dataset for binary classification
np.random.seed(42)

# Generate synthetic data
n_samples = 200
X1 = np.random.randn(n_samples)
X2 = np.random.randn(n_samples)

# Target: y=1 if X1 <= 0 and X2 <= 0, else y=0
y = ((X1 <= 0) & (X2 <= 0)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'y': y
})

print("Dataset shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))
print("\nClass distribution:")
print(df['y'].value_counts())

# Train Decision Tree
X = df[['X1', 'X2']].values
y = df['y'].values

# Train with limited depth for simplicity
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

print("\n" + "="*50)
print("Model trained successfully!")
print("="*50)
print(f"Number of features: {model.n_features_in_}")
print(f"Number of classes: {model.n_classes_}")
print(f"Tree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")

# Save model
joblib.dump(model, 'decision_tree.joblib')
print("\nModel saved to: decision_tree.joblib")

# Test predictions
test_cases = [
    [1.0, 1.0],    # Expected: 0
    [-1.0, 1.0],   # Expected: 0
    [-1.0, -1.0],  # Expected: 1
    [0.0, 0.0],    # Expected: 1
]

print("\n" + "="*50)
print("Test predictions:")
print("="*50)
for features in test_cases:
    pred = model.predict([features])[0]
    proba = model.predict_proba([features])[0]
    print(f"Features: {features} → Prediction: {pred} (proba: {proba})")
