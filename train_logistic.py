#!/usr/bin/env python3
"""
Entraîne une régression logistique et la sauvegarde
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Créer un dataset simple de classification binaire
# Exemple: admission à l'université basée sur scores
np.random.seed(42)

n_samples = 100
exam1 = np.random.uniform(30, 100, n_samples)
exam2 = np.random.uniform(30, 100, n_samples)
hours_studied = np.random.uniform(0, 10, n_samples)

# Règle simple: admis si moyenne > 65 ET heures > 3
admitted = ((exam1 + exam2) / 2 > 65) & (hours_studied > 3)
admitted = admitted.astype(int)

# Créer DataFrame
df = pd.DataFrame({
    'exam1': exam1,
    'exam2': exam2,
    'hours_studied': hours_studied,
    'admitted': admitted
})

# Sauvegarder le dataset
df.to_csv('students.csv', index=False)
print(f"✓ Dataset créé : {len(df)} étudiants")
print(f"  - Admis : {admitted.sum()}")
print(f"  - Refusés : {len(admitted) - admitted.sum()}")

# Entraîner le modèle
X = df[['exam1', 'exam2', 'hours_studied']]
y = df['admitted']

model = LogisticRegression()
model.fit(X, y)

print(f"\n✓ Modèle entraîné")
print(f"  - Intercept : {model.intercept_[0]:.4f}")
print(f"  - Coefficients : {model.coef_[0]}")
print(f"  - Score : {model.score(X, y):.4f}")

# Sauvegarder le modèle
joblib.dump(model, "logistic_regression.joblib")
print(f"\n✓ Modèle sauvegardé dans 'logistic_regression.joblib'")

# Test rapide
test_student = [[75, 80, 5]]  # Bon étudiant
prob = model.predict_proba(test_student)[0][1]
print(f"\n📊 Test: étudiant avec exam1=75, exam2=80, hours=5")
print(f"   Probabilité d'admission: {prob:.4f} ({prob*100:.1f}%)")
