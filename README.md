# sklearn2c

Projet pour convertir des modèles scikit-learn en code C pour microcontrôleurs.

## Pourquoi ce projet ?

J'avais besoin de déployer des modèles ML sur un Arduino pour un projet IoT, mais sklearn ne tourne pas sur microcontrôleur. J'ai donc créé ce transpileur qui prend un modèle entraîné et génère du code C pur sans dépendances.

## Ce qui marche

- Régression linéaire
- Régression logistique  
- Arbres de décision

## Installation

```bash
pip install scikit-learn pandas joblib numpy
```

## Utilisation

### 1. Entraîner un modèle

```bash
python train_model.py
```

### 2. Le convertir en C

```bash
python transpile_simple_model.py regression.joblib
```

### 3. Compiler et tester

```bash
./generated_model
```

## Exemples

### Régression linéaire

```python
from sklearn.linear_model import LinearRegression
import joblib

model = LinearRegression()
model.fit(X, y)
joblib.dump(model, 'model.joblib')
```

Puis :
```bash
python transpile_simple_model.py model.joblib
```

### Arbre de décision

```python
from sklearn.tree import DecisionTreeClassifier
import joblib

model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)
joblib.dump(model, 'tree.joblib')
```

## Code C généré

Le code généré est du C pur, juste avec `stdio.h`, pas besoin de bibliothèque ML.

### Régression

```c
float prediction(float *features, int n_features) {
    float result = INTERCEPT;
    for (int i = 0; i < n_features; i++) {
        result += COEFFICIENTS[i] * features[i];
    }
    return result;
}
```

### Arbre

```c
int predict(float *features, int n_features) {
    if (features[1] <= -0.299f) {
        if (features[0] <= -0.077f) {
            return 1;
        } else {
            return 0;
        }
    } else {
        // ... autres branches
    }
}
```

## Sur Arduino

Il suffit de copier la fonction `prediction()` générée dans votre code Arduino :

```cpp
void loop() {
    float sensors[] = {analogRead(A0), analogRead(A1)};
    float pred = prediction(sensors, 2);
    Serial.println(pred);
    delay(1000);
}
```

## Structure du projet

```
mlops/
├── transpile_simple_model.py    # Script principal
├── train_model.py               # Exemple régression linéaire
├── train_logistic.py            # Exemple régression logistique
├── train_decision_tree.py       # Exemple arbre de décision
├── generated_model.c            # Code C généré
├── houses.csv                   # Données d'exemple
├── students.csv                 # Données d'exemple
└── README.md
```

## Comment ça marche

1. Le script charge le modèle `.joblib`
2. Il extrait les paramètres (coefficients, structure de l'arbre, etc.)
3. Il génère du code C avec ces paramètres en dur
4. Il compile avec gcc
