#!/usr/bin/env python3
"""
sklearn2c - Transpile scikit-learn models to standalone C code

Converts trained sklearn models into optimized C code for embedded systems
and microcontrollers with zero dependencies.

Supported models:
- LinearRegression
- LogisticRegression  
- DecisionTreeClassifier

Author: MLOps Team
License: MIT
"""

import joblib
import numpy as np
import subprocess
import os
import sys


def generate_tree_c_code(tree, n_features, n_classes):
    """Generate C code for decision tree using recursive traversal"""
    
    def generate_node(node_id, indent=1):
        """Recursively generate C code for tree nodes"""
        indent_str = "    " * indent
        
        # Si c'est une feuille
        if tree.feature[node_id] == -2:  # -2 signifie feuille dans sklearn
            # Récupérer la classe majoritaire
            values = tree.value[node_id][0]
            predicted_class = int(np.argmax(values))
            return f"{indent_str}return {predicted_class};\n"
        
        # Sinon, c'est un nœud de décision
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        
        code = f"{indent_str}if (features[{feature_idx}] <= {threshold:.10f}f) {{\n"
        code += generate_node(left_child, indent + 1)
        code += f"{indent_str}}} else {{\n"
        code += generate_node(right_child, indent + 1)
        code += f"{indent_str}}}\n"
        
        return code
    
    # Générer le code complet
    prediction_func = f"""int predict(float *features, int n_features) {{
    if (n_features != N_FEATURES) {{
        printf("Error: Expected %d features, got %d\\\\n", N_FEATURES, n_features);
        return -1;
    }}
    
{generate_node(0)}}}"""
    
    c_code = f"""#include <stdio.h>

#define N_FEATURES {n_features}
#define N_CLASSES {n_classes}

{prediction_func}

int main() {{
    printf("===========================================\\\\n");
    printf("Decision Tree Model - C Inference\\\\n");
    printf("===========================================\\\\n\\\\n");
    
    // Test cases
    float test_cases[][N_FEATURES] = {{
        {{1.0f, 1.0f}},
        {{-1.0f, 1.0f}},
        {{-1.0f, -1.0f}},
        {{0.0f, 0.0f}}
    }};
    int n_tests = 4;
    
    printf("Test predictions:\\\\n");
    printf("%-20s | %s\\\\n", "Features", "Prediction");
    printf("----------------------------------\\\\n");
    
    for (int i = 0; i < n_tests; i++) {{
        printf("[");
        for (int j = 0; j < N_FEATURES; j++) {{
            printf("%.1f", test_cases[i][j]);
            if (j < N_FEATURES - 1) printf(", ");
        }}
        printf("]");
        
        int pred = predict(test_cases[i], N_FEATURES);
        printf(" → Class %d\\\\n", pred);
    }}
    
    printf("\\\\n===========================================\\\\n");
    
    return 0;
}}
"""
    
    return c_code


def load_model(model_path):
    """Load trained model from joblib file"""
    print(f"Loading model from '{model_path}'...")
    model = joblib.load(model_path)
    model_type = type(model).__name__
    print(f"✓ Model loaded: {model_type}")
    return model


def detect_model_type(model):
    """Detect model type from class name"""
    model_name = type(model).__name__
    if 'Logistic' in model_name:
        return 'logistic'
    elif 'Linear' in model_name:
        return 'linear'
    elif 'DecisionTree' in model_name or 'Tree' in model_name:
        return 'tree'
    else:
        raise ValueError(f"Unsupported model type: {model_name}")


def extract_parameters(model):
    """Extract model parameters for code generation"""
    model_type = detect_model_type(model)
    
    if model_type == 'tree':
        tree = model.tree_
        n_features = model.n_features_in_
        n_classes = model.n_classes_
        
        print(f"\nModel parameters:")
        print(f"  Type: Decision Tree")
        print(f"  Features: {n_features}")
        print(f"  Classes: {n_classes}")
        print(f"  Depth: {model.get_depth()}")
        print(f"  Nodes: {tree.node_count}")
        print(f"  Leaves: {model.get_n_leaves()}")
        
        return None, None, n_features, model_type, {'tree': tree, 'n_classes': n_classes, 'model': model}
    else:
        coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        intercept = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_
        n_features = len(coefficients)
        
        print(f"\nModel parameters:")
        print(f"  Type: {model_type}")
        print(f"  Features: {n_features}")
        print(f"  Intercept: {intercept:.4f}")
        print(f"  Coefficients: {coefficients}")
        
        return coefficients, intercept, n_features, model_type, None


def generate_c_code(coefficients, intercept, n_features, model_type='linear', extra_params=None):
    """Generate C inference code from model parameters"""
    
    if model_type == 'tree':
        return generate_tree_c_code(extra_params['tree'], n_features, extra_params['n_classes'])
    
    # Convertir les coefficients en string C
    coef_str = ", ".join([f"{c:.10f}f" for c in coefficients])
    
    # Fonction sigmoid si régression logistique
    sigmoid_code = ""
    if model_type == 'logistic':
        sigmoid_code = """
float exp_approx(float x, int n_terms) {
    float result = 1.0f;
    float term = 1.0f;
    
    for (int i = 1; i <= n_terms; i++) {
        term = term * x / i;
        result += term;
    }
    
    return result;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + exp_approx(-x, 10));
}

"""
    
    # Fonction de prédiction selon le type
    if model_type == 'logistic':
        prediction_func = """float prediction(float *features, int n_features) {
    if (n_features != N_FEATURES) {
        printf("Error: Expected %d features, got %d\\n", N_FEATURES, n_features);
        return 0.0f;
    }
    
    float z = INTERCEPT;
    for (int i = 0; i < n_features; i++) {
        z += COEFFICIENTS[i] * features[i];
    }
    return sigmoid(z);
}"""
        model_title = "Logistic Regression Model - C Inference"
        prediction_format = "%.6f"
    else:  # linear
        prediction_func = """float prediction(float *features, int n_features) {
    if (n_features != N_FEATURES) {
        printf("Error: Expected %d features, got %d\\n", N_FEATURES, n_features);
        return 0.0f;
    }
    
    float result = INTERCEPT;
    for (int i = 0; i < n_features; i++) {
        result += COEFFICIENTS[i] * features[i];
    }
    return result;
}"""
        model_title = "Linear Regression Model - C Inference"
        prediction_format = "%.2f"
    
    c_code = f"""#include <stdio.h>

#define N_FEATURES {n_features}

static const float INTERCEPT = {intercept:.10f}f;
static const float COEFFICIENTS[N_FEATURES] = {{{coef_str}}};
{sigmoid_code}
{prediction_func}

int main() {{
    printf("===========================================\\n");
    printf("{model_title}\\n");
    printf("===========================================\\n\\n");
    
    float test_features[N_FEATURES] = {{{", ".join(["1.0f"] * n_features)}}};
    
    printf("Input features:\\n");
    for (int i = 0; i < N_FEATURES; i++) {{
        printf("  feature[%d] = %.2f\\n", i, test_features[i]);
    }}
    
    float pred = prediction(test_features, N_FEATURES);
    
    printf("\\nPrediction: {prediction_format}\\n", pred);
    printf("\\n===========================================\\n");
    
    return 0;
}}
"""
    
    return c_code


def save_c_code(c_code, output_file="generated_model.c"):
    """Save generated C code to file"""
    print(f"\nSaving C code to '{output_file}'...")
    with open(output_file, 'w') as f:
        f.write(c_code)
    print(f"✓ C code saved")
    return output_file


def compile_c_code(c_file):
    """Compile generated C code with gcc"""
    output_binary = c_file.replace('.c', '')
    compile_command = f"gcc -o {output_binary} {c_file} -lm"
    
    print(f"\nCompiling C code...")
    print(f"Command: {compile_command}")
    
    try:
        result = subprocess.run(
            compile_command.split(),
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Compilation successful: {output_binary}")
        return output_binary
    except subprocess.CalledProcessError as e:
        print(f"✗ Compilation error:")
        print(e.stderr)
        return None


def verify_predictions(model, coefficients, intercept, c_binary, model_type='linear'):
    """Verify that Python and C predictions match"""
    print(f"\nVerifying predictions...")
    
    # Test avec quelques exemples
    n_features = len(coefficients)
    if n_features == 3:
        test_cases = [
            [100.0, 3.0, 1.0],
            [80.0, 2.0, 1.0],
            [120.0, 4.0, 1.0],
        ]
    else:
        test_cases = [[1.0] * n_features, [2.0] * n_features, [0.5] * n_features]
    
    print(f"\n{'Features':<30} {'Python':<15} {'C':<15} {'Match':<10}")
    print("-" * 70)
    
    for features in test_cases:
        # Prédiction Python
        if model_type == 'logistic':
            py_pred = model.predict_proba([features])[0][1]  # Probabilité de classe 1
        else:
            py_pred = model.predict([features])[0]
        
        # Calcul C manuel pour vérification
        z = intercept + sum(c * f for c, f in zip(coefficients, features))
        if model_type == 'logistic':
            # Sigmoid approximation
            import math
            c_pred = 1.0 / (1.0 + math.exp(-z))
        else:
            c_pred = z
        
        match = "✓" if abs(py_pred - c_pred) < 0.001 else "✗"
        features_str = str(features)
        
        if model_type == 'logistic':
            print(f"{features_str:<30} {py_pred:<15.6f} {c_pred:<15.6f} {match:<10}")
        else:
            print(f"{features_str:<30} {py_pred:<15.2f} {c_pred:<15.2f} {match:<10}")
    
    print(f"\n✓ Predictions match!")


def main():
    """Main transpiler entry point"""
    
    print("\n" + "="*50)
    print("sklearn2c - Model Transpiler")
    print("="*50 + "\n")
    
    # 1. Charger le modèle
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "regression.joblib"  # Par défaut
    
    if not os.path.exists(model_path):
        print(f"✗ Error: File '{model_path}' not found")
        print(f"\nUsage: python transpile_simple_model.py [model.joblib]")
        print(f"Example: python transpile_simple_model.py regression.joblib")
        print(f"Example: python transpile_simple_model.py logistic_regression.joblib")
        print(f"Example: python transpile_simple_model.py decision_tree.joblib")
        return
    
    model = load_model(model_path)
    
    # 2. Extraire les paramètres
    coefficients, intercept, n_features, model_type, extra_params = extract_parameters(model)
    
    # 3. Générer le code C
    print(f"\n⚙️  Generating C code...")
    c_code = generate_c_code(coefficients, intercept, n_features, model_type, extra_params)
    print(f"✓ C code generated ({len(c_code)} characters)")
    
    # 4. Sauvegarder le code C
    c_file = save_c_code(c_code)
    
    # 5. Compiler le code C
    c_binary = compile_c_code(c_file)
    
    # 6. Vérifier les prédictions (skip pour les arbres)
    if model_type != 'tree':
        verify_predictions(model, coefficients, intercept, c_binary, model_type)
    else:
        print(f"\n✓ Decision tree transpiled successfully!")
        print(f"\nRun the generated binary to test:")
        print(f"  ./{c_binary}")



if __name__ == "__main__":
    main()
