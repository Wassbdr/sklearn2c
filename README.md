# sklearn2c

**Transpile trained scikit-learn models to dependency-free C code — deploy ML on any microcontroller.**

```bash
python transpile_simple_model.py model.joblib
# → generated_model.c  (pure C, no libraries, compiles anywhere)
```

---

## Why

scikit-learn doesn't run on microcontrollers. Existing solutions (ONNX, TFLite Micro) require a runtime and add significant flash/RAM overhead. sklearn2c takes a different approach: it **walks the trained model's internal parameters** and hard-codes them as a pure C function — no interpreter, no memory allocation, no dependencies beyond `stdio.h`.

The output compiles with `gcc`, `arm-none-eabi-gcc`, or any standard C compiler and drops directly into Arduino, STM32, ESP32 firmware.

## Supported Models

| Model | Output type | Generated function |
|---|---|---|
| `LinearRegression` | `float` | `float predict(float *features, int n)` |
| `LogisticRegression` | `int` | `int predict(float *features, int n)` |
| `DecisionTreeClassifier` | `int` | `int predict(float *features, int n)` |

## Quick Start

```bash
# 1. Install
pip install scikit-learn pandas joblib numpy

# 2. Train a model
python train_model.py             # LinearRegression on house prices
python train_decision_tree.py     # DecisionTree on student data

# 3. Transpile to C
python transpile_simple_model.py regression.joblib
# → generated_model.c

# 4. Compile and run
gcc generated_model.c -o model && ./model

# 5. Copy predict() into your Arduino/firmware sketch
```

## Generated Code Example

Decision tree with depth 3 produces clean, readable C:

```c
int predict(float *features, int n_features) {
    if (features[1] <= -0.2990f) {
        if (features[0] <= -0.0770f) {
            return 1;
        } else {
            return 0;
        }
    } else {
        return 2;
    }
}
```

Linear/logistic regression:

```c
float prediction(float *features, int n_features) {
    float result = 150432.123f;           // intercept
    result += features[0] * 85.34f;
    result += features[1] * -312.77f;
    return result;
}
```

> Note: regression models generate a function named `prediction()`, decision trees generate `predict()`.


## Deploy on Arduino

Copy the generated `predict()` into your sketch — no libraries needed:

```cpp
// paste generated predict() here

void loop() {
    float sensors[2] = { analogRead(A0) / 1023.0f, analogRead(A1) / 1023.0f };
    float pred = predict(sensors, 2);
    Serial.println(pred);
    delay(1000);
}
```

## How It Works

1. Load the `.joblib` model with Python
2. Introspect internal parameters (coefficients, tree nodes, thresholds)
3. Generate C source with parameters as compile-time constants
4. The `predict()` function is a series of branches / dot products — no loops over parameters

## Project Structure

```
sklearn2c/
├── transpile_simple_model.py    # Core transpiler
├── train_model.py               # Example: LinearRegression
├── train_logistic.py            # Example: LogisticRegression
├── train_decision_tree.py       # Example: DecisionTree
├── generated_model.c            # Example output
├── houses.csv / students.csv    # Sample datasets
└── README.md
```

## Roadmap

- [ ] `RandomForestClassifier` (ensemble of trees)
- [ ] `SVC` with RBF kernel
- [ ] `MLPClassifier` (dense layers)
- [ ] Automated flash/RAM size estimation
- [ ] Arduino `.ino` file output

## License

MIT
