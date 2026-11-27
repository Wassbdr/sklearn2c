| Model | Samples compared | Mismatches | Max abs err | x86-64 .o (B) | Cortex-M4 flash (B) | Cortex-M4 RAM: static + stack (B) | Latency / sample (x86 VM) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DecisionTreeClassifier (max_depth=8) | 1176 | 0 | exact | 1,917 | 1,652 | 0 + 8 | 2.9 ns |
| ExtraTreeClassifier (max_depth=8) | 1078 | 0 | exact | 872 | 732 | 0 + 8 | 2.4 ns |
| RandomForestClassifier (50 trees, max_depth=8) | 5000 | 0 | exact | 82,443 | 82,320 | 0 + 72 | 1916.5 ns |
| ExtraTreesClassifier (50 trees, max_depth=8) | 5000 | 0 | exact | 78,555 | 78,452 | 0 + 72 | 1833.9 ns |
| LogisticRegression (3 classes) | 1000 | 0 | exact | 449 | 264 | 0 + 52 | 14.2 ns |
| RidgeClassifier (3 classes) | 1000 | 0 | exact | 449 | 264 | 0 + 52 | 14.2 ns |
| LinearSVC (3 classes) | 1000 | 0 | exact | 449 | 264 | 0 + 52 | 14.2 ns |
| DecisionTreeRegressor (max_depth=8) | 1464 | 0 | 8.8e-06 | 7,689 | 7,076 | 0 + 0 | 3.9 ns |
| ExtraTreeRegressor (max_depth=8) | 1344 | 0 | 8.4e-06 | 5,797 | 4,920 | 0 + 0 | 3.5 ns |
| RandomForestRegressor (50 trees, max_depth=8) | 5000 | 0 | 6.8e-05 | 174,216 | 174,080 | 0 + 20 | 2387.7 ns |
| ExtraTreesRegressor (50 trees, max_depth=8) | 5000 | 0 | 7.8e-05 | 141,068 | 140,964 | 0 + 20 | 2472.2 ns |
| LinearRegression | 1000 | 0 | 6.1e-05 | 256 | 92 | 0 + 16 | 3.1 ns |
| Ridge (alpha=1) | 1000 | 0 | 6.1e-05 | 256 | 92 | 0 + 16 | 3.4 ns |
| Lasso (alpha=0.1) | 1000 | 0 | 0.00012 | 256 | 92 | 0 + 16 | 3.5 ns |
| ElasticNet (alpha=0.1) | 1000 | 0 | 6.1e-05 | 256 | 92 | 0 + 16 | 3.1 ns |

- **data**: make_classification(2000, 8 features, 3 classes -> labels 10/20/30) and make_regression(2000, 8 features, noise=10), random_state=0, 75/25 split
- **compared**: held-out test rows (500) + 500 random rows + split-boundary probes (float32 on each threshold and one ulp above)
- **hardware**: x86 VM, not a microcontroller: Intel(R) Xeon(R) Processor
- **compiler**: gcc 11.4.0 (-O2 for latency, -Os for size)
- **arm_compiler**: arm-none-eabi-gcc 10.3.1 -Os -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16
- **versions**: python 3.10.12, scikit-learn 1.7.2, numpy 2.2.6, sklearn2c 0.2.0
- **regression_tolerance**: |C - sklearn| <= 1e-5 * max(1, |y|, sum|w*x|)
- **latency**: best of 7 passes over the 500 test rows, per sample, single thread

**Linear classifier near its decision boundary** (binary LogisticRegression; inputs placed at relative distance r from z = 0, then cast to float32). A flip is a label that differs from the exact sign of z, computed in rational arithmetic on the float32 inputs. sklearn's own `predict` is shown for comparison: this scikit-learn keeps `coef_` in float64 and evaluates the decision function in float64.

| r = \|z\| / (\|b\| + sum\|w*x\|) | samples | sklearn.predict flips | C float flips | C double flips |
|---|---:|---:|---:|---:|
| [0e+00, 1e-08) | 226 | 0 | 104 | 0 |
| [1e-08, 1e-07) | 242 | 0 | 19 | 0 |
| [1e-07, 1e-06) | 576 | 0 | 0 | 0 |
| [1e-06, 1e-05) | 602 | 0 | 0 | 0 |
| [1e-05, 1e-04) | 313 | 0 | 0 | 0 |
| >= 1e-04 | 141 | 0 | 0 | 0 |

Rounding bound gamma_(n+2): float 6.0e-07, double 1.1e-15. Flips beyond it: C float 0, C double 0 (guaranteed 0).
