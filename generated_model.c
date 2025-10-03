#include <stdio.h>

#define N_FEATURES 3

static const float INTERCEPT = -8152.9377101565f;
static const float COEFFICIENTS[N_FEATURES] = {717.2583697097f, 36824.1959742563f, 101571.8400215703f};

float prediction(float *features, int n_features) {
    if (n_features != N_FEATURES) {
        printf("Error: Expected %d features, got %d\n", N_FEATURES, n_features);
        return 0.0f;
    }
    
    float result = INTERCEPT;
    for (int i = 0; i < n_features; i++) {
        result += COEFFICIENTS[i] * features[i];
    }
    return result;
}

int main() {
    printf("===========================================\n");
    printf("Linear Regression Model - C Inference\n");
    printf("===========================================\n\n");
    
    float test_features[N_FEATURES] = {1.0f, 1.0f, 1.0f};
    
    printf("Input features:\n");
    for (int i = 0; i < N_FEATURES; i++) {
        printf("  feature[%d] = %.2f\n", i, test_features[i]);
    }
    
    float pred = prediction(test_features, N_FEATURES);
    
    printf("\nPrediction: %.2f\n", pred);
    printf("\n===========================================\n");
    
    return 0;
}
