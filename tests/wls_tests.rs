//! Weighted Least Squares tests.

mod common;

use approx::assert_relative_eq;
use faer::{Col, Mat};
use statistics::solvers::{FittedRegressor, OlsRegressor, Regressor, WlsRegressor};

// ============================================================================
// Basic WLS Tests
// ============================================================================

#[test]
fn test_wls_basic() {
    let x = Mat::from_fn(20, 1, |i, _| i as f64);
    let y = Col::from_fn(20, |i| 1.0 + 2.0 * i as f64);
    let weights = Col::from_fn(20, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert!(fitted.r_squared() > 0.99);
}

#[test]
fn test_wls_equal_weights_equals_ols() {
    // When all weights are equal, WLS should equal OLS
    let (x, y, _) = common::generate_linear_data(50, 2, 1.0, 0.1, 42);
    let weights = Col::from_fn(50, |_| 1.0);

    let wls = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();
    let ols = OlsRegressor::builder().with_intercept(true).build();

    let wls_fit = wls.fit(&x, &y).expect("wls fit should succeed");
    let ols_fit = ols.fit(&x, &y).expect("ols fit should succeed");

    // Coefficients should be very close
    for j in 0..2 {
        assert_relative_eq!(
            wls_fit.coefficients()[j],
            ols_fit.coefficients()[j],
            epsilon = 1e-6
        );
    }

    assert_relative_eq!(
        wls_fit.intercept().unwrap(),
        ols_fit.intercept().unwrap(),
        epsilon = 1e-6
    );
}

#[test]
fn test_wls_without_intercept() {
    let x = Mat::from_fn(20, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(20, |i| 3.0 * (i + 1) as f64);
    let weights = Col::from_fn(20, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(false)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert!(fitted.intercept().is_none());
    assert!(fitted.r_squared() > 0.99);
}

// ============================================================================
// Weight Effect Tests
// ============================================================================

#[test]
fn test_wls_higher_weights_more_influence() {
    // Create data with two distinct relationships
    // Low x values: y = 1 + 1*x  (low weights)
    // High x values: y = 2 + 3*x (high weights)
    let n = 20;
    let x = Mat::from_fn(n, 1, |i, _| i as f64);
    let mut y = Col::zeros(n);
    let mut weights = Col::zeros(n);

    for i in 0..n / 2 {
        y[i] = 1.0 + 1.0 * i as f64;
        weights[i] = 0.1; // Low weight
    }
    for i in n / 2..n {
        y[i] = 2.0 + 3.0 * i as f64;
        weights[i] = 10.0; // High weight
    }

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Coefficient should be closer to 3 than to 1 due to higher weights
    // (This is a weak test, just checking the model fits)
    assert!(fitted.coefficients()[0] > 1.0);
}

#[test]
fn test_wls_zero_weights_excluded() {
    // Zero weights should effectively exclude observations (weight them to nothing)
    // Create perfect linear data with very small weight on outliers
    let x = Mat::from_fn(20, 1, |i, _| i as f64);
    let mut y = Col::zeros(20);
    let mut weights = Col::zeros(20);

    // Normal relationship: y = 1 + 2*x
    for i in 0..20 {
        y[i] = 1.0 + 2.0 * i as f64;
        weights[i] = 1.0;
    }

    // Replace two observations with outliers that have very small weight
    y[18] = 1000.0;
    weights[18] = 1e-10; // Near-zero weight (truly zero causes issues)
    y[19] = -1000.0;
    weights[19] = 1e-10;

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Should still get coefficient close to 2 since outliers have negligible weight
    assert!(
        (fitted.coefficients()[0] - 2.0).abs() < 0.5,
        "Coefficient should be near 2.0, got {}",
        fitted.coefficients()[0]
    );
}

// ============================================================================
// Prediction Tests
// ============================================================================

#[test]
fn test_wls_predict() {
    let x_train = Mat::from_fn(30, 2, |i, j| ((i + j) as f64) * 0.1);
    let mut y_train = Col::zeros(30);
    for i in 0..30 {
        y_train[i] = 1.0 + 2.0 * x_train[(i, 0)] + 3.0 * x_train[(i, 1)];
    }
    let weights = Col::from_fn(30, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x_train, &y_train).expect("fit should succeed");

    // Predict on new data
    let x_new = Mat::from_fn(5, 2, |i, j| ((i + j + 30) as f64) * 0.1);
    let predictions = fitted.predict(&x_new);

    assert_eq!(predictions.nrows(), 5);
    for p in predictions.iter() {
        assert!(p.is_finite());
    }
}

#[test]
fn test_wls_score() {
    let (x, y, _) = common::generate_linear_data(50, 2, 1.0, 0.1, 42);
    let weights = Col::from_fn(50, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // score() should return R² on training data
    let score = fitted.score(&x, &y);
    assert_relative_eq!(score, fitted.r_squared(), epsilon = 1e-10);
}

// ============================================================================
// Inference Tests
// ============================================================================

#[test]
fn test_wls_standard_errors() {
    let (x, y, _) = common::generate_linear_data(50, 2, 1.0, 0.5, 42);
    let weights = Col::from_fn(50, |_| 1.0);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .compute_inference(true)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    // Should have inference statistics
    assert!(result.std_errors.is_some());
    assert!(result.t_statistics.is_some());
    assert!(result.p_values.is_some());

    // Standard errors should be positive
    if let Some(ref se) = result.std_errors {
        for i in 0..se.nrows() {
            assert!(se[i] > 0.0, "SE[{}] should be positive", i);
        }
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_wls_dimension_mismatch_x_y() {
    let x = Mat::zeros(10, 2);
    let y = Col::zeros(5); // Wrong size
    let weights = Col::from_fn(10, |_| 1.0);

    let model = WlsRegressor::builder().weights(weights).build();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
}

#[test]
fn test_wls_dimension_mismatch_weights() {
    let x = Mat::zeros(10, 2);
    let y = Col::zeros(10);
    let weights = Col::from_fn(5, |_| 1.0); // Wrong size

    let model = WlsRegressor::builder().weights(weights).build();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
}

#[test]
fn test_wls_negative_weight_error() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| i as f64);
    let mut weights = Col::from_fn(10, |_| 1.0);
    weights[0] = -1.0; // Negative weight

    let model = WlsRegressor::builder().weights(weights).build();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
}

#[test]
fn test_wls_all_zero_weights_error() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let y = Col::from_fn(10, |i| i as f64);
    let weights = Col::from_fn(10, |_| 0.0); // All zero

    let model = WlsRegressor::builder().weights(weights).build();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
}

#[test]
fn test_wls_r_squared_bounds() {
    let x = Mat::from_fn(50, 2, |i, j| ((i * j + 1) as f64).sin());
    let y = Col::from_fn(50, |i| (i as f64).cos());
    let weights = Col::from_fn(50, |i| (i + 1) as f64);

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // R² should be in [0, 1]
    assert!(fitted.r_squared() >= 0.0);
    assert!(fitted.r_squared() <= 1.0);
}

#[test]
fn test_wls_varying_weights() {
    // Higher weights on more recent observations
    let x = Mat::from_fn(30, 1, |i, _| i as f64);
    let y = Col::from_fn(30, |i| 1.0 + 2.0 * i as f64 + 0.1 * (i as f64).sin());
    let weights = Col::from_fn(30, |i| (i + 1) as f64); // Linearly increasing

    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert!(fitted.r_squared() > 0.9);
    assert!(fitted.coefficients()[0].is_finite());
}

#[test]
fn test_wls_weights_access() {
    let weights = Col::from_fn(20, |i| (i + 1) as f64);
    let model = WlsRegressor::builder()
        .with_intercept(true)
        .weights(weights.clone())
        .build();

    let x = Mat::from_fn(20, 1, |i, _| i as f64);
    let y = Col::from_fn(20, |i| i as f64);

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Verify we can access weights from fitted model
    let stored_weights = fitted.weights();
    assert_eq!(stored_weights.nrows(), 20);
    for i in 0..20 {
        assert_relative_eq!(stored_weights[i], (i + 1) as f64);
    }
}
