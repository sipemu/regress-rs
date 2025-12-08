//! Ridge regression tests.

mod common;

use approx::assert_relative_eq;
use faer::{Col, Mat};
use regress_rs::solvers::{FittedRegressor, Regressor, RidgeRegressor};

// ============================================================================
// Basic Ridge Regression Tests
// ============================================================================

#[test]
fn test_basic_ridge_regression() {
    // Simple linear regression with small regularization
    let x = Mat::from_fn(10, 2, |i, j| ((i + j) as f64) * 0.1);
    let mut y = Col::zeros(10);
    for i in 0..10 {
        y[i] = 1.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 1)];
    }

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.01)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // With small lambda, should be close to OLS solution
    assert!(fitted.r_squared() > 0.99);
}

#[test]
fn test_lambda_zero_equals_ols() {
    // When lambda = 0, Ridge should equal OLS
    let (x, y, _) = common::generate_linear_data(50, 3, 2.0, 0.1, 42);

    let ridge = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.0)
        .build();

    let ols = regress_rs::solvers::OlsRegressor::builder()
        .with_intercept(true)
        .build();

    let ridge_fit = ridge.fit(&x, &y).expect("ridge fit should succeed");
    let ols_fit = ols.fit(&x, &y).expect("ols fit should succeed");

    // Coefficients should be nearly identical
    for i in 0..3 {
        assert_relative_eq!(
            ridge_fit.coefficients()[i],
            ols_fit.coefficients()[i],
            epsilon = 1e-8
        );
    }

    assert_relative_eq!(
        ridge_fit.intercept().unwrap(),
        ols_fit.intercept().unwrap(),
        epsilon = 1e-8
    );
}

#[test]
fn test_increasing_lambda_shrinks_coefficients() {
    // As lambda increases, coefficients should shrink toward zero
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.5, 42);

    let lambdas = [0.0, 0.1, 1.0, 10.0, 100.0];
    let mut prev_coef_norm = f64::INFINITY;

    for lambda in lambdas {
        let model = RidgeRegressor::builder()
            .with_intercept(true)
            .lambda(lambda)
            .build();

        let fitted = model.fit(&x, &y).expect("fit should succeed");

        // Compute L2 norm of coefficients
        let coef_norm: f64 = fitted
            .coefficients()
            .iter()
            .map(|&c| c.powi(2))
            .sum::<f64>()
            .sqrt();

        // Coefficient norm should decrease (or stay same) as lambda increases
        assert!(
            coef_norm <= prev_coef_norm + 1e-10,
            "lambda={}: coef_norm={} should be <= prev={}",
            lambda,
            coef_norm,
            prev_coef_norm
        );

        prev_coef_norm = coef_norm;
    }
}

#[test]
fn test_intercept_not_penalized() {
    // Intercept should not shrink toward zero with increasing lambda
    let mut x = Mat::zeros(20, 1);
    let mut y = Col::zeros(20);

    // y = 100 + x with large intercept
    for i in 0..20 {
        x[(i, 0)] = i as f64 * 0.1;
        y[i] = 100.0 + x[(i, 0)];
    }

    let fitted_low = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    let fitted_high = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(10.0)
        .build()
        .fit(&x, &y)
        .expect("fit should succeed");

    // Intercept should remain relatively stable
    let intercept_diff = (fitted_low.intercept().unwrap() - fitted_high.intercept().unwrap()).abs();
    assert!(
        intercept_diff < 10.0,
        "Intercept should be relatively stable, diff={}",
        intercept_diff
    );
}

#[test]
fn test_ridge_without_intercept() {
    let x = Mat::from_fn(10, 1, |i, _| (i + 1) as f64);
    let y = Col::from_fn(10, |i| 3.0 * (i + 1) as f64);

    let model = RidgeRegressor::builder()
        .with_intercept(false)
        .lambda(0.01)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    assert!(fitted.intercept().is_none());
    assert!(fitted.r_squared() > 0.99);
}

// ============================================================================
// Prediction Tests
// ============================================================================

#[test]
fn test_ridge_predict() {
    let x_train = Mat::from_fn(20, 2, |i, j| ((i + j) as f64) * 0.1);
    let mut y_train = Col::zeros(20);
    for i in 0..20 {
        y_train[i] = 1.0 + 2.0 * x_train[(i, 0)] + 3.0 * x_train[(i, 1)];
    }

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.01)
        .build();

    let fitted = model.fit(&x_train, &y_train).expect("fit should succeed");

    // Predict on new data
    let x_new = Mat::from_fn(5, 2, |i, j| ((i + j + 20) as f64) * 0.1);
    let predictions = fitted.predict(&x_new);

    assert_eq!(predictions.nrows(), 5);

    // Predictions should be reasonable
    for p in predictions.iter() {
        assert!(p.is_finite());
    }
}

#[test]
fn test_ridge_score() {
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.1, 42);

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    let score = fitted.score(&x, &y);

    // Score should be close to r_squared on training data
    assert_relative_eq!(score, fitted.r_squared(), epsilon = 1e-10);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_ridge_handles_collinearity() {
    // Ridge should handle collinear features better than OLS
    let (x, y) = common::generate_collinear_data(20);

    // OLS will have aliased coefficients
    let ols = regress_rs::solvers::OlsRegressor::builder()
        .with_intercept(true)
        .build();
    let ols_fit = ols.fit(&x, &y).expect("ols fit should succeed");
    assert!(ols_fit.result().has_aliased());

    // Ridge with lambda > 0 should have all finite coefficients
    let ridge = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(0.1)
        .build();
    let ridge_fit = ridge.fit(&x, &y).expect("ridge fit should succeed");

    // All coefficients should be finite (no NaN)
    for coef in ridge_fit.coefficients().iter() {
        assert!(coef.is_finite(), "Ridge coefficient should be finite");
    }
}

#[test]
fn test_ridge_r_squared_bounds() {
    let (x, y, _) = common::generate_linear_data(100, 3, 1.0, 0.5, 42);

    let model = RidgeRegressor::builder()
        .with_intercept(true)
        .lambda(1.0)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // R² should be in [0, 1]
    assert!(fitted.r_squared() >= 0.0);
    assert!(fitted.r_squared() <= 1.0);
}

#[test]
fn test_ridge_dimension_mismatch() {
    let x = Mat::zeros(10, 2);
    let y = Col::zeros(5); // Wrong size

    let model = RidgeRegressor::builder().lambda(0.1).build();
    let result = model.fit(&x, &y);

    assert!(result.is_err());
}
