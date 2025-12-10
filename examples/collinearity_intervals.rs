//! Example demonstrating prediction intervals with collinear and constant features.
//!
//! This example shows that prediction intervals are correctly computed even when
//! some features are aliased (collinear or constant). The OLS solver automatically
//! detects aliased columns and computes intervals using only the non-aliased features.

use anofox_regression::core::IntervalType;
use anofox_regression::solvers::{FittedRegressor, OlsRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Test 1: Normal case ===\n");
    test_normal_case();

    println!("\n\n=== Test 2: Collinear features ===\n");
    test_collinear_case();

    println!("\n\n=== Test 3: Constant feature ===\n");
    test_constant_feature();
}

fn test_normal_case() {
    let x = Mat::from_fn(10, 1, |i, _| i as f64);
    let noise = [0.5, -0.3, 0.8, -0.2, 0.1, -0.6, 0.4, -0.1, 0.3, -0.5];
    let y = Col::from_fn(10, |i| 2.0 + 3.0 * (i as f64) + noise[i]);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!(
        "Coefficients: {:?}",
        fitted.coefficients().iter().collect::<Vec<_>>()
    );
    println!("Aliased: {:?}", fitted.result().aliased);

    let x_new = Mat::from_fn(2, 1, |i, _| (i + 10) as f64);
    let pred = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    for i in 0..2 {
        println!(
            "x={}: fit={:.3}, lower={:.3}, upper={:.3}",
            i + 10,
            pred.fit[i],
            pred.lower[i],
            pred.upper[i]
        );
    }
}

fn test_collinear_case() {
    // x1 and x2 are perfectly collinear (x2 = 2 * x1)
    let x = Mat::from_fn(10, 2, |i, j| if j == 0 { i as f64 } else { 2.0 * i as f64 });
    let noise = [0.5, -0.3, 0.8, -0.2, 0.1, -0.6, 0.4, -0.1, 0.3, -0.5];
    let y = Col::from_fn(10, |i| 2.0 + 3.0 * (i as f64) + noise[i]);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!(
        "Coefficients: {:?}",
        fitted.coefficients().iter().collect::<Vec<_>>()
    );
    println!("Aliased: {:?}", fitted.result().aliased);

    let x_new = Mat::from_fn(2, 2, |i, j| {
        if j == 0 {
            (i + 10) as f64
        } else {
            2.0 * (i + 10) as f64
        }
    });
    let pred = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    for i in 0..2 {
        println!(
            "x={}: fit={:.3}, lower={:.3}, upper={:.3}",
            i + 10,
            pred.fit[i],
            pred.lower[i],
            pred.upper[i]
        );
    }
}

fn test_constant_feature() {
    // x1 is normal, x2 is constant
    let x = Mat::from_fn(10, 2, |i, j| {
        if j == 0 {
            i as f64
        } else {
            5.0
        } // constant column
    });
    let noise = [0.5, -0.3, 0.8, -0.2, 0.1, -0.6, 0.4, -0.1, 0.3, -0.5];
    let y = Col::from_fn(10, |i| 2.0 + 3.0 * (i as f64) + noise[i]);

    let model = OlsRegressor::builder().with_intercept(true).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!(
        "Coefficients: {:?}",
        fitted.coefficients().iter().collect::<Vec<_>>()
    );
    println!("Aliased: {:?}", fitted.result().aliased);

    let x_new = Mat::from_fn(2, 2, |i, j| if j == 0 { (i + 10) as f64 } else { 5.0 });
    let pred = fitted.predict_with_interval(&x_new, Some(IntervalType::Prediction), 0.95);

    for i in 0..2 {
        println!(
            "x={}: fit={:.3}, lower={:.3}, upper={:.3}",
            i + 10,
            pred.fit[i],
            pred.lower[i],
            pred.upper[i]
        );
    }
}
