//! # Recursive Least Squares (RLS)
//!
//! RLS is an online learning algorithm that updates regression estimates
//! incrementally as new data arrives, with optional forgetting factor
//! to weight recent observations more heavily.
//!
//! ## When to Use
//! - Streaming data / online learning
//! - Non-stationary relationships (time-varying coefficients)
//! - Adaptive filtering and signal processing
//! - Real-time prediction systems
//!
//! ## Key Features
//! - Processes data incrementally
//! - Forgetting factor (lambda) for adaptation speed
//! - lambda = 1: All observations weighted equally (converges to OLS)
//! - lambda < 1: Recent observations weighted more heavily
//!
//! Run with: `cargo run --example rls`

use anofox_regression::solvers::{FittedRegressor, Regressor, RlsRegressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Recursive Least Squares (RLS) ===\n");

    basic_rls();
    forgetting_factor_comparison();
    converges_to_ols();
    tracking_nonstationary();
}

/// Basic RLS with default settings
fn basic_rls() {
    println!("--- Basic RLS ---\n");

    let n = 100;
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i as f64) * 0.1
        } else {
            ((i as f64) * 0.2).sin()
        }
    });

    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.7).sin()) * 0.3;
        2.0 + 1.5 * x[(i, 0)] - 0.8 * x[(i, 1)] + noise
    });

    // RLS with forgetting factor = 0.99
    let model = RlsRegressor::builder()
        .with_intercept(true)
        .forgetting_factor(0.99)
        .build();

    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!("True model: y = 2 + 1.5*x1 - 0.8*x2 + noise");
    println!("Forgetting factor: 0.99\n");
    println!("Intercept: {:.4}", fitted.intercept().unwrap_or(0.0));
    println!("Coefficient x1: {:.4}", fitted.coefficients()[0]);
    println!("Coefficient x2: {:.4}", fitted.coefficients()[1]);
    println!("R-squared: {:.4}", fitted.r_squared());
    println!();
}

/// Compare different forgetting factors
fn forgetting_factor_comparison() {
    println!("--- Forgetting Factor Comparison ---\n");

    let n = 100;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.1);

    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.8).sin()) * 0.5;
        3.0 + 2.0 * x[(i, 0)] + noise
    });

    let factors = [1.0, 0.99, 0.95, 0.9, 0.8];

    println!(
        "{:<15} {:>12} {:>12} {:>12}",
        "Forget Factor", "Intercept", "Slope", "R-squared"
    );
    println!("{}", "-".repeat(55));

    for &factor in &factors {
        let model = RlsRegressor::builder()
            .with_intercept(true)
            .forgetting_factor(factor)
            .build();

        let fitted = model.fit(&x, &y).expect("fit should succeed");

        println!(
            "{:<15.2} {:>12.4} {:>12.4} {:>12.4}",
            factor,
            fitted.intercept().unwrap_or(0.0),
            fitted.coefficients()[0],
            fitted.r_squared()
        );
    }

    println!("\nNote: lambda = 1.0 gives equal weight to all observations (like OLS).");
    println!("      Lower lambda values weight recent observations more heavily.");
    println!();
}

/// Show that RLS with lambda=1 converges to OLS
fn converges_to_ols() {
    println!("--- RLS Converges to OLS ---\n");

    use anofox_regression::solvers::OlsRegressor;

    let n = 80;
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i as f64) * 0.15
        } else {
            ((i as f64) * 0.25).cos()
        }
    });

    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.6).sin()) * 0.4;
        1.5 + 2.5 * x[(i, 0)] - 1.2 * x[(i, 1)] + noise
    });

    // OLS
    let ols_model = OlsRegressor::builder().with_intercept(true).build();
    let ols_fitted = ols_model.fit(&x, &y).expect("OLS fit");

    // RLS with lambda = 1 (equivalent to OLS)
    let rls_model = RlsRegressor::builder()
        .with_intercept(true)
        .forgetting_factor(1.0)
        .build();
    let rls_fitted = rls_model.fit(&x, &y).expect("RLS fit");

    println!("True model: y = 1.5 + 2.5*x1 - 1.2*x2 + noise\n");

    println!(
        "{:<8} {:>12} {:>12} {:>12}",
        "Method", "Intercept", "x1", "x2"
    );
    println!("{}", "-".repeat(48));
    println!(
        "{:<8} {:>12.4} {:>12.4} {:>12.4}",
        "OLS",
        ols_fitted.intercept().unwrap_or(0.0),
        ols_fitted.coefficients()[0],
        ols_fitted.coefficients()[1]
    );
    println!(
        "{:<8} {:>12.4} {:>12.4} {:>12.4}",
        "RLS",
        rls_fitted.intercept().unwrap_or(0.0),
        rls_fitted.coefficients()[0],
        rls_fitted.coefficients()[1]
    );

    // Show differences
    let intercept_diff =
        (ols_fitted.intercept().unwrap_or(0.0) - rls_fitted.intercept().unwrap_or(0.0)).abs();
    let coef1_diff = (ols_fitted.coefficients()[0] - rls_fitted.coefficients()[0]).abs();
    let coef2_diff = (ols_fitted.coefficients()[1] - rls_fitted.coefficients()[1]).abs();

    println!("\nDifferences (should be very small):");
    println!("  |OLS - RLS| intercept: {:.2e}", intercept_diff);
    println!("  |OLS - RLS| x1:        {:.2e}", coef1_diff);
    println!("  |OLS - RLS| x2:        {:.2e}", coef2_diff);
    println!();
}

/// Tracking a non-stationary relationship
fn tracking_nonstationary() {
    println!("--- Tracking Non-Stationary Data ---\n");

    // Generate data where the relationship changes over time
    let n = 200;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.05);

    // Slope changes from 2 to 4 halfway through
    let y = Col::from_fn(n, |i| {
        let noise = ((i as f64 * 0.9).sin()) * 0.3;
        let slope = if i < n / 2 { 2.0 } else { 4.0 };
        1.0 + slope * x[(i, 0)] + noise
    });

    println!("Data with changing slope: 2.0 for first half, 4.0 for second half\n");

    // Fit with different forgetting factors using all data
    let factors = [1.0, 0.99, 0.95];

    // Also fit OLS on just the second half (ideal for second half prediction)
    let x_second = Mat::from_fn(n / 2, 1, |i, _| x[(i + n / 2, 0)]);
    let y_second = Col::from_fn(n / 2, |i| y[i + n / 2]);

    use anofox_regression::solvers::OlsRegressor;
    let ols_second = OlsRegressor::builder().with_intercept(true).build();
    let ols_second_fit = ols_second.fit(&x_second, &y_second).expect("fit");

    println!(
        "{:<20} {:>12} {:>15}",
        "Method", "Final Slope", "Tracks Change?"
    );
    println!("{}", "-".repeat(50));

    println!(
        "{:<20} {:>12.4} {:>15}",
        "OLS (2nd half only)",
        ols_second_fit.coefficients()[0],
        "Reference"
    );

    for &factor in &factors {
        let model = RlsRegressor::builder()
            .with_intercept(true)
            .forgetting_factor(factor)
            .build();

        let fitted = model.fit(&x, &y).expect("fit should succeed");

        // How close to the true second-half slope of 4.0?
        let tracks = if (fitted.coefficients()[0] - 4.0).abs() < 0.5 {
            "Good"
        } else if (fitted.coefficients()[0] - 4.0).abs() < 1.0 {
            "Moderate"
        } else {
            "Poor"
        };

        println!(
            "{:<20} {:>12.4} {:>15}",
            format!("RLS (lambda={})", factor),
            fitted.coefficients()[0],
            tracks
        );
    }

    println!("\nNote: Lower forgetting factors adapt faster to changes,");
    println!("      but may also be more sensitive to noise.");
    println!("      lambda = 1.0 averages all data equally (doesn't adapt).");
}
