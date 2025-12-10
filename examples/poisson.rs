//! # Poisson Regression
//!
//! Poisson regression models count data where the response variable
//! represents the number of events occurring in a fixed interval.
//!
//! ## When to Use
//! - Count data (non-negative integers)
//! - Rate data (counts per unit of exposure)
//! - Events occur independently
//! - Mean approximately equals variance
//!
//! ## Key Features
//! - Log link (canonical): log(mu) = X*beta
//! - Identity link: mu = X*beta
//! - Square root link: sqrt(mu) = X*beta
//! - Offset term for exposure adjustment
//!
//! Run with: `cargo run --example poisson`

use anofox_regression::solvers::{FittedRegressor, PoissonRegressor, Regressor};
use faer::{Col, Mat};

fn main() {
    println!("=== Poisson Regression ===\n");

    basic_poisson();
    link_functions();
    with_offset();
    predict_counts();
}

/// Basic Poisson regression with log link
fn basic_poisson() {
    println!("--- Basic Poisson Regression (Log Link) ---\n");

    let n = 100;
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i as f64) * 0.05
        } else {
            ((i as f64) * 0.1).sin() * 0.5 + 0.5
        }
    });

    // Generate Poisson-like counts
    // True model: log(mu) = 0.5 + 0.3*x1 + 0.2*x2
    let y = Col::from_fn(n, |i| {
        let eta = 0.5 + 0.3 * x[(i, 0)] + 0.2 * x[(i, 1)];
        let mu = eta.exp();
        // Add some discrete noise around the mean
        let noise = ((i as f64 * 1.7).sin()) * 0.5;
        (mu + noise).round().max(0.0)
    });

    let model = PoissonRegressor::log().build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");
    let result = fitted.result();

    println!("True model: log(mu) = 0.5 + 0.3*x1 + 0.2*x2\n");
    println!(
        "Intercept: {:.4} (true: 0.5)",
        fitted.intercept().unwrap_or(0.0)
    );
    println!(
        "Coefficient x1: {:.4} (true: 0.3)",
        fitted.coefficients()[0]
    );
    println!(
        "Coefficient x2: {:.4} (true: 0.2)",
        fitted.coefficients()[1]
    );
    println!("\nDeviance: {:.4}", result.log_likelihood * -2.0);
    println!("AIC: {:.4}", result.aic);
    println!();
}

/// Compare different link functions
fn link_functions() {
    println!("--- Link Function Comparison ---\n");

    let n = 80;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.1);

    // Generate count data
    let y = Col::from_fn(n, |i| {
        let mu = 2.0 + 0.5 * x[(i, 0)];
        let noise = ((i as f64 * 0.9).sin()) * 0.5;
        (mu + noise).round().max(0.0)
    });

    // Log link (canonical)
    let log_model = PoissonRegressor::log().build();
    let log_fit = log_model.fit(&x, &y).expect("log fit");

    // Identity link
    let identity_model = PoissonRegressor::identity().build();
    let identity_fit = identity_model.fit(&x, &y).expect("identity fit");

    // Sqrt link
    let sqrt_model = PoissonRegressor::sqrt().build();
    let sqrt_fit = sqrt_model.fit(&x, &y).expect("sqrt fit");

    println!(
        "{:<12} {:>12} {:>12} {:>12}",
        "Link", "Intercept", "Slope", "AIC"
    );
    println!("{}", "-".repeat(52));

    println!(
        "{:<12} {:>12.4} {:>12.4} {:>12.4}",
        "Log",
        log_fit.intercept().unwrap_or(0.0),
        log_fit.coefficients()[0],
        log_fit.result().aic
    );
    println!(
        "{:<12} {:>12.4} {:>12.4} {:>12.4}",
        "Identity",
        identity_fit.intercept().unwrap_or(0.0),
        identity_fit.coefficients()[0],
        identity_fit.result().aic
    );
    println!(
        "{:<12} {:>12.4} {:>12.4} {:>12.4}",
        "Sqrt",
        sqrt_fit.intercept().unwrap_or(0.0),
        sqrt_fit.coefficients()[0],
        sqrt_fit.result().aic
    );

    println!("\nNote: Lower AIC indicates better model fit.");
    println!("      Coefficients are on different scales due to link functions.");
    println!();
}

/// Poisson regression with offset for rate models
fn with_offset() {
    println!("--- Poisson with Offset (Rate Model) ---\n");

    // Model counts with different exposure times
    let n = 60;

    // Exposure times (e.g., observation periods)
    let exposure = Col::from_fn(n, |i| 1.0 + (i % 5) as f64);

    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.1);

    // Counts depend on exposure and predictor
    // Rate model: log(count/exposure) = beta0 + beta1*x
    // Equivalent to: log(count) = log(exposure) + beta0 + beta1*x
    let y = Col::from_fn(n, |i| {
        let rate = (0.5 + 0.2 * x[(i, 0)]).exp();
        let expected_count = rate * exposure[i];
        let noise = ((i as f64 * 1.3).sin()) * 0.5;
        (expected_count + noise).round().max(0.0)
    });

    // Log of exposure as offset
    let offset = Col::from_fn(n, |i| exposure[i].ln());

    let model = PoissonRegressor::log().offset(offset).build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    println!("True rate model: log(rate) = 0.5 + 0.2*x");
    println!("With varying exposure times [1, 2, 3, 4, 5]\n");
    println!(
        "Intercept (log rate): {:.4} (true: 0.5)",
        fitted.intercept().unwrap_or(0.0)
    );
    println!(
        "Coefficient x1:       {:.4} (true: 0.2)",
        fitted.coefficients()[0]
    );

    // Show example rates
    println!("\nExample rate interpretation:");
    let base_rate = fitted.intercept().unwrap_or(0.0).exp();
    println!("  Base rate (x=0): {:.4} events per unit time", base_rate);
    let rate_at_5 = (fitted.intercept().unwrap_or(0.0) + fitted.coefficients()[0] * 5.0).exp();
    println!("  Rate at x=5:     {:.4} events per unit time", rate_at_5);
    println!();
}

/// Predicting counts with confidence intervals
fn predict_counts() {
    println!("--- Predicting Counts ---\n");

    use anofox_regression::core::IntervalType;

    let n = 80;
    let x = Mat::from_fn(n, 1, |i, _| (i as f64) * 0.1);

    let y = Col::from_fn(n, |i| {
        let mu = (1.0 + 0.25 * x[(i, 0)]).exp();
        let noise = ((i as f64 * 0.8).sin()) * (mu * 0.1);
        (mu + noise).round().max(0.0)
    });

    let model = PoissonRegressor::log().build();
    let fitted = model.fit(&x, &y).expect("fit should succeed");

    // Predict on new data
    let x_new = Mat::from_fn(5, 1, |i, _| (8 + i * 2) as f64);

    // Point predictions (expected counts)
    let predictions = fitted.predict(&x_new);

    // With confidence intervals
    let pred_with_ci = fitted.predict_with_interval(&x_new, Some(IntervalType::Confidence), 0.95);

    println!("Predictions for new x values:\n");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "x", "Expected", "Lower 95%", "Upper 95%"
    );
    println!("{}", "-".repeat(48));

    for i in 0..5 {
        println!(
            "{:>8.1} {:>12.2} {:>12.2} {:>12.2}",
            x_new[(i, 0)],
            predictions[i],
            pred_with_ci.lower[i],
            pred_with_ci.upper[i]
        );
    }

    println!("\nNote: Predictions are on the response scale (expected counts).");
    println!("      Confidence intervals are for the mean count, not individual observations.");
}
