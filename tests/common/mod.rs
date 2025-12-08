//! Common test utilities and data generators.

use faer::{Col, Mat};

/// Generate simple linear data: y = x * beta + intercept + noise
pub fn generate_linear_data(
    n_samples: usize,
    n_features: usize,
    intercept: f64,
    noise_std: f64,
    seed: u64,
) -> (Mat<f64>, Col<f64>, Col<f64>) {
    // Simple deterministic "random" for reproducibility
    let mut rng_state = seed;
    let next_rand = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
    };

    let mut x = Mat::zeros(n_samples, n_features);
    let mut y = Col::zeros(n_samples);
    let mut true_coefficients = Col::zeros(n_features);

    // Generate true coefficients
    for j in 0..n_features {
        true_coefficients[j] = (j + 1) as f64;
    }

    // Generate X and y
    for i in 0..n_samples {
        let mut yi = intercept;
        for j in 0..n_features {
            x[(i, j)] = next_rand(&mut rng_state);
            yi += x[(i, j)] * true_coefficients[j];
        }
        yi += noise_std * next_rand(&mut rng_state);
        y[i] = yi;
    }

    (x, y, true_coefficients)
}

/// Generate data with collinear features.
pub fn generate_collinear_data(n_samples: usize) -> (Mat<f64>, Col<f64>) {
    let mut x = Mat::zeros(n_samples, 3);
    let mut y = Col::zeros(n_samples);

    for i in 0..n_samples {
        x[(i, 0)] = i as f64;
        x[(i, 1)] = 2.0 * i as f64; // Perfectly collinear with x0
        x[(i, 2)] = (i * i) as f64;
        y[i] = 1.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 2)];
    }

    (x, y)
}

/// Generate data with constant columns.
pub fn generate_constant_column_data(n_samples: usize) -> (Mat<f64>, Col<f64>) {
    let mut x = Mat::zeros(n_samples, 3);
    let mut y = Col::zeros(n_samples);

    for i in 0..n_samples {
        x[(i, 0)] = i as f64;
        x[(i, 1)] = 5.0; // Constant column
        x[(i, 2)] = (i * 2) as f64;
        y[i] = 1.0 + 2.0 * x[(i, 0)] + 3.0 * x[(i, 2)];
    }

    (x, y)
}

/// Approximate equality check for floating point values.
pub fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}
