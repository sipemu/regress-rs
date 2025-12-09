//! Variance Inflation Factor (VIF) for multicollinearity detection.

use crate::solvers::{FittedRegressor, OlsRegressor, Regressor};
use faer::{Col, Mat};

/// Compute Variance Inflation Factor for each predictor.
///
/// VIF measures how much the variance of a coefficient estimate is inflated
/// due to multicollinearity. For predictor j:
///
/// VIF_j = 1 / (1 - R²_j)
///
/// where R²_j is the R² from regressing x_j on all other predictors.
///
/// # Interpretation
/// - VIF = 1: No correlation with other predictors
/// - VIF > 5: Moderate multicollinearity (some sources say > 10)
/// - VIF > 10: High multicollinearity
///
/// # Returns
/// Vector of VIF values, one per predictor column.
pub fn variance_inflation_factor(x: &Mat<f64>) -> Col<f64> {
    let n = x.nrows();
    let p = x.ncols();

    if n < 3 || p < 2 {
        return Col::from_fn(p, |_| 1.0);
    }

    Col::from_fn(p, |j| compute_single_vif(x, j))
}

/// Compute VIF for a single predictor column.
fn compute_single_vif(x: &Mat<f64>, j: usize) -> f64 {
    let x_other = build_other_predictors_matrix(x, j);
    let y_j = extract_predictor_column(x, j);

    let model = OlsRegressor::builder().with_intercept(true).build();
    model
        .fit(&x_other, &y_j)
        .map(|fitted| r_squared_to_vif(fitted.r_squared()))
        .unwrap_or(1.0)
}

/// Build a matrix with all predictor columns except the specified one.
fn build_other_predictors_matrix(x: &Mat<f64>, exclude_col: usize) -> Mat<f64> {
    let n = x.nrows();
    let p = x.ncols();

    Mat::from_fn(n, p - 1, |i, out_col| {
        let src_col = if out_col < exclude_col {
            out_col
        } else {
            out_col + 1
        };
        x[(i, src_col)]
    })
}

/// Extract a single predictor column as the response vector.
fn extract_predictor_column(x: &Mat<f64>, j: usize) -> Col<f64> {
    Col::from_fn(x.nrows(), |i| x[(i, j)])
}

/// Extract a subset of columns from a matrix.
fn extract_columns(x: &Mat<f64>, start: usize, count: usize) -> Mat<f64> {
    let n = x.nrows();
    Mat::from_fn(n, count, |i, j| x[(i, start + j)])
}

/// Extract all columns except a specified range.
fn extract_other_columns(x: &Mat<f64>, exclude_start: usize, exclude_count: usize) -> Mat<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let other_size = p - exclude_count;

    Mat::from_fn(n, other_size, |i, out_col| {
        let src_col = if out_col < exclude_start {
            out_col
        } else {
            out_col + exclude_count
        };
        x[(i, src_col)]
    })
}

/// Compute max R² for a group of columns regressed on other predictors.
fn compute_group_r_squared(x_group: &Mat<f64>, x_other: &Mat<f64>) -> f64 {
    let n = x_group.nrows();
    let size = x_group.ncols();
    let model = OlsRegressor::builder().with_intercept(true).build();

    (0..size)
        .filter_map(|j| {
            let y_j = Col::from_fn(n, |i| x_group[(i, j)]);
            model.fit(x_other, &y_j).ok().map(|f| f.r_squared())
        })
        .fold(0.0_f64, f64::max)
}

/// Convert R² to VIF value.
fn r_squared_to_vif(r_squared: f64) -> f64 {
    if r_squared < 1.0 - 1e-14 {
        (1.0 / (1.0 - r_squared)).max(1.0)
    } else {
        f64::INFINITY
    }
}

/// Compute generalized VIF (GVIF) for categorical predictors.
///
/// For a predictor that spans multiple columns (dummy variables),
/// GVIF^(1/(2*df)) is comparable to regular VIF.
pub fn generalized_vif(x: &Mat<f64>, group_sizes: &[usize]) -> Vec<f64> {
    let p = x.ncols();

    // Verify group sizes sum to p
    if group_sizes.iter().sum::<usize>() != p {
        return vec![1.0; group_sizes.len()];
    }

    let mut gvif = Vec::with_capacity(group_sizes.len());
    let mut start_col = 0;

    for &size in group_sizes {
        let result = compute_gvif_for_group(x, start_col, size);
        gvif.push(result);
        start_col += size;
    }

    gvif
}

/// Compute GVIF for a single group of columns.
fn compute_gvif_for_group(x: &Mat<f64>, start_col: usize, size: usize) -> f64 {
    let p = x.ncols();

    // Empty groups have VIF = 1
    if size == 0 {
        return 1.0;
    }

    // Single-column predictors use regular VIF
    if size == 1 {
        return variance_inflation_factor(x)[start_col];
    }

    // No other columns means no collinearity
    let other_size = p - size;
    if other_size == 0 {
        return 1.0;
    }

    let x_group = extract_columns(x, start_col, size);
    let x_other = extract_other_columns(x, start_col, size);
    let r_squared = compute_group_r_squared(&x_group, &x_other);

    r_squared_to_vif(r_squared)
}

/// Identify predictors with high multicollinearity.
///
/// Returns indices of predictors with VIF > threshold.
/// Common threshold: 5 or 10.
pub fn high_vif_predictors(vif: &Col<f64>, threshold: f64) -> Vec<usize> {
    vif.iter()
        .enumerate()
        .filter(|(_, &v)| v > threshold)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vif_orthogonal_predictors() {
        // Orthogonal predictors should have VIF = 1
        let mut x: Mat<f64> = Mat::zeros(100, 2);
        for i in 0..100 {
            x[(i, 0)] = (i as f64 * 0.1).sin();
            x[(i, 1)] = (i as f64 * 0.1).cos();
        }

        let vif = variance_inflation_factor(&x);

        // VIF should be close to 1 for orthogonal predictors
        assert!(
            (vif[0] - 1.0).abs() < 0.5,
            "VIF[0] = {} should be near 1 for orthogonal predictor",
            vif[0]
        );
        assert!(
            (vif[1] - 1.0).abs() < 0.5,
            "VIF[1] = {} should be near 1 for orthogonal predictor",
            vif[1]
        );
    }

    #[test]
    fn test_vif_collinear_predictors() {
        // Highly collinear predictors should have high VIF
        let mut x: Mat<f64> = Mat::zeros(100, 2);
        for i in 0..100 {
            x[(i, 0)] = i as f64;
            x[(i, 1)] = i as f64 + 0.01 * (i as f64).sin(); // Almost identical
        }

        let vif = variance_inflation_factor(&x);

        // VIF should be very high for collinear predictors
        assert!(vif[0] > 10.0, "VIF[0] = {} should be > 10", vif[0]);
        assert!(vif[1] > 10.0, "VIF[1] = {} should be > 10", vif[1]);
    }

    #[test]
    fn test_vif_minimum_is_one() {
        let x = Mat::from_fn(50, 3, |i, j| ((i + j * 17) as f64).sin());

        let vif = variance_inflation_factor(&x);

        for j in 0..vif.nrows() {
            assert!(vif[j] >= 1.0, "VIF[{}] = {} should be >= 1", j, vif[j]);
        }
    }

    #[test]
    fn test_high_vif_detection() {
        // Create data with one collinear predictor
        let mut x: Mat<f64> = Mat::zeros(50, 3);
        for i in 0..50 {
            x[(i, 0)] = i as f64;
            x[(i, 1)] = (i as f64).sin(); // Independent
            x[(i, 2)] = i as f64 * 1.01 + 0.5; // Collinear with x0
        }

        let vif = variance_inflation_factor(&x);
        let high = high_vif_predictors(&vif, 5.0);

        // x0 and x2 should have high VIF
        assert!(
            high.contains(&0) || high.contains(&2),
            "At least one collinear predictor should be flagged"
        );
    }
}
