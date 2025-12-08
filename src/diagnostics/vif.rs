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
        // Can't compute VIF with fewer than 2 predictors
        return Col::from_fn(p, |_| 1.0);
    }

    let mut vif = Col::zeros(p);

    for j in 0..p {
        // Build design matrix with all predictors except j
        let mut x_other: Mat<f64> = Mat::zeros(n, p - 1);
        let mut col_idx = 0;
        for k in 0..p {
            if k != j {
                for i in 0..n {
                    x_other[(i, col_idx)] = x[(i, k)];
                }
                col_idx += 1;
            }
        }

        // Response is predictor j
        let y_j = Col::from_fn(n, |i| x[(i, j)]);

        // Regress x_j on other predictors
        let model = OlsRegressor::builder().with_intercept(true).build();

        match model.fit(&x_other, &y_j) {
            Ok(fitted) => {
                let r_squared = fitted.r_squared();
                // VIF = 1 / (1 - R²)
                let vif_j = if r_squared < 1.0 - 1e-14 {
                    1.0 / (1.0 - r_squared)
                } else {
                    f64::INFINITY
                };
                vif[j] = vif_j.max(1.0); // VIF is always >= 1
            }
            Err(_) => {
                // If regression fails, assume no collinearity
                vif[j] = 1.0;
            }
        }
    }

    vif
}

/// Compute generalized VIF (GVIF) for categorical predictors.
///
/// For a predictor that spans multiple columns (dummy variables),
/// GVIF^(1/(2*df)) is comparable to regular VIF.
pub fn generalized_vif(x: &Mat<f64>, group_sizes: &[usize]) -> Vec<f64> {
    let n = x.nrows();
    let p = x.ncols();

    // Verify group sizes sum to p
    let total: usize = group_sizes.iter().sum();
    if total != p {
        return vec![1.0; group_sizes.len()];
    }

    let mut gvif = Vec::with_capacity(group_sizes.len());
    let mut start_col = 0;

    for &size in group_sizes {
        if size == 0 {
            gvif.push(1.0);
            continue;
        }

        // For single-column predictors, use regular VIF
        if size == 1 {
            let vif_all = variance_inflation_factor(x);
            gvif.push(vif_all[start_col]);
            start_col += size;
            continue;
        }

        // Build design matrix for this group of columns
        let mut x_group: Mat<f64> = Mat::zeros(n, size);
        for i in 0..n {
            for j in 0..size {
                x_group[(i, j)] = x[(i, start_col + j)];
            }
        }

        // Build design matrix for all other columns
        let other_size = p - size;
        if other_size == 0 {
            gvif.push(1.0);
            start_col += size;
            continue;
        }

        let mut x_other: Mat<f64> = Mat::zeros(n, other_size);
        let mut col_idx = 0;
        for k in 0..p {
            if k < start_col || k >= start_col + size {
                for i in 0..n {
                    x_other[(i, col_idx)] = x[(i, k)];
                }
                col_idx += 1;
            }
        }

        // Compute R² for each column in group using other predictors
        let mut r_squared_group: f64 = 0.0;
        let model = OlsRegressor::builder().with_intercept(true).build();

        for j in 0..size {
            let y_j = Col::from_fn(n, |i| x_group[(i, j)]);

            if let Ok(fitted) = model.fit(&x_other, &y_j) {
                r_squared_group = r_squared_group.max(fitted.r_squared());
            }
        }

        // GVIF approximation
        let gvif_val = if r_squared_group < 1.0 - 1e-14 {
            1.0 / (1.0 - r_squared_group)
        } else {
            f64::INFINITY
        };

        gvif.push(gvif_val.max(1.0));
        start_col += size;
    }

    gvif
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
