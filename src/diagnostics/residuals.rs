//! Standardized and studentized residuals.

use faer::Col;

/// Compute standardized residuals: e_i / s
///
/// Where s is the residual standard error (sqrt of MSE).
pub fn standardized_residuals(residuals: &Col<f64>, mse: f64) -> Col<f64> {
    if mse <= 0.0 || !mse.is_finite() {
        return Col::from_fn(residuals.nrows(), |i| {
            if residuals[i].abs() < 1e-14 {
                0.0
            } else {
                f64::NAN
            }
        });
    }

    let s = mse.sqrt();
    Col::from_fn(residuals.nrows(), |i| residuals[i] / s)
}

/// Compute internally studentized residuals: e_i / (s * sqrt(1 - h_ii))
///
/// These account for the varying variance of residuals due to leverage.
pub fn studentized_residuals(residuals: &Col<f64>, leverage: &Col<f64>, mse: f64) -> Col<f64> {
    let n = residuals.nrows();

    if mse <= 0.0 || !mse.is_finite() {
        return Col::from_fn(n, |_| f64::NAN);
    }

    let s = mse.sqrt();

    Col::from_fn(n, |i| {
        let h_ii = leverage[i];
        let denominator = s * (1.0 - h_ii).max(1e-14).sqrt();
        residuals[i] / denominator
    })
}

/// Compute externally studentized residuals (deleted residuals).
///
/// Uses leave-one-out MSE: e_i / (s_{(i)} * sqrt(1 - h_ii))
/// where s_{(i)} is the standard error computed without observation i.
///
/// These follow a t-distribution with n-p-1 degrees of freedom under null.
pub fn externally_studentized_residuals(
    residuals: &Col<f64>,
    leverage: &Col<f64>,
    mse: f64,
    n_params: usize,
) -> Col<f64> {
    let n = residuals.nrows();
    let df_resid = n - n_params;

    if df_resid <= 1 || mse <= 0.0 || !mse.is_finite() {
        return Col::from_fn(n, |_| f64::NAN);
    }

    // RSS = MSE * df_resid
    let rss = mse * df_resid as f64;

    Col::from_fn(n, |i| {
        let h_ii = leverage[i];
        let e_i = residuals[i];

        // Leave-one-out RSS: RSS_{(i)} = RSS - e_i² / (1 - h_ii)
        let one_minus_h = (1.0 - h_ii).max(1e-14);
        let rss_loo = rss - e_i * e_i / one_minus_h;

        // Leave-one-out MSE
        let df_loo = (df_resid - 1) as f64;
        if df_loo <= 0.0 {
            return f64::NAN;
        }
        let mse_loo = rss_loo / df_loo;

        if mse_loo <= 0.0 {
            return f64::NAN;
        }

        // Externally studentized residual
        e_i / (mse_loo.sqrt() * one_minus_h.sqrt())
    })
}

/// Identify outliers based on studentized residuals.
///
/// Returns indices of observations with |r_i| > threshold.
/// Common threshold is 2 or 3.
pub fn residual_outliers(studentized: &Col<f64>, threshold: f64) -> Vec<usize> {
    studentized
        .iter()
        .enumerate()
        .filter(|(_, &r)| r.abs() > threshold)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standardized_residuals() {
        let residuals = Col::from_fn(10, |i| (i as f64 - 4.5));
        let mse = 10.0; // arbitrary

        let std_resid = standardized_residuals(&residuals, mse);

        // Check scaling
        let s = mse.sqrt();
        for i in 0..10 {
            let expected = residuals[i] / s;
            assert!((std_resid[i] - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_studentized_residuals() {
        let residuals = Col::from_fn(10, |i| (i as f64 - 4.5));
        let leverage = Col::from_fn(10, |_| 0.2); // uniform leverage
        let mse = 10.0;

        let stud_resid = studentized_residuals(&residuals, &leverage, mse);

        // All residuals should be scaled by same factor
        let s = mse.sqrt();
        let factor = s * (1.0 - 0.2_f64).sqrt();

        for i in 0..10 {
            let expected = residuals[i] / factor;
            assert!((stud_resid[i] - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_outlier_detection() {
        let studentized = Col::from_fn(10, |i| {
            if i == 5 {
                4.0 // outlier
            } else {
                (i as f64 - 4.5) * 0.1 // small values
            }
        });

        let outliers = residual_outliers(&studentized, 2.0);
        assert_eq!(outliers, vec![5]);
    }
}
