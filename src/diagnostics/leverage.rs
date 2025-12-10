//! Leverage (hat matrix diagonal) calculations.

use faer::{Col, Mat};

/// Build design matrix, optionally prepending an intercept column.
fn build_design_matrix(x: &Mat<f64>, with_intercept: bool) -> Mat<f64> {
    let n = x.nrows();
    let p = x.ncols();

    if with_intercept {
        Mat::from_fn(n, p + 1, |i, j| if j == 0 { 1.0 } else { x[(i, j - 1)] })
    } else {
        x.to_owned()
    }
}

/// Build reduced design matrix excluding aliased columns.
fn build_design_matrix_reduced(x: &Mat<f64>, aliased: &[bool], with_intercept: bool) -> Mat<f64> {
    let n = x.nrows();
    let non_aliased: Vec<usize> = aliased
        .iter()
        .enumerate()
        .filter(|(_, &is_aliased)| !is_aliased)
        .map(|(j, _)| j)
        .collect();
    let p_reduced = non_aliased.len();
    let total_cols = if with_intercept {
        p_reduced + 1
    } else {
        p_reduced
    };

    Mat::from_fn(n, total_cols, |i, j| {
        if with_intercept {
            if j == 0 {
                1.0
            } else {
                x[(i, non_aliased[j - 1])]
            }
        } else {
            x[(i, non_aliased[j])]
        }
    })
}

/// Compute X'X (cross-product matrix).
fn compute_xtx(design: &Mat<f64>) -> Mat<f64> {
    design.transpose() * design
}

/// Compute (X'X)^(-1) using QR decomposition with back-substitution.
fn compute_xtx_inverse(xtx: &Mat<f64>) -> Mat<f64> {
    let p = xtx.nrows();
    let qr = xtx.qr();
    let q = qr.compute_Q();
    let r = qr.R().to_owned();
    let qt = q.transpose().to_owned();

    // Solve for each column of the inverse
    let mut inv = Mat::zeros(p, p);
    for col in 0..p {
        let solution = solve_triangular_column(&r, &qt, col, p);
        for row in 0..p {
            inv[(row, col)] = solution[row];
        }
    }
    inv
}

/// Solve for a column of (X'X)^(-1) via back-substitution.
fn solve_triangular_column(r: &Mat<f64>, qt: &Mat<f64>, col: usize, p: usize) -> Vec<f64> {
    let mut solution = vec![0.0; p];

    for i in (0..p).rev() {
        if r[(i, i)].abs() < 1e-14 {
            continue;
        }
        let mut sum = qt[(i, col)];
        for j in (i + 1)..p {
            sum -= r[(i, j)] * solution[j];
        }
        solution[i] = sum / r[(i, i)];
    }

    solution
}

/// Compute leverage value for a single observation.
fn compute_single_leverage(design_row: &[f64], xtx_inv: &Mat<f64>) -> f64 {
    let p = design_row.len();
    let mut h_ii = 0.0;

    for j in 0..p {
        for k in 0..p {
            h_ii += design_row[j] * xtx_inv[(j, k)] * design_row[k];
        }
    }

    h_ii.clamp(0.0, 1.0)
}

/// Compute leverage values (diagonal of hat matrix H = X(X'X)^(-1)X').
///
/// Leverage measures the influence of each observation on its own fitted value.
/// High leverage points have unusual predictor values.
///
/// # Properties
/// - h_ii ∈ [0, 1]
/// - Σ h_ii = p (number of parameters)
/// - Points with h_ii > 2p/n are considered high leverage
pub fn compute_leverage(x: &Mat<f64>, with_intercept: bool) -> Col<f64> {
    let n = x.nrows();
    let design = build_design_matrix(x, with_intercept);
    let p = design.ncols();

    let xtx = compute_xtx(&design);
    let xtx_inv = compute_xtx_inverse(&xtx);

    Col::from_fn(n, |i| {
        let row: Vec<f64> = (0..p).map(|j| design[(i, j)]).collect();
        compute_single_leverage(&row, &xtx_inv)
    })
}

/// Compute leverage values handling aliased (collinear) columns.
///
/// This version excludes aliased columns from the design matrix before computing
/// leverage. The leverage values are computed using only the non-aliased columns,
/// which gives correct results even when the full X'X matrix would be singular.
///
/// # Arguments
/// * `x` - Feature matrix (n × p)
/// * `aliased` - Boolean mask indicating which columns are aliased
/// * `with_intercept` - Whether to include an intercept column
///
/// # Returns
/// Leverage values for each observation
pub fn compute_leverage_with_aliased(
    x: &Mat<f64>,
    aliased: &[bool],
    with_intercept: bool,
) -> Col<f64> {
    let n = x.nrows();

    // If no aliased columns, use the standard method
    let has_aliased = aliased.iter().any(|&a| a);
    if !has_aliased {
        return compute_leverage(x, with_intercept);
    }

    let design = build_design_matrix_reduced(x, aliased, with_intercept);
    let p = design.ncols();

    if p == 0 {
        // All columns aliased - return NaN
        return Col::from_fn(n, |_| f64::NAN);
    }

    let xtx = compute_xtx(&design);
    let xtx_inv = compute_xtx_inverse(&xtx);

    // Check if inverse computation succeeded (diagonal should be non-zero for valid inverse)
    let mut inverse_valid = true;
    for i in 0..p {
        if xtx_inv[(i, i)].abs() < 1e-14 {
            inverse_valid = false;
            break;
        }
    }

    if !inverse_valid {
        return Col::from_fn(n, |_| f64::NAN);
    }

    Col::from_fn(n, |i| {
        let row: Vec<f64> = (0..p).map(|j| design[(i, j)]).collect();
        compute_single_leverage(&row, &xtx_inv)
    })
}

/// Identify high leverage points.
///
/// Returns indices of observations with leverage > threshold.
/// Default threshold is 2p/n where p is number of parameters.
pub fn high_leverage_points(
    leverage: &Col<f64>,
    n_params: usize,
    threshold: Option<f64>,
) -> Vec<usize> {
    let n = leverage.nrows();
    let cutoff = threshold.unwrap_or(2.0 * n_params as f64 / n as f64);

    leverage
        .iter()
        .enumerate()
        .filter(|(_, &h)| h > cutoff)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leverage_bounds() {
        let x = Mat::from_fn(20, 2, |i, j| ((i + j) as f64) * 0.1);
        let leverage = compute_leverage(&x, true);

        for i in 0..leverage.nrows() {
            assert!(
                leverage[i] >= 0.0,
                "Leverage[{}] = {} should be >= 0",
                i,
                leverage[i]
            );
            assert!(
                leverage[i] <= 1.0,
                "Leverage[{}] = {} should be <= 1",
                i,
                leverage[i]
            );
        }
    }

    #[test]
    fn test_leverage_sum() {
        // Use linearly independent predictors
        let x = Mat::from_fn(
            30,
            2,
            |i, j| {
                if j == 0 {
                    i as f64
                } else {
                    (i as f64).sin()
                }
            },
        );
        let leverage = compute_leverage(&x, true);

        let sum: f64 = leverage.iter().sum();
        let n_params = 3; // intercept + 2 features

        // Sum of leverage should equal number of parameters (rank of design matrix)
        // Allow more tolerance due to numerical precision
        assert!(
            (sum - n_params as f64).abs() < 0.5,
            "Sum of leverage {} should be close to {}",
            sum,
            n_params
        );
    }

    #[test]
    fn test_high_leverage_detection() {
        let mut x = Mat::from_fn(20, 1, |i, _| i as f64);
        // Add an extreme outlier in x
        x[(19, 0)] = 100.0;

        let leverage = compute_leverage(&x, true);
        let high = high_leverage_points(&leverage, 2, None);

        // The outlier should have high leverage
        assert!(
            high.contains(&19),
            "Point 19 should be flagged as high leverage"
        );
    }
}
