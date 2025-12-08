//! Tweedie Generalized Linear Model regression solver.
//!
//! Implements GLM with Tweedie family using Iteratively Reweighted Least Squares (IRLS).
//!
//! # Reference
//!
//! - R package `statmod`: <https://cran.r-project.org/web/packages/statmod/index.html>
//! - Dunn, P.K. and Smyth, G.K. (2018). "Generalized linear models with examples in R".
//!   Springer, New York, NY.

use crate::core::{
    IntervalType, PredictionResult, RegressionOptions, RegressionOptionsBuilder, RegressionResult,
    TweedieFamily,
};
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

/// Tweedie GLM regression estimator.
///
/// Fits a generalized linear model with Tweedie family using IRLS
/// (Iteratively Reweighted Least Squares).
///
/// # Model
///
/// The Tweedie GLM models:
/// - E[Y] = μ = g^(-1)(Xβ) where g is the link function
/// - Var[Y] = φ * V(μ) where V(μ) = μ^var_power
///
/// # Example
///
/// ```rust,ignore
/// use regress_rs::solvers::{TweedieRegressor, Regressor, FittedRegressor};
/// use faer::{Mat, Col};
///
/// let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
/// let y = Col::from_fn(100, |i| (i as f64 + 0.1).max(0.0));
///
/// // Gamma regression with log link
/// let fitted = TweedieRegressor::gamma()
///     .build()
///     .fit(&x, &y)?;
///
/// // Compound Poisson-Gamma for zero-inflated data
/// let fitted = TweedieRegressor::builder()
///     .var_power(1.5)
///     .link_power(0.0)  // log link
///     .build()
///     .fit(&x, &y)?;
/// ```
#[derive(Debug, Clone)]
pub struct TweedieRegressor {
    options: RegressionOptions,
    family: TweedieFamily,
}

impl TweedieRegressor {
    /// Create a new Tweedie regressor with the given options and family.
    pub fn new(options: RegressionOptions, family: TweedieFamily) -> Self {
        Self { options, family }
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> TweedieRegressorBuilder {
        TweedieRegressorBuilder::default()
    }

    /// Create a builder for Gaussian (Normal) regression.
    pub fn gaussian() -> TweedieRegressorBuilder {
        TweedieRegressorBuilder::default()
            .var_power(0.0)
            .link_power(1.0)
    }

    /// Create a builder for Poisson regression with log link.
    pub fn poisson() -> TweedieRegressorBuilder {
        TweedieRegressorBuilder::default()
            .var_power(1.0)
            .link_power(0.0)
    }

    /// Create a builder for Gamma regression with log link.
    pub fn gamma() -> TweedieRegressorBuilder {
        TweedieRegressorBuilder::default()
            .var_power(2.0)
            .link_power(0.0)
    }

    /// Create a builder for Inverse-Gaussian regression with log link.
    pub fn inverse_gaussian() -> TweedieRegressorBuilder {
        TweedieRegressorBuilder::default()
            .var_power(3.0)
            .link_power(0.0)
    }

    /// Fit the GLM using IRLS (Iteratively Reweighted Least Squares).
    ///
    /// IRLS Algorithm:
    /// 1. Initialize μ and compute η = g(μ)
    /// 2. Compute working weights W = 1 / (V(μ) * (dη/dμ)²)
    /// 3. Compute working response z = η + (y - μ) * (dη/dμ)
    /// 4. Solve weighted least squares: β = (X'WX)^(-1) X'Wz
    /// 5. Update η = Xβ, μ = g^(-1)(η)
    /// 6. Check convergence, repeat until done
    fn fit_irls(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedTweedie, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Determine number of parameters
        let n_params = if self.options.with_intercept {
            n_features + 1
        } else {
            n_features
        };

        // Build design matrix (with intercept if needed)
        let x_design = if self.options.with_intercept {
            let mut x_aug = Mat::zeros(n_samples, n_features + 1);
            for i in 0..n_samples {
                x_aug[(i, 0)] = 1.0;
                for j in 0..n_features {
                    x_aug[(i, j + 1)] = x[(i, j)];
                }
            }
            x_aug
        } else {
            x.clone()
        };

        // Initialize μ
        let y_vec: Vec<f64> = (0..n_samples).map(|i| y[i]).collect();
        let mut mu: Vec<f64> = self.family.initialize_mu(&y_vec);

        // Initialize η = g(μ)
        let mut eta: Vec<f64> = mu.iter().map(|&m| self.family.link(m)).collect();

        // Initialize β (will be updated in IRLS)
        let mut beta = Col::zeros(n_params);

        let max_iter = self.options.max_iterations;
        let tol = self.options.tolerance;
        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..max_iter {
            iterations = iter + 1;

            // Compute working weights and response
            let (weights, z) = self.compute_irls_quantities(&y_vec, &mu, &eta);

            // Solve weighted least squares: min_β Σ wᵢ (zᵢ - xᵢ'β)²
            let beta_new = self.solve_weighted_ls(&x_design, &z, &weights)?;

            // Check convergence: max|β_new - β_old| < tol
            let max_change: f64 = beta_new
                .iter()
                .zip(beta.iter())
                .map(|(&b_new, &b_old)| {
                    let diff: f64 = b_new - b_old;
                    diff.abs()
                })
                .fold(0.0_f64, f64::max);

            beta = beta_new;

            // Update η and μ
            for i in 0..n_samples {
                let mut eta_i = 0.0;
                for j in 0..n_params {
                    eta_i += x_design[(i, j)] * beta[j];
                }
                eta[i] = eta_i;
                mu[i] = self.family.link_inverse(eta_i);

                // Ensure μ > 0 for var_power > 0
                if self.family.var_power > 0.0 && mu[i] <= 0.0 {
                    mu[i] = 1e-6;
                }
            }

            if max_change < tol {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(RegressionError::ConvergenceFailed {
                iterations: max_iter,
            });
        }

        // Build result
        self.build_result(x, y, &x_design, &beta, &mu, &eta, n_params, iterations)
    }

    /// Compute IRLS working weights and working response.
    fn compute_irls_quantities(&self, y: &[f64], mu: &[f64], eta: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = y.len();
        let mut weights = vec![0.0; n];
        let mut z = vec![0.0; n];

        for i in 0..n {
            // Weight = 1 / (V(μ) * (dη/dμ)²)
            weights[i] = self.family.irls_weight(mu[i]);

            // Working response z = η + (y - μ) * (dη/dμ)
            z[i] = self.family.working_response(y[i], mu[i], eta[i]);
        }

        (weights, z)
    }

    /// Solve weighted least squares: min_β Σ wᵢ (zᵢ - xᵢ'β)²
    fn solve_weighted_ls(
        &self,
        x: &Mat<f64>,
        z: &[f64],
        weights: &[f64],
    ) -> Result<Col<f64>, RegressionError> {
        let n_samples = x.nrows();
        let n_params = x.ncols();

        // Transform: X_w = sqrt(W) * X, z_w = sqrt(W) * z
        let mut x_weighted = Mat::zeros(n_samples, n_params);
        let mut z_weighted = Col::zeros(n_samples);

        for i in 0..n_samples {
            let sqrt_w = weights[i].sqrt();
            for j in 0..n_params {
                x_weighted[(i, j)] = sqrt_w * x[(i, j)];
            }
            z_weighted[i] = sqrt_w * z[i];
        }

        // Solve via QR decomposition
        let qr = x_weighted.col_piv_qr();
        let q = qr.compute_q();
        let r = qr.compute_r();
        let perm = qr.col_permutation();

        // Compute Q'z
        let qtz = q.transpose() * z_weighted;

        // Back substitution
        let mut beta_perm = Col::zeros(n_params);
        for i in (0..n_params).rev() {
            let mut sum = qtz[i];
            for j in (i + 1)..n_params {
                sum -= r[(i, j)] * beta_perm[j];
            }
            if r[(i, i)].abs() > self.options.rank_tolerance {
                beta_perm[i] = sum / r[(i, i)];
            } else {
                beta_perm[i] = 0.0;
            }
        }

        // Unpermute
        let mut beta = Col::zeros(n_params);
        for i in 0..n_params {
            use faer::Index;
            beta[perm.inverse().arrays().0[i].to_signed().unsigned_abs()] = beta_perm[i];
        }

        Ok(beta)
    }

    /// Build the regression result from fitted values.
    #[allow(clippy::too_many_arguments)]
    fn build_result(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        x_design: &Mat<f64>,
        beta: &Col<f64>,
        mu: &[f64],
        _eta: &[f64],
        n_params: usize,
        iterations: usize,
    ) -> Result<FittedTweedie, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Extract intercept and coefficients
        let (intercept, coefficients) = if self.options.with_intercept {
            let int = Some(beta[0]);
            let coefs = Col::from_fn(n_features, |j| beta[j + 1]);
            (int, coefs)
        } else {
            (None, beta.clone())
        };

        // Compute fitted values and residuals
        let fitted_values = Col::from_fn(n_samples, |i| mu[i]);
        let residuals = Col::from_fn(n_samples, |i| y[i] - mu[i]);

        // Compute deviance
        let y_vec: Vec<f64> = (0..n_samples).map(|i| y[i]).collect();
        let deviance = self.family.deviance(&y_vec, mu);
        let null_deviance = self.family.null_deviance(&y_vec);

        // Estimate dispersion parameter φ
        // φ = D / (n - p) where D is deviance
        let df_resid = (n_samples.saturating_sub(n_params)) as f64;
        let dispersion = if df_resid > 0.0 {
            deviance / df_resid
        } else {
            1.0
        };

        // Compute R² (using deviance-based definition for GLM)
        let r_squared = if null_deviance > 0.0 {
            1.0 - deviance / null_deviance
        } else {
            f64::NAN
        };

        let df_total = (n_samples - 1) as f64;
        let adj_r_squared = if df_resid > 0.0 && df_total > 0.0 {
            1.0 - (1.0 - r_squared) * df_total / df_resid
        } else {
            f64::NAN
        };

        // MSE approximation for GLM
        let rss: f64 = residuals.iter().map(|&r| r.powi(2)).sum();
        let mse = if df_resid > 0.0 {
            rss / df_resid
        } else {
            f64::NAN
        };
        let rmse = mse.sqrt();

        // F-statistic (approximate)
        let df_model = (n_params - if intercept.is_some() { 1 } else { 0 }) as f64;
        let f_statistic = if df_model > 0.0 && df_resid > 0.0 && dispersion > 0.0 {
            ((null_deviance - deviance) / df_model) / dispersion
        } else {
            f64::NAN
        };

        let f_pvalue = if f_statistic.is_finite() && df_model > 0.0 && df_resid > 0.0 {
            FisherSnedecor::new(df_model, df_resid)
                .map(|d| 1.0 - d.cdf(f_statistic))
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };

        // Information criteria
        // For GLM: log-likelihood ≈ -D / (2φ) + constant
        let n = n_samples as f64;
        let k = n_params as f64;
        let log_likelihood = -deviance / (2.0 * dispersion)
            - n * (2.0 * std::f64::consts::PI * dispersion).ln() / 2.0;

        let aic = 2.0 * k - 2.0 * log_likelihood;
        let aicc = if (n - k - 1.0) > 0.0 {
            aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
        } else {
            f64::NAN
        };
        let bic = k * n.ln() - 2.0 * log_likelihood;

        // Determine rank
        let rank = n_params; // Full rank assumed for converged model

        let mut result = RegressionResult::empty(n_features, n_samples);
        result.coefficients = coefficients;
        result.intercept = intercept;
        result.residuals = residuals;
        result.fitted_values = fitted_values;
        result.rank = rank;
        result.n_parameters = n_params;
        result.n_observations = n_samples;
        result.aliased = vec![false; n_features];
        result.r_squared = r_squared;
        result.adj_r_squared = adj_r_squared;
        result.mse = mse;
        result.rmse = rmse;
        result.f_statistic = f_statistic;
        result.f_pvalue = f_pvalue;
        result.aic = aic;
        result.aicc = aicc;
        result.bic = bic;
        result.log_likelihood = log_likelihood;
        result.confidence_level = self.options.confidence_level;

        // GLM-specific: compute standard errors if requested
        if self.options.compute_inference {
            // Standard errors from Fisher information: SE = sqrt(diag(φ * (X'WX)^(-1)))
            if let Ok(se) = self.compute_standard_errors(x_design, mu, dispersion) {
                result.std_errors = Some(if self.options.with_intercept {
                    Col::from_fn(n_features, |j| se[j + 1])
                } else {
                    se.clone()
                });

                if self.options.with_intercept {
                    result.intercept_std_error = Some(se[0]);
                }
            }
        }

        Ok(FittedTweedie {
            result,
            options: self.options.clone(),
            family: self.family,
            deviance,
            null_deviance,
            dispersion,
            iterations,
        })
    }

    /// Compute standard errors from (X'WX)^(-1).
    fn compute_standard_errors(
        &self,
        x: &Mat<f64>,
        mu: &[f64],
        dispersion: f64,
    ) -> Result<Col<f64>, RegressionError> {
        let n_samples = x.nrows();
        let n_params = x.ncols();

        // Compute X'WX
        let mut xtwx: Mat<f64> = Mat::zeros(n_params, n_params);
        for i in 0..n_samples {
            let w = self.family.irls_weight(mu[i]);
            for j in 0..n_params {
                for k in 0..n_params {
                    xtwx[(j, k)] += w * x[(i, j)] * x[(i, k)];
                }
            }
        }

        // Invert via QR
        let qr = xtwx.qr();
        let q = qr.compute_q();
        let r: Mat<f64> = qr.compute_r();

        // Compute inverse column by column
        let mut xtwx_inv: Mat<f64> = Mat::zeros(n_params, n_params);
        for col in 0..n_params {
            let mut e = Col::zeros(n_params);
            e[col] = 1.0;
            let qte = q.transpose() * e;

            let mut sol = Col::zeros(n_params);
            for i in (0..n_params).rev() {
                let mut sum = qte[i];
                for j in (i + 1)..n_params {
                    sum -= r[(i, j)] * sol[j];
                }
                if r[(i, i)].abs() > 1e-14 {
                    sol[i] = sum / r[(i, i)];
                }
            }

            for i in 0..n_params {
                xtwx_inv[(i, col)] = sol[i];
            }
        }

        // SE = sqrt(φ * diag((X'WX)^(-1)))
        let se = Col::from_fn(n_params, |j| (dispersion * xtwx_inv[(j, j)]).sqrt());

        Ok(se)
    }
}

impl Regressor for TweedieRegressor {
    type Fitted = FittedTweedie;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Validate dimensions
        if x.nrows() != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: x.nrows(),
                y_len: y.nrows(),
            });
        }

        // Need at least 2 observations
        if n_samples < 2 {
            return Err(RegressionError::InsufficientObservations {
                needed: 2,
                got: n_samples,
            });
        }

        // Minimum observations
        let n_params = if self.options.with_intercept {
            n_features + 1
        } else {
            n_features
        };

        if n_samples < n_params {
            return Err(RegressionError::InsufficientObservations {
                needed: n_params,
                got: n_samples,
            });
        }

        // Validate family
        if !self.family.is_valid() {
            return Err(RegressionError::NumericalError(
                "var_power in (0, 1) is not allowed".to_string(),
            ));
        }

        // Check for valid y values based on family
        for i in 0..n_samples {
            if self.family.var_power > 0.0 && y[i] < 0.0 {
                return Err(RegressionError::NumericalError(format!(
                    "y values must be non-negative for var_power > 0, got y[{}] = {}",
                    i, y[i]
                )));
            }
        }

        self.fit_irls(x, y)
    }
}

/// Fitted Tweedie GLM model.
#[derive(Debug, Clone)]
pub struct FittedTweedie {
    result: RegressionResult,
    #[allow(dead_code)]
    options: RegressionOptions,
    family: TweedieFamily,
    /// Total deviance.
    pub deviance: f64,
    /// Null deviance (intercept-only model).
    pub null_deviance: f64,
    /// Estimated dispersion parameter.
    pub dispersion: f64,
    /// Number of IRLS iterations.
    pub iterations: usize,
}

impl FittedTweedie {
    /// Get the Tweedie family used for this model.
    pub fn family(&self) -> &TweedieFamily {
        &self.family
    }

    /// Compute predicted μ values on the response scale.
    pub fn predict_mu(&self, x: &Mat<f64>) -> Col<f64> {
        self.predict(x)
    }

    /// Compute predicted η values on the linear predictor scale.
    pub fn predict_eta(&self, x: &Mat<f64>) -> Col<f64> {
        let mu = self.predict(x);
        Col::from_fn(mu.nrows(), |i| self.family.link(mu[i]))
    }
}

impl FittedRegressor for FittedTweedie {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let intercept = self.result.intercept.unwrap_or(0.0);

        Col::from_fn(n_samples, |i| {
            let mut eta = intercept;
            for j in 0..n_features {
                eta += x[(i, j)] * self.result.coefficients[j];
            }
            self.family.link_inverse(eta)
        })
    }

    fn result(&self) -> &RegressionResult {
        &self.result
    }

    fn predict_with_interval(
        &self,
        x: &Mat<f64>,
        interval: Option<IntervalType>,
        _level: f64,
    ) -> PredictionResult {
        let predictions = self.predict(x);

        // GLM prediction intervals are complex due to non-constant variance
        // Return NaN for now (full implementation would need simulation or delta method)
        match interval {
            None => PredictionResult::point_only(predictions),
            Some(_) => {
                let n = x.nrows();
                let nan_vec = Col::from_fn(n, |_| f64::NAN);
                PredictionResult::with_intervals(
                    predictions,
                    nan_vec.clone(),
                    nan_vec.clone(),
                    nan_vec,
                )
            }
        }
    }
}

/// Builder for `TweedieRegressor`.
#[derive(Debug, Clone, Default)]
pub struct TweedieRegressorBuilder {
    options_builder: RegressionOptionsBuilder,
    var_power: f64,
    link_power: Option<f64>,
}

impl TweedieRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self {
            options_builder: RegressionOptionsBuilder::default(),
            var_power: 1.5, // Default: compound Poisson-Gamma
            link_power: None,
        }
    }

    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.options_builder = self.options_builder.with_intercept(include);
        self
    }

    /// Set whether to compute inference statistics.
    pub fn compute_inference(mut self, compute: bool) -> Self {
        self.options_builder = self.options_builder.compute_inference(compute);
        self
    }

    /// Set the confidence level for confidence intervals.
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.options_builder = self.options_builder.confidence_level(level);
        self
    }

    /// Set the maximum iterations for IRLS.
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.options_builder = self.options_builder.max_iterations(max_iter);
        self
    }

    /// Set the convergence tolerance.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.options_builder = self.options_builder.tolerance(tol);
        self
    }

    /// Set the variance power for the Tweedie family.
    ///
    /// Common values:
    /// - 0: Normal (Gaussian)
    /// - 1: Poisson
    /// - 1.5: Compound Poisson-Gamma
    /// - 2: Gamma
    /// - 3: Inverse-Gaussian
    pub fn var_power(mut self, p: f64) -> Self {
        self.var_power = p;
        self
    }

    /// Set the link power for the link function.
    ///
    /// - 0: Log link (most common)
    /// - 1: Identity link
    /// - -1: Inverse link
    /// - None: Use canonical link (1 - var_power)
    pub fn link_power(mut self, q: f64) -> Self {
        self.link_power = Some(q);
        self
    }

    /// Build the regressor.
    pub fn build(self) -> TweedieRegressor {
        let link_power = self.link_power.unwrap_or(1.0 - self.var_power);
        let family = TweedieFamily::new(self.var_power, link_power);

        TweedieRegressor {
            options: self.options_builder.build_unchecked(),
            family,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_regression() {
        // Simple linear regression
        let x = Mat::from_fn(20, 1, |i, _| i as f64);
        let y = Col::from_fn(20, |i| 2.0 + 3.0 * i as f64);

        let fitted = TweedieRegressor::gaussian()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .unwrap();

        // Should be close to true values
        assert!((fitted.result.intercept.unwrap() - 2.0).abs() < 0.5);
        assert!((fitted.result.coefficients[0] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_poisson_regression() {
        // Poisson-like data: y = exp(0.1 + 0.2*x)
        let x = Mat::from_fn(50, 1, |i, _| i as f64 / 10.0);
        let y = Col::from_fn(50, |i| {
            let eta = 0.1 + 0.2 * (i as f64 / 10.0);
            eta.exp().max(0.1)
        });

        let fitted = TweedieRegressor::poisson()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .unwrap();

        assert!(fitted.result.r_squared > 0.5);
        assert!(fitted.deviance < fitted.null_deviance);
    }

    #[test]
    fn test_gamma_regression() {
        // Gamma-like data
        let x = Mat::from_fn(30, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(30, |i| {
            let eta = 1.0 + 0.05 * (i + 1) as f64;
            eta.exp()
        });

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .unwrap();

        assert!(fitted.result.r_squared > 0.5);
    }

    #[test]
    fn test_compound_poisson_gamma() {
        // Data with zeros
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| {
            if i % 5 == 0 {
                0.0
            } else {
                (1.0 + 0.1 * i as f64).exp()
            }
        });

        let fitted = TweedieRegressor::builder()
            .var_power(1.5)
            .link_power(0.0)
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .unwrap();

        assert!(fitted.iterations > 0);
        assert!(fitted.dispersion > 0.0);
    }

    #[test]
    fn test_deviance_decrease() {
        // Generate data from known log-linear model: y = exp(1 + 0.1*x)
        // Use non-collinear predictor
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| (1.0 + 0.1 * i as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .unwrap();

        // Deviance should be less than null deviance for a good model
        assert!(
            fitted.deviance <= fitted.null_deviance * 1.01,
            "Deviance {} should be <= null deviance {}",
            fitted.deviance,
            fitted.null_deviance
        );
    }

    #[test]
    fn test_predict() {
        let x = Mat::from_fn(20, 1, |i, _| i as f64);
        let y = Col::from_fn(20, |i| (1.0 + 0.1 * i as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .build()
            .fit(&x, &y)
            .unwrap();

        let x_new = Mat::from_fn(5, 1, |i, _| (i + 20) as f64);
        let pred = fitted.predict(&x_new);

        // Predictions should be positive
        for i in 0..5 {
            assert!(pred[i] > 0.0);
        }
    }

    #[test]
    fn test_negative_y_error() {
        let x = Mat::from_fn(10, 1, |i, _| i as f64);
        let y = Col::from_fn(10, |i| if i < 5 { i as f64 } else { -1.0 });

        let result = TweedieRegressor::gamma().build().fit(&x, &y);

        assert!(matches!(result, Err(RegressionError::NumericalError(_))));
    }

    #[test]
    fn test_standard_errors() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| (1.0 + 0.1 * i as f64).exp());

        let fitted = TweedieRegressor::gamma()
            .with_intercept(true)
            .compute_inference(true)
            .build()
            .fit(&x, &y)
            .unwrap();

        assert!(fitted.result.std_errors.is_some());
        let se = fitted.result.std_errors.as_ref().unwrap();
        assert!(se[0] > 0.0);
    }
}
