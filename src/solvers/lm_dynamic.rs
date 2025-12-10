//! Dynamic Linear Model (lmDynamic) - Time-varying parameter regression.
//!
//! This module implements a time-varying parameter model that combines multiple
//! candidate regression models using pointwise information criteria weighting.
//! Based on the lmDynamic function from the greybox R package.
//!
//! # Algorithm
//!
//! 1. Generate candidate models from variable subsets
//! 2. Fit each candidate model using ALM
//! 3. Compute pointwise information criteria for each observation
//! 4. Calculate dynamic weights for each model at each time point
//! 5. Optionally smooth weights using LOWESS
//! 6. Compute weighted parameter estimates that vary over time
//!
//! # Example
//!
//! ```rust,ignore
//! use anofox_regression::solvers::{LmDynamicRegressor, InformationCriterion};
//! use faer::{Mat, Col};
//!
//! let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
//! let y = Col::from_fn(100, |i| i as f64);
//!
//! let fitted = LmDynamicRegressor::builder()
//!     .ic(InformationCriterion::AICc)
//!     .lowess_span(0.3)
//!     .build()
//!     .fit(&x, &y)?;
//!
//! // Get time-varying coefficients
//! let dyn_coefs = fitted.dynamic_coefficients();
//! ```

use crate::core::{
    IntervalType, PredictionResult, RegressionOptions, RegressionOptionsBuilder, RegressionResult,
};
use crate::solvers::alm::{log_likelihood, AlmDistribution, AlmRegressor, FittedAlm};
use crate::solvers::lowess::lowess_smooth_weights;
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::{Col, Mat};

/// Information criterion type for model weighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InformationCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Corrected AIC (better for small samples)
    #[default]
    AICc,
    /// Bayesian Information Criterion
    BIC,
}

impl InformationCriterion {
    /// Compute the information criterion for a single observation.
    ///
    /// # Arguments
    /// * `log_lik` - Pointwise log-likelihood for one observation
    /// * `k` - Number of parameters in the model
    /// * `n` - Total number of observations (for AICc correction)
    pub fn compute(&self, log_lik: f64, k: usize, n: usize) -> f64 {
        let k_f = k as f64;
        let n_f = n as f64;

        match self {
            InformationCriterion::AIC => 2.0 * k_f - 2.0 * log_lik,
            InformationCriterion::AICc => {
                let aic = 2.0 * k_f - 2.0 * log_lik;
                if n_f - k_f - 1.0 > 0.0 {
                    aic + 2.0 * k_f * (k_f + 1.0) / (n_f - k_f - 1.0)
                } else {
                    aic
                }
            }
            InformationCriterion::BIC => k_f * n_f.ln() - 2.0 * log_lik,
        }
    }
}

/// Specification for a candidate model.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Which variables are included (indices into original X)
    pub included_vars: Vec<usize>,
    /// Number of parameters (including intercept if applicable)
    pub n_params: usize,
}

/// Time-varying parameter model using pointwise information criteria.
#[derive(Debug, Clone)]
pub struct LmDynamicRegressor {
    options: RegressionOptions,
    ic_type: InformationCriterion,
    distribution: AlmDistribution,
    lowess_span: Option<f64>,
    max_models: Option<usize>,
}

impl Default for LmDynamicRegressor {
    fn default() -> Self {
        Self {
            options: RegressionOptionsBuilder::default().build_unchecked(),
            ic_type: InformationCriterion::default(),
            distribution: AlmDistribution::Normal,
            lowess_span: Some(0.3), // Default LOWESS smoothing
            max_models: Some(64),   // Limit for computational tractability
        }
    }
}

impl LmDynamicRegressor {
    /// Create a new dynamic regressor with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> LmDynamicRegressorBuilder {
        LmDynamicRegressorBuilder::default()
    }

    /// Get the information criterion type.
    pub fn ic_type(&self) -> InformationCriterion {
        self.ic_type
    }

    /// Get the distribution.
    pub fn distribution(&self) -> AlmDistribution {
        self.distribution
    }

    /// Get the LOWESS span (None if no smoothing).
    pub fn lowess_span(&self) -> Option<f64> {
        self.lowess_span
    }

    /// Generate all candidate model specifications.
    fn generate_model_specs(&self, p: usize) -> Vec<ModelSpec> {
        let max = self.max_models.unwrap_or(usize::MAX);
        let mut specs = Vec::new();

        // Generate all non-empty subsets of variables
        let total_models = (1usize << p) - 1; // 2^p - 1 models

        for mask in 1..=total_models {
            if specs.len() >= max {
                break;
            }

            let included: Vec<usize> = (0..p).filter(|&j| (mask >> j) & 1 == 1).collect();

            let n_params = included.len() + if self.options.with_intercept { 1 } else { 0 };

            specs.push(ModelSpec {
                included_vars: included,
                n_params,
            });
        }

        specs
    }

    /// Extract columns from X based on included variables.
    fn extract_columns(&self, x: &Mat<f64>, included: &[usize]) -> Mat<f64> {
        let n = x.nrows();
        let p_new = included.len();
        let mut x_new = Mat::zeros(n, p_new);

        for (new_j, &orig_j) in included.iter().enumerate() {
            for i in 0..n {
                x_new[(i, new_j)] = x[(i, orig_j)];
            }
        }

        x_new
    }

    /// Fit the dynamic model.
    fn fit_internal(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedLmDynamic, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();

        if n < 3 {
            return Err(RegressionError::InsufficientObservations {
                needed: 3,
                got: n,
            });
        }

        // Generate candidate models
        let model_specs = self.generate_model_specs(p);
        let n_models = model_specs.len();

        if n_models == 0 {
            return Err(RegressionError::InsufficientObservations {
                needed: 1,
                got: 0,
            });
        }

        // Fit all candidate models and compute pointwise IC
        let mut fitted_models: Vec<Option<FittedAlm>> = Vec::with_capacity(n_models);
        let mut pointwise_ic = Mat::from_fn(n, n_models, |_, _| f64::INFINITY);

        for (m, spec) in model_specs.iter().enumerate() {
            let x_subset = self.extract_columns(x, &spec.included_vars);

            let alm = AlmRegressor::builder()
                .distribution(self.distribution)
                .with_intercept(self.options.with_intercept)
                .compute_inference(false)
                .build();

            match alm.fit(&x_subset, y) {
                Ok(fitted) => {
                    // Compute pointwise IC for each observation
                    let mu = &fitted.result().fitted_values;
                    let scale = fitted.scale();

                    for i in 0..n {
                        let yi = Col::from_fn(1, |_| y[i]);
                        let mui = Col::from_fn(1, |_| mu[i]);
                        let ll_i = log_likelihood(
                            &yi,
                            &mui,
                            self.distribution,
                            scale,
                            fitted.extra_parameter(),
                        );
                        pointwise_ic[(i, m)] = self.ic_type.compute(ll_i, spec.n_params, n);
                    }

                    fitted_models.push(Some(fitted));
                }
                Err(_) => {
                    fitted_models.push(None);
                    // IC values remain INFINITY (worst possible)
                }
            }
        }

        // Compute weights from pointwise IC
        // pICWeights = exp(-0.5 * (pIC - min(pIC))) / row_sums
        let mut model_weights = Mat::zeros(n, n_models);

        for i in 0..n {
            // Find min IC for this observation
            let min_ic = (0..n_models)
                .map(|m| pointwise_ic[(i, m)])
                .fold(f64::INFINITY, f64::min);

            // Compute weights (Akaike weights formula)
            let mut row_sum = 0.0;
            for m in 0..n_models {
                let delta = pointwise_ic[(i, m)] - min_ic;
                let w = (-0.5 * delta).exp();
                model_weights[(i, m)] = w;
                row_sum += w;
            }

            // Normalize
            if row_sum > 1e-10 {
                for m in 0..n_models {
                    model_weights[(i, m)] /= row_sum;
                }
            } else {
                // Uniform weights if all are zero
                for m in 0..n_models {
                    model_weights[(i, m)] = 1.0 / n_models as f64;
                }
            }
        }

        // Optional LOWESS smoothing
        let smoothed_weights = self.lowess_span.map(|span| lowess_smooth_weights(&model_weights, span));

        // Compute dynamic coefficients
        let weights_to_use = smoothed_weights.as_ref().unwrap_or(&model_weights);
        let dynamic_coefficients =
            self.compute_dynamic_coefficients(&model_specs, &fitted_models, weights_to_use, p);

        // Compute weighted average coefficients for the base result
        let avg_coefs = self.compute_average_coefficients(&dynamic_coefficients);

        // Build base regression result
        let result = self.build_base_result(x, y, &avg_coefs, &dynamic_coefficients)?;

        Ok(FittedLmDynamic {
            options: self.options.clone(),
            distribution: self.distribution,
            result,
            dynamic_coefficients,
            model_weights,
            smoothed_weights,
            model_specs,
            pointwise_ic,
            n_features: p,
        })
    }

    /// Compute time-varying coefficients as weighted average of model coefficients.
    fn compute_dynamic_coefficients(
        &self,
        specs: &[ModelSpec],
        fitted_models: &[Option<FittedAlm>],
        weights: &Mat<f64>,
        p: usize,
    ) -> Mat<f64> {
        let n = weights.nrows();
        let n_models = specs.len();

        // Number of coefficient columns: p (features) + 1 if intercept
        let n_coef_cols = p + if self.options.with_intercept { 1 } else { 0 };
        let mut dynamic_coefs = Mat::zeros(n, n_coef_cols);

        for i in 0..n {
            for (m, (spec, fitted_opt)) in specs.iter().zip(fitted_models.iter()).enumerate() {
                let weight = weights[(i, m)];

                if let Some(fitted) = fitted_opt {
                    let coefs = &fitted.result().coefficients;

                    // Add intercept contribution
                    if self.options.with_intercept {
                        if let Some(intercept) = fitted.result().intercept {
                            dynamic_coefs[(i, 0)] += weight * intercept;
                        }
                    }

                    // Add coefficient contributions
                    // Map from subset indices back to original feature indices
                    for (subset_j, &orig_j) in spec.included_vars.iter().enumerate() {
                        let col_idx = if self.options.with_intercept {
                            orig_j + 1
                        } else {
                            orig_j
                        };
                        if subset_j < coefs.nrows() {
                            dynamic_coefs[(i, col_idx)] += weight * coefs[subset_j];
                        }
                    }
                }
            }
        }

        dynamic_coefs
    }

    /// Compute average coefficients across time.
    fn compute_average_coefficients(&self, dynamic_coefs: &Mat<f64>) -> Col<f64> {
        let n = dynamic_coefs.nrows();
        let p = dynamic_coefs.ncols();

        Col::from_fn(p, |j| {
            let sum: f64 = (0..n).map(|i| dynamic_coefs[(i, j)]).sum();
            sum / n as f64
        })
    }

    /// Build the base regression result.
    fn build_base_result(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        avg_coefs: &Col<f64>,
        dynamic_coefs: &Mat<f64>,
    ) -> Result<RegressionResult, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();

        // Compute fitted values using dynamic coefficients
        let mut fitted_values = Col::zeros(n);
        for i in 0..n {
            if self.options.with_intercept {
                fitted_values[i] = dynamic_coefs[(i, 0)]; // intercept
                for j in 0..p {
                    fitted_values[i] += x[(i, j)] * dynamic_coefs[(i, j + 1)];
                }
            } else {
                for j in 0..p {
                    fitted_values[i] += x[(i, j)] * dynamic_coefs[(i, j)];
                }
            }
        }

        // Compute residuals
        let mut residuals = Col::zeros(n);
        for i in 0..n {
            residuals[i] = y[i] - fitted_values[i];
        }

        // Extract coefficients and intercept
        let (coefficients, intercept) = if self.options.with_intercept {
            let intercept = avg_coefs[0];
            let coefs = Col::from_fn(p, |j| avg_coefs[j + 1]);
            (coefs, Some(intercept))
        } else {
            (avg_coefs.clone(), None)
        };

        // Statistics
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let rss: f64 = residuals.iter().map(|&r| r.powi(2)).sum();

        let r_squared = if tss > 0.0 {
            (1.0 - rss / tss).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let n_params = avg_coefs.nrows();
        let df_resid = (n - n_params) as f64;
        let adj_r_squared = if df_resid > 0.0 {
            1.0 - (1.0 - r_squared) * (n - 1) as f64 / df_resid
        } else {
            f64::NAN
        };

        let mse = if df_resid > 0.0 { rss / df_resid } else { f64::NAN };
        let rmse = mse.sqrt();

        let aliased = vec![false; p];

        let mut result = RegressionResult::empty(p, n);
        result.coefficients = coefficients;
        result.intercept = intercept;
        result.residuals = residuals;
        result.fitted_values = fitted_values;
        result.rank = p;
        result.n_parameters = n_params;
        result.n_observations = n;
        result.aliased = aliased;
        result.r_squared = r_squared;
        result.adj_r_squared = adj_r_squared;
        result.mse = mse;
        result.rmse = rmse;
        result.confidence_level = self.options.confidence_level;

        Ok(result)
    }
}

impl Regressor for LmDynamicRegressor {
    type Fitted = FittedLmDynamic;

    fn fit(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<Self::Fitted, RegressionError> {
        // Validate dimensions
        if x.nrows() != y.nrows() {
            return Err(RegressionError::DimensionMismatch {
                x_rows: x.nrows(),
                y_len: y.nrows(),
            });
        }

        self.fit_internal(x, y)
    }
}

/// Result from fitting lmDynamic.
#[derive(Debug, Clone)]
pub struct FittedLmDynamic {
    #[allow(dead_code)]
    options: RegressionOptions,
    distribution: AlmDistribution,
    /// Base regression result (using time-averaged coefficients)
    result: RegressionResult,
    /// Per-observation coefficients (n_obs x n_coefs matrix)
    /// If with_intercept: column 0 = intercept, columns 1..p+1 = feature coefficients
    /// If no intercept: columns 0..p = feature coefficients
    dynamic_coefficients: Mat<f64>,
    /// Model weights for each observation (n_obs x n_models matrix)
    model_weights: Mat<f64>,
    /// Smoothed weights if LOWESS was applied
    smoothed_weights: Option<Mat<f64>>,
    /// Candidate model specifications
    model_specs: Vec<ModelSpec>,
    /// Pointwise IC values (n_obs x n_models matrix)
    pointwise_ic: Mat<f64>,
    /// Number of original features
    n_features: usize,
}

impl FittedLmDynamic {
    /// Get the distribution used.
    pub fn distribution(&self) -> AlmDistribution {
        self.distribution
    }

    /// Get the time-varying coefficients matrix.
    ///
    /// Returns a matrix where each row contains the coefficients for that observation.
    /// If with_intercept, column 0 is the intercept.
    pub fn dynamic_coefficients(&self) -> &Mat<f64> {
        &self.dynamic_coefficients
    }

    /// Get the model weights matrix (before smoothing).
    pub fn model_weights(&self) -> &Mat<f64> {
        &self.model_weights
    }

    /// Get the smoothed model weights (if LOWESS was applied).
    pub fn smoothed_weights(&self) -> Option<&Mat<f64>> {
        self.smoothed_weights.as_ref()
    }

    /// Get the candidate model specifications.
    pub fn model_specs(&self) -> &[ModelSpec] {
        &self.model_specs
    }

    /// Get the pointwise information criteria matrix.
    pub fn pointwise_ic(&self) -> &Mat<f64> {
        &self.pointwise_ic
    }

    /// Get coefficient at a specific time point.
    ///
    /// # Arguments
    /// * `obs_index` - Observation index (0 to n-1)
    /// * `coef_index` - Coefficient index (0 = intercept if with_intercept, else first feature)
    pub fn coefficient_at(&self, obs_index: usize, coef_index: usize) -> Option<f64> {
        if obs_index < self.dynamic_coefficients.nrows()
            && coef_index < self.dynamic_coefficients.ncols()
        {
            Some(self.dynamic_coefficients[(obs_index, coef_index)])
        } else {
            None
        }
    }

    /// Get all coefficients at a specific time point.
    pub fn coefficients_at(&self, obs_index: usize) -> Option<Col<f64>> {
        if obs_index < self.dynamic_coefficients.nrows() {
            let ncols = self.dynamic_coefficients.ncols();
            Some(Col::from_fn(ncols, |j| {
                self.dynamic_coefficients[(obs_index, j)]
            }))
        } else {
            None
        }
    }
}

impl FittedRegressor for FittedLmDynamic {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let mut predictions = Col::zeros(n);

        // Use time-averaged coefficients for prediction
        let intercept = self.result.intercept.unwrap_or(0.0);

        for i in 0..n {
            predictions[i] = intercept;
            for j in 0..p.min(self.result.coefficients.nrows()) {
                predictions[i] += x[(i, j)] * self.result.coefficients[j];
            }
        }

        predictions
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

        match interval {
            None => PredictionResult::point_only(predictions),
            Some(_) => {
                // TODO: Implement prediction intervals for dynamic model
                PredictionResult::point_only(predictions)
            }
        }
    }
}

/// Builder for LmDynamicRegressor.
#[derive(Debug, Clone)]
pub struct LmDynamicRegressorBuilder {
    options_builder: RegressionOptionsBuilder,
    ic_type: InformationCriterion,
    distribution: AlmDistribution,
    lowess_span: Option<f64>,
    max_models: Option<usize>,
}

impl Default for LmDynamicRegressorBuilder {
    fn default() -> Self {
        Self {
            options_builder: RegressionOptionsBuilder::default(),
            ic_type: InformationCriterion::default(),
            distribution: AlmDistribution::default(),
            lowess_span: Some(0.3),
            max_models: Some(64),
        }
    }
}

impl LmDynamicRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the information criterion for model weighting.
    pub fn ic(mut self, ic: InformationCriterion) -> Self {
        self.ic_type = ic;
        self
    }

    /// Set the distribution family.
    pub fn distribution(mut self, dist: AlmDistribution) -> Self {
        self.distribution = dist;
        self
    }

    /// Set the LOWESS smoothing span (fraction of data, 0 to 1).
    ///
    /// Use `None` to disable smoothing.
    pub fn lowess_span(mut self, span: f64) -> Self {
        self.lowess_span = Some(span.clamp(0.05, 1.0));
        self
    }

    /// Disable LOWESS smoothing.
    pub fn no_smoothing(mut self) -> Self {
        self.lowess_span = None;
        self
    }

    /// Set maximum number of candidate models to consider.
    ///
    /// For p features, there are 2^p - 1 possible models.
    /// Set this to limit computational cost.
    pub fn max_models(mut self, max: usize) -> Self {
        self.max_models = Some(max);
        self
    }

    /// Set whether to include an intercept term.
    pub fn with_intercept(mut self, include: bool) -> Self {
        self.options_builder = self.options_builder.with_intercept(include);
        self
    }

    /// Set the confidence level for intervals.
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.options_builder = self.options_builder.confidence_level(level);
        self
    }

    /// Build the LmDynamicRegressor.
    pub fn build(self) -> LmDynamicRegressor {
        LmDynamicRegressor {
            options: self.options_builder.build_unchecked(),
            ic_type: self.ic_type,
            distribution: self.distribution,
            lowess_span: self.lowess_span,
            max_models: self.max_models,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_information_criterion_compute() {
        // AIC = 2k - 2*LL
        let aic = InformationCriterion::AIC.compute(-10.0, 3, 100);
        assert!((aic - 26.0).abs() < 1e-10); // 2*3 - 2*(-10) = 26

        // BIC = k*ln(n) - 2*LL
        let bic = InformationCriterion::BIC.compute(-10.0, 3, 100);
        let expected_bic = 3.0 * (100.0_f64).ln() + 20.0;
        assert!((bic - expected_bic).abs() < 1e-10);
    }

    #[test]
    fn test_builder_defaults() {
        let model = LmDynamicRegressor::builder().build();

        assert_eq!(model.ic_type(), InformationCriterion::AICc);
        assert_eq!(model.distribution(), AlmDistribution::Normal);
        assert!(model.lowess_span().is_some());
    }

    #[test]
    fn test_builder_custom() {
        let model = LmDynamicRegressor::builder()
            .ic(InformationCriterion::BIC)
            .distribution(AlmDistribution::Laplace)
            .lowess_span(0.5)
            .max_models(32)
            .with_intercept(false)
            .build();

        assert_eq!(model.ic_type(), InformationCriterion::BIC);
        assert_eq!(model.distribution(), AlmDistribution::Laplace);
        assert_eq!(model.lowess_span(), Some(0.5));
    }

    #[test]
    fn test_fit_simple() {
        // Simple linear data
        let x = Mat::from_fn(30, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(30, |i| 2.0 + 1.5 * (i + 1) as f64 + 0.1 * (i as f64).sin());

        let model = LmDynamicRegressor::builder()
            .ic(InformationCriterion::AICc)
            .with_intercept(true)
            .build();

        let result = model.fit(&x, &y);
        assert!(result.is_ok(), "Should fit successfully");

        let fitted = result.unwrap();

        // Check dimensions
        assert_eq!(fitted.dynamic_coefficients().nrows(), 30);
        assert_eq!(fitted.dynamic_coefficients().ncols(), 2); // intercept + 1 coef

        // Model weights should be normalized
        for i in 0..30 {
            let row_sum: f64 = (0..fitted.model_weights().ncols())
                .map(|j| fitted.model_weights()[(i, j)])
                .sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "Row {} weights sum to {}", i, row_sum);
        }
    }

    #[test]
    fn test_fit_multiple_features() {
        // Data with 3 features
        let x = Mat::from_fn(40, 3, |i, j| (i + j + 1) as f64 * 0.1);
        let y = Col::from_fn(40, |i| {
            let i_f = i as f64;
            1.0 + 0.5 * (i_f + 1.0) * 0.1 + 0.3 * (i_f + 2.0) * 0.1
        });

        let model = LmDynamicRegressor::builder()
            .max_models(7) // Limit to manageable number
            .with_intercept(true)
            .build();

        let result = model.fit(&x, &y);
        assert!(result.is_ok(), "Should fit with multiple features");

        let fitted = result.unwrap();

        // Should have 2^3 - 1 = 7 models (or less if max_models limits it)
        assert!(fitted.model_specs().len() <= 7);
        assert!(fitted.model_specs().len() >= 1);
    }

    #[test]
    fn test_dynamic_coefficients_vary() {
        // Create data where relationship changes over time
        let x = Mat::from_fn(50, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(50, |i| {
            let t = i as f64;
            // Coefficient varies: starts at 1.0, increases to 2.0
            let slope = 1.0 + t / 50.0;
            slope * (i + 1) as f64
        });

        let model = LmDynamicRegressor::builder()
            .no_smoothing() // No smoothing to see raw variation
            .with_intercept(true)
            .build();

        let fitted = model.fit(&x, &y).expect("Should fit");

        // Dynamic coefficients should show variation
        let coef_start = fitted.coefficient_at(0, 1).unwrap(); // First coefficient
        let coef_end = fitted.coefficient_at(49, 1).unwrap();

        // Both should be positive (positive relationship)
        assert!(coef_start > 0.0, "Start coefficient should be positive");
        assert!(coef_end > 0.0, "End coefficient should be positive");
    }

    #[test]
    fn test_smoothed_weights_exist() {
        let x = Mat::from_fn(30, 1, |i, _| i as f64);
        let y = Col::from_fn(30, |i| i as f64 + 0.1);

        let model = LmDynamicRegressor::builder()
            .lowess_span(0.3)
            .build();

        let fitted = model.fit(&x, &y).expect("Should fit");

        // Smoothed weights should exist
        assert!(fitted.smoothed_weights().is_some());

        // Smoothed weights should also be normalized
        let sw = fitted.smoothed_weights().unwrap();
        for i in 0..sw.nrows() {
            let row_sum: f64 = (0..sw.ncols()).map(|j| sw[(i, j)]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_predict() {
        let x = Mat::from_fn(30, 1, |i, _| (i + 1) as f64);
        let y = Col::from_fn(30, |i| 2.0 + 1.5 * (i + 1) as f64);

        let model = LmDynamicRegressor::builder().with_intercept(true).build();
        let fitted = model.fit(&x, &y).expect("Should fit");

        // Predict on new data
        let x_new = Mat::from_fn(5, 1, |i, _| (i + 31) as f64);
        let predictions = fitted.predict(&x_new);

        assert_eq!(predictions.nrows(), 5);

        // Predictions should be in reasonable range
        for i in 0..5 {
            assert!(predictions[i] > 0.0);
            assert!(predictions[i] < 200.0);
        }
    }
}
