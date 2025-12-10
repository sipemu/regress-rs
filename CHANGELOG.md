# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2025-12-10

### Changed

- Updated `faer` dependency to disable default features for WASM compatibility
  - Removes `spindle` threadpool dependency (which depends on `atomic-wait`)
  - Enables only `std` and `linalg` features
  - Linear algebra functionality remains intact; parallel operations disabled (not supported on WASM)

## [0.3.0] - 2025-12-09

### Added

- Augmented Linear Model (ALM) from greybox R package with 24+ distributions:
  - Continuous: Normal, Laplace, Student's t, Logistic, Asymmetric Laplace, Generalised Normal, S
  - Positive continuous: Log-Normal, Log-Laplace, Log-S, Log-Generalised Normal, Gamma, Inverse Gaussian, Exponential, Folded Normal, Rectified Normal
  - Bounded (0,1): Beta, Logit-Normal
  - Count data: Poisson, Negative Binomial, Binomial, Geometric
  - Ordinal: Cumulative Logistic, Cumulative Normal
  - Transformed: Box-Cox Normal
- Link functions: Identity, Log, Logit, Probit, Inverse, Sqrt, Complementary log-log
- Comprehensive R validation tests for GLM, WLS, Ridge, Elastic Net, Tweedie, and ALM

### Changed

- Updated `faer` from 0.20 to 0.23
- Updated `statrs` from 0.17 to 0.18
- Updated `getrandom` from 0.2 to 0.3 (WASM target)

### Fixed

- Prediction interval calculation for perfect fit scenarios (MSE = 0)
- Binomial deviance residuals numerical stability

## [0.2.0] - 2025-12-08

### Added

- Poisson GLM with log and identity links
- Negative Binomial GLM with theta estimation
- Binomial GLM with logit, probit, and cloglog links
- Tweedie GLM for compound Poisson-Gamma distributions
- GLM residuals: Pearson, deviance, and working residuals
- Prediction with standard errors for GLM models
- Offset support for exposure adjustment in count models

## [0.1.0] - 2025-12-08

### Added

- Ordinary Least Squares (OLS) regression with full inference
- Weighted Least Squares (WLS) regression
- Ridge Regression with L2 regularization
- Elastic Net with L1 + L2 regularization
- Recursive Least Squares (RLS) with online learning support
- Coefficient standard errors, t-statistics, and p-values
- Confidence and prediction intervals
- Model diagnostics: R², Adjusted R², RMSE, F-statistic, AIC, AICc, BIC
- Residual analysis: standardized and studentized residuals
- Leverage and influence measures: Cook's distance, DFFITS
- Variance Inflation Factor (VIF) for multicollinearity detection
- Automatic handling of collinear and constant columns
