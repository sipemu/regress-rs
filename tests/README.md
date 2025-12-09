# Test Suite

This directory contains comprehensive tests for the `anofox-regression` statistical library.

## Test Overview

| Test File | Tests | Description |
|-----------|-------|-------------|
| `ols_tests.rs` | 29 | Ordinary Least Squares regression |
| `wls_tests.rs` | 28 | Weighted Least Squares regression |
| `ridge_tests.rs` | 18 | Ridge regression (L2 regularization) |
| `elastic_net_tests.rs` | 16 | Elastic Net regression (L1 + L2) |
| `rls_tests.rs` | 13 | Recursive Least Squares |
| `alm_tests.rs` | 44 | Augmented Linear Model (ALM) |
| `diagnostics_tests.rs` | 21 | Regression diagnostics |
| `prediction_interval_tests.rs` | 10 | Prediction intervals and confidence bands |
| `validation_tests.rs` | 53 | Cross-validation against R |
| `r_validation_glm.rs` | 34 | GLM validation against R's `glm()` |
| `r_validation_alm.rs` | 9 | ALM validation against R's `greybox::alm()` |
| `r_validation_regression.rs` | 7 | WLS, Ridge, Elastic Net, Tweedie validation |

**Integration tests: 282** | **Total with unit tests: 481**

## Linear Regression Models

### OLS (Ordinary Least Squares) - `ols_tests.rs`
- Basic fitting and coefficient estimation
- Intercept handling (with/without)
- R-squared and adjusted R-squared
- Standard errors and inference
- Predictions with confidence/prediction intervals
- Multi-collinearity handling
- Edge cases (single predictor, perfect fit, etc.)

### WLS (Weighted Least Squares) - `wls_tests.rs`
- Heteroscedasticity correction
- Weight handling and validation
- Comparison with OLS under equal weights
- Weighted residuals and diagnostics

### Ridge Regression - `ridge_tests.rs`
- L2 regularization effects
- Lambda parameter tuning
- Coefficient shrinkage behavior
- Comparison with OLS as lambda → 0

### Elastic Net - `elastic_net_tests.rs`
- Combined L1/L2 regularization
- Alpha parameter (L1/L2 ratio)
- Feature selection behavior
- Convergence testing

### RLS (Recursive Least Squares) - `rls_tests.rs`
- Online/streaming updates
- Forgetting factor behavior
- Convergence to batch OLS

## Generalized Linear Models (GLM)

### Poisson GLM
- Log link (canonical)
- Identity link
- Square root link
- Offset/exposure modeling
- Deviance and residuals

### Binomial GLM
- Logit link (logistic regression)
- Probit link
- Complementary log-log link
- Binary and proportion responses

### Negative Binomial GLM
- Overdispersion modeling
- Theta (dispersion) estimation
- Fixed vs estimated theta
- Comparison with Poisson

### Gamma GLM
- Log link
- Positive continuous responses
- Dispersion estimation

### Inverse Gaussian GLM
- Log link
- Right-skewed positive data

### Tweedie GLM
- Power parameter (p) variations
- Special cases: Poisson (p=1), Gamma (p=2), Inverse Gaussian (p=3)
- Compound Poisson-Gamma (1 < p < 2)

## Augmented Linear Model (ALM) - `alm_tests.rs`

The ALM implementation supports 24+ distributions from the [greybox R package](https://github.com/config-i1/greybox):

### Continuous Distributions
- **Normal** - Standard Gaussian errors
- **Laplace** - Robust LAD (Least Absolute Deviations) regression
- **S (Asymmetric Laplace)** - Asymmetric errors
- **Student-t** - Heavy-tailed errors with estimated degrees of freedom
- **Logistic** - Logistic error distribution
- **Generalised Normal (GN)** - Flexible tail behavior

### Log-transformed Distributions
- **Log-Normal** - Multiplicative errors
- **Log-Laplace** - Robust log-space regression
- **Log-S** - Asymmetric log-space errors
- **Log-Generalised Normal**

### Box-Cox Transformed
- **BCNormal** - Box-Cox transformed normal

### Folded Distributions (non-negative)
- **Folded Normal** - Absolute value of normal
- **Folded Logistic**
- **Folded Student-t**

### Truncated Distributions
- **Truncated Normal** - Left-truncated at zero
- **Truncated Logistic**
- **Truncated Student-t**

### Discrete Distributions
- **Poisson** - Count data
- **Negative Binomial** - Overdispersed counts

### Probability Distributions (0,1)
- **Beta** - Proportions/rates
- **Logit-Normal** - Transformed proportions

### ALM Test Categories
1. **Likelihood Tests** - Log-likelihood computation for each distribution
2. **Fitting Tests** - IRLS convergence and coefficient estimation
3. **Prediction Tests** - Point predictions and intervals
4. **Inference Tests** - Standard errors, confidence intervals, p-values
5. **Link Function Tests** - Identity, Log, Logit, Probit, Inverse, Sqrt, Cloglog

## Diagnostics - `diagnostics_tests.rs`

### Residual Analysis
- Raw residuals
- Standardized residuals
- Studentized residuals (internal/external)
- Deviance residuals (GLM)
- Pearson residuals (GLM)

### Influence Measures
- **Leverage (Hat values)** - Diagonal of hat matrix
- **Cook's Distance** - Overall influence measure
- **DFFITS** - Scaled difference in fits
- **DFBETAS** - Coefficient-specific influence

### Multicollinearity
- **VIF (Variance Inflation Factor)** - Collinearity detection
- **Generalized VIF** - For categorical predictors
- **Condition Number** - Design matrix conditioning

### Outlier Detection
- High leverage point identification
- Influential observation detection
- Residual outlier flagging

## Prediction Intervals - `prediction_interval_tests.rs`

- Confidence intervals for mean response
- Prediction intervals for new observations
- Coverage probability validation
- Interval width behavior

## R Validation Tests

### OLS Validation - `validation_tests.rs`
Validates against R's `lm()` function:
- Coefficients (exact match)
- Standard errors (exact match)
- t-statistics (exact match)
- p-values (exact match)
- F-statistic (exact match)
- R², Adjusted R²
- Residuals
- AIC, BIC (formula differences noted)

### GLM Validation - `r_validation_glm.rs`
Validates against R's `glm()` function:

| Model | Coefficients | Deviance | SE | Link Functions |
|-------|-------------|----------|-----|----------------|
| Poisson | ✓ | ✓ | ✓ | log, identity |
| Binomial | ✓ | ✓ | ✓ | logit, probit, cloglog |
| Negative Binomial | ✓ | ✓ | ✓ | log |
| Gamma | ✓ | ✓ | ✓ | log |
| Inverse Gaussian | ✓ | ✓ | - | log |
| Poisson + offset | ✓ | ✓ | - | log |

### ALM Validation - `r_validation_alm.rs`
Validates against R's `greybox::alm()` function:

| Distribution | Coefficients | Scale | Log-likelihood |
|-------------|-------------|-------|----------------|
| Normal | ✓ | ✓ | ✓ |
| Laplace | ✓ | ✓ | ✓ |
| Student-t | ✓ | ✓ | ✓ |
| Log-Normal | ✓ | ✓ | ✓ |
| Poisson | ✓ | - | ✓ |
| Gamma | ✓ | ✓ | ✓ |
| Logistic | ✓ | ✓ | ✓ |

### Regression Validation - `r_validation_regression.rs`
Validates WLS, Ridge, Elastic Net, and Tweedie against R:

| Model | Coefficients | Intercept | SE | R² | Deviance |
|-------|-------------|-----------|----|----|----------|
| WLS | ✓ | ✓ | ✓ | ✓ | - |
| Ridge | ✓ | ✓ | - | - | - |
| Elastic Net | ✓ | ✓ | - | - | - |
| Tweedie (Gamma) | ✓ | ✓ | - | - | ✓ |
| Tweedie (Poisson) | ✓ | ✓ | - | - | ✓ |
| Tweedie (InvGauss) | ✓ | ✓ | - | - | ✓ |
| Tweedie (CPG) | ✓ | ✓ | - | - | ✓ |

Note: Ridge and Elastic Net are validated against closed-form solutions
rather than glmnet's coordinate descent algorithm.

## R Scripts

The `r_scripts/` directory contains R scripts used to generate validation data:

- `generate_alm_validation.R` - Generates ALM test cases using greybox package
- `generate_glm_validation.R` - Generates GLM test cases using base R's glm()
- `generate_regression_validation.R` - Generates WLS, Ridge, Elastic Net, Tweedie test cases

### Reproducing R Results

```r
# For GLM validation
source("tests/r_scripts/generate_glm_validation.R")

# For ALM validation (requires greybox package)
install.packages("greybox")
source("tests/r_scripts/generate_alm_validation.R")

# For regression validation (requires glmnet, statmod)
install.packages(c("glmnet", "statmod"))
source("tests/r_scripts/generate_regression_validation.R")
```

## Running Tests

```bash
# Run all tests
cargo test

# Run specific test file
cargo test --test ols_tests
cargo test --test r_validation_glm

# Run tests matching a pattern
cargo test poisson
cargo test r_validation

# Run with output
cargo test -- --nocapture

# Run single test
cargo test test_r_validation_poisson_log -- --exact
```

## Test Data

Tests use:
1. **Synthetic data** - Mathematically constructed test cases
2. **R-generated data** - `set.seed(42)` for reproducibility
3. **Edge cases** - Perfect fits, collinear data, outliers

## Tolerance Levels

- **Coefficient estimates**: ε = 0.01-0.05 (1-5% relative error)
- **Standard errors**: ε = 0.05-0.10
- **Deviance/likelihood**: ε = 0.1-1.0
- **Dispersion parameters**: ε = 0.1-0.5

These tolerances account for minor numerical differences between R and Rust implementations while ensuring statistical equivalence.
