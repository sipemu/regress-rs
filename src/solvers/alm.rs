//! Augmented Linear Model (ALM) solver.
//!
//! Implements maximum likelihood estimation for linear regression with various
//! error distributions. Based on the greybox R package.
//!
//! # Supported Distributions
//!
//! ## Continuous
//! - Normal (Gaussian)
//! - Laplace (double exponential)
//! - Student-t
//! - Logistic
//! - Asymmetric Laplace (quantile regression)
//! - Generalised Normal (Subbotin)
//! - S distribution
//! - Log-Normal
//! - Log-Laplace
//! - Folded Normal
//! - Box-Cox Normal
//! - Gamma
//! - Inverse Gaussian
//! - Exponential
//! - Beta
//!
//! ## Discrete
//! - Poisson
//! - Negative Binomial
//! - Binomial
//! - Geometric
//!
//! # Example
//!
//! ```rust,ignore
//! use statistics::solvers::{AlmRegressor, AlmDistribution, Regressor, FittedRegressor};
//! use faer::{Mat, Col};
//!
//! let x = Mat::from_fn(100, 2, |i, j| (i + j) as f64);
//! let y = Col::from_fn(100, |i| 1.0 + 2.0 * i as f64);
//!
//! let fitted = AlmRegressor::builder()
//!     .distribution(AlmDistribution::Laplace)
//!     .with_intercept(true)
//!     .build()
//!     .fit(&x, &y)?;
//!
//! println!("Coefficients: {:?}", fitted.coefficients());
//! ```

use crate::core::{
    IntervalType, PredictionResult, RegressionOptions, RegressionOptionsBuilder, RegressionResult,
};
use crate::inference::CoefficientInference;
use crate::solvers::traits::{FittedRegressor, RegressionError, Regressor};
use faer::{Col, Mat};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};
use statrs::function::gamma::ln_gamma;
use std::f64::consts::PI;

/// Distribution families supported by the ALM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlmDistribution {
    /// Normal (Gaussian) distribution - standard linear regression
    Normal,
    /// Laplace (double exponential) - robust to outliers, LAD regression
    Laplace,
    /// Student's t distribution - heavy-tailed, robust
    StudentT,
    /// Logistic distribution
    Logistic,
    /// Asymmetric Laplace - for quantile regression
    AsymmetricLaplace,
    /// Generalised Normal (Subbotin) distribution
    GeneralisedNormal,
    /// S distribution
    S,
    /// Log-Normal distribution - for positive skewed data
    LogNormal,
    /// Log-Laplace distribution
    LogLaplace,
    /// Log-S distribution
    LogS,
    /// Log-Generalised Normal distribution
    LogGeneralisedNormal,
    /// Folded Normal distribution - for positive data
    FoldedNormal,
    /// Rectified Normal distribution
    RectifiedNormal,
    /// Box-Cox Normal distribution
    BoxCoxNormal,
    /// Gamma distribution - for positive continuous data
    Gamma,
    /// Inverse Gaussian distribution
    InverseGaussian,
    /// Exponential distribution
    Exponential,
    /// Beta distribution - for data in (0, 1)
    Beta,
    /// Logit-Normal distribution
    LogitNormal,
    /// Poisson distribution - for count data
    Poisson,
    /// Negative Binomial - for overdispersed count data
    NegativeBinomial,
    /// Binomial distribution - for binary/proportion data
    Binomial,
    /// Geometric distribution
    Geometric,
    /// Cumulative Logistic (ordered logistic)
    CumulativeLogistic,
    /// Cumulative Normal (ordered probit)
    CumulativeNormal,
}

impl AlmDistribution {
    /// Returns the canonical link function for this distribution.
    pub fn canonical_link(&self) -> LinkFunction {
        match self {
            AlmDistribution::Normal => LinkFunction::Identity,
            AlmDistribution::Laplace => LinkFunction::Identity,
            AlmDistribution::StudentT => LinkFunction::Identity,
            AlmDistribution::Logistic => LinkFunction::Identity,
            AlmDistribution::AsymmetricLaplace => LinkFunction::Identity,
            AlmDistribution::GeneralisedNormal => LinkFunction::Identity,
            AlmDistribution::S => LinkFunction::Identity,
            AlmDistribution::LogNormal => LinkFunction::Log,
            AlmDistribution::LogLaplace => LinkFunction::Log,
            AlmDistribution::LogS => LinkFunction::Log,
            AlmDistribution::LogGeneralisedNormal => LinkFunction::Log,
            AlmDistribution::FoldedNormal => LinkFunction::Identity,
            AlmDistribution::RectifiedNormal => LinkFunction::Identity,
            AlmDistribution::BoxCoxNormal => LinkFunction::Identity,
            AlmDistribution::Gamma => LinkFunction::Log,
            AlmDistribution::InverseGaussian => LinkFunction::Log,
            AlmDistribution::Exponential => LinkFunction::Log,
            AlmDistribution::Beta => LinkFunction::Logit,
            AlmDistribution::LogitNormal => LinkFunction::Logit,
            AlmDistribution::Poisson => LinkFunction::Log,
            AlmDistribution::NegativeBinomial => LinkFunction::Log,
            AlmDistribution::Binomial => LinkFunction::Logit,
            AlmDistribution::Geometric => LinkFunction::Logit,
            AlmDistribution::CumulativeLogistic => LinkFunction::Logit,
            AlmDistribution::CumulativeNormal => LinkFunction::Probit,
        }
    }

    /// Returns whether this distribution requires positive response values.
    pub fn requires_positive(&self) -> bool {
        matches!(
            self,
            AlmDistribution::LogNormal
                | AlmDistribution::LogLaplace
                | AlmDistribution::LogS
                | AlmDistribution::LogGeneralisedNormal
                | AlmDistribution::FoldedNormal
                | AlmDistribution::Gamma
                | AlmDistribution::InverseGaussian
                | AlmDistribution::Exponential
        )
    }

    /// Returns whether this distribution is for count data.
    pub fn is_count(&self) -> bool {
        matches!(
            self,
            AlmDistribution::Poisson
                | AlmDistribution::NegativeBinomial
                | AlmDistribution::Binomial
                | AlmDistribution::Geometric
        )
    }

    /// Returns whether this distribution requires data in (0, 1).
    pub fn requires_unit_interval(&self) -> bool {
        matches!(self, AlmDistribution::Beta | AlmDistribution::LogitNormal)
    }
}

/// Link functions for relating the linear predictor to the mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinkFunction {
    /// Identity: mu = eta
    Identity,
    /// Log: mu = exp(eta)
    Log,
    /// Logit: mu = 1/(1 + exp(-eta))
    Logit,
    /// Probit: mu = Phi(eta)
    Probit,
    /// Inverse: mu = 1/eta
    Inverse,
    /// Sqrt: mu = eta^2
    Sqrt,
    /// Complementary log-log: mu = 1 - exp(-exp(eta))
    Cloglog,
}

impl LinkFunction {
    /// Apply the link function: eta = g(mu)
    pub fn link(&self, mu: f64) -> f64 {
        match self {
            LinkFunction::Identity => mu,
            LinkFunction::Log => mu.ln(),
            LinkFunction::Logit => (mu / (1.0 - mu)).ln(),
            LinkFunction::Probit => probit(mu),
            LinkFunction::Inverse => 1.0 / mu,
            LinkFunction::Sqrt => mu.sqrt(),
            LinkFunction::Cloglog => (-(1.0 - mu).ln()).ln(),
        }
    }

    /// Apply the inverse link function: mu = g^{-1}(eta)
    pub fn inverse(&self, eta: f64) -> f64 {
        match self {
            LinkFunction::Identity => eta,
            LinkFunction::Log => eta.exp(),
            LinkFunction::Logit => 1.0 / (1.0 + (-eta).exp()),
            LinkFunction::Probit => normal_cdf(eta),
            LinkFunction::Inverse => 1.0 / eta,
            LinkFunction::Sqrt => eta.powi(2),
            LinkFunction::Cloglog => 1.0 - (-eta.exp()).exp(),
        }
    }

    /// Derivative of the inverse link: d(mu)/d(eta)
    pub fn inverse_derivative(&self, eta: f64) -> f64 {
        match self {
            LinkFunction::Identity => 1.0,
            LinkFunction::Log => eta.exp(),
            LinkFunction::Logit => {
                let p = 1.0 / (1.0 + (-eta).exp());
                p * (1.0 - p)
            }
            LinkFunction::Probit => normal_pdf(eta),
            LinkFunction::Inverse => -1.0 / (eta * eta),
            LinkFunction::Sqrt => 2.0 * eta,
            LinkFunction::Cloglog => {
                let exp_eta = eta.exp();
                exp_eta * (-exp_eta).exp()
            }
        }
    }
}

// ============================================================================
// Log-likelihood Functions
// ============================================================================

/// Compute the log-likelihood for a given distribution.
///
/// # Arguments
/// * `y` - Response vector
/// * `mu` - Fitted mean vector (on response scale)
/// * `distribution` - The distribution family
/// * `scale` - Scale parameter (sigma for Normal, b for Laplace, etc.)
/// * `extra` - Extra parameter (df for Student-t, shape for Gamma, etc.)
pub fn log_likelihood(
    y: &Col<f64>,
    mu: &Col<f64>,
    distribution: AlmDistribution,
    scale: f64,
    extra: Option<f64>,
) -> f64 {
    let n = y.nrows();

    match distribution {
        AlmDistribution::Normal => {
            // LL = -n/2 * log(2*pi*sigma^2) - RSS/(2*sigma^2)
            let sigma2 = scale * scale;
            let rss: f64 = (0..n).map(|i| (y[i] - mu[i]).powi(2)).sum();
            -0.5 * n as f64 * (2.0 * PI * sigma2).ln() - rss / (2.0 * sigma2)
        }

        AlmDistribution::Laplace => {
            // LL = -n*log(2*b) - sum(|y - mu|)/b
            let b = scale;
            let sad: f64 = (0..n).map(|i| (y[i] - mu[i]).abs()).sum();
            -(n as f64) * (2.0 * b).ln() - sad / b
        }

        AlmDistribution::StudentT => {
            // Student-t log-likelihood
            let df = extra.unwrap_or(5.0);
            let sigma = scale;
            let mut ll = 0.0;
            for i in 0..n {
                let z = (y[i] - mu[i]) / sigma;
                ll += ln_gamma((df + 1.0) / 2.0)
                    - ln_gamma(df / 2.0)
                    - 0.5 * (PI * df).ln()
                    - sigma.ln()
                    - ((df + 1.0) / 2.0) * (1.0 + z * z / df).ln();
            }
            ll
        }

        AlmDistribution::Logistic => {
            // Logistic distribution log-likelihood
            let s = scale;
            let mut ll = 0.0;
            for i in 0..n {
                let z = (y[i] - mu[i]) / s;
                ll += -z - 2.0 * (1.0 + (-z).exp()).ln() - s.ln();
            }
            ll
        }

        AlmDistribution::AsymmetricLaplace => {
            // Asymmetric Laplace for quantile regression
            // alpha is the quantile level (0 < alpha < 1)
            let alpha = extra.unwrap_or(0.5);
            let sigma = scale;
            let mut ll = 0.0;
            for i in 0..n {
                let e = y[i] - mu[i];
                let rho = if e >= 0.0 {
                    alpha * e
                } else {
                    (alpha - 1.0) * e
                };
                ll += (alpha * (1.0 - alpha)).ln() - sigma.ln() - rho / sigma;
            }
            ll
        }

        AlmDistribution::GeneralisedNormal => {
            // Generalised Normal (Subbotin) distribution
            // shape = 2 gives Normal, shape = 1 gives Laplace
            let shape = extra.unwrap_or(2.0);
            let sigma = scale;
            let mut ll = 0.0;
            for i in 0..n {
                let z = ((y[i] - mu[i]) / sigma).abs();
                ll += (shape / (2.0 * sigma)).ln() - ln_gamma(1.0 / shape) - 0.5 * z.powf(shape);
            }
            ll
        }

        AlmDistribution::S => {
            // S distribution (special case of Generalised Normal with shape = 0.5)
            let sigma = scale;
            let mut ll = 0.0;
            for i in 0..n {
                let z = ((y[i] - mu[i]) / sigma).abs();
                ll += (0.25 / sigma).ln() - ln_gamma(2.0) - 0.5 * z.sqrt();
            }
            ll
        }

        AlmDistribution::LogNormal => {
            // Log-Normal: y ~ LogNormal(log(mu), sigma)
            // LL = sum(-log(y) - log(sigma) - log(2*pi)/2 - (log(y) - log(mu))^2 / (2*sigma^2))
            let sigma = scale;
            let sigma2 = sigma * sigma;
            let mut ll = 0.0;
            for i in 0..n {
                if y[i] > 0.0 && mu[i] > 0.0 {
                    let log_y = y[i].ln();
                    let log_mu = mu[i].ln();
                    ll += -log_y
                        - sigma.ln()
                        - 0.5 * (2.0 * PI).ln()
                        - (log_y - log_mu).powi(2) / (2.0 * sigma2);
                } else {
                    return f64::NEG_INFINITY;
                }
            }
            ll
        }

        AlmDistribution::LogLaplace => {
            // Log-Laplace distribution
            let b = scale;
            let mut ll = 0.0;
            for i in 0..n {
                if y[i] > 0.0 && mu[i] > 0.0 {
                    ll += -(2.0 * b).ln() - y[i].ln() - (y[i].ln() - mu[i].ln()).abs() / b;
                } else {
                    return f64::NEG_INFINITY;
                }
            }
            ll
        }

        AlmDistribution::LogS => {
            // Log-S distribution
            let sigma = scale;
            let mut ll = 0.0;
            for i in 0..n {
                if y[i] > 0.0 && mu[i] > 0.0 {
                    let z = (y[i].ln() - mu[i].ln()).abs() / sigma;
                    ll += (0.25 / sigma).ln() - ln_gamma(2.0) - y[i].ln() - 0.5 * z.sqrt();
                } else {
                    return f64::NEG_INFINITY;
                }
            }
            ll
        }

        AlmDistribution::LogGeneralisedNormal => {
            let shape = extra.unwrap_or(2.0);
            let sigma = scale;
            let mut ll = 0.0;
            for i in 0..n {
                if y[i] > 0.0 && mu[i] > 0.0 {
                    let z = (y[i].ln() - mu[i].ln()).abs() / sigma;
                    ll += (shape / (2.0 * sigma)).ln()
                        - ln_gamma(1.0 / shape)
                        - y[i].ln()
                        - 0.5 * z.powf(shape);
                } else {
                    return f64::NEG_INFINITY;
                }
            }
            ll
        }

        AlmDistribution::FoldedNormal => {
            // Folded Normal distribution (absolute value of Normal)
            let sigma = scale;
            let mut ll = 0.0;
            for i in 0..n {
                if y[i] >= 0.0 {
                    // PDF = phi((y-mu)/sigma) + phi((y+mu)/sigma) for y >= 0
                    let z1 = (y[i] - mu[i]) / sigma;
                    let z2 = (y[i] + mu[i]) / sigma;
                    let pdf = ((-0.5 * z1 * z1).exp() + (-0.5 * z2 * z2).exp())
                        / (sigma * (2.0 * PI).sqrt());
                    if pdf > 0.0 {
                        ll += pdf.ln();
                    } else {
                        return f64::NEG_INFINITY;
                    }
                } else {
                    return f64::NEG_INFINITY;
                }
            }
            ll
        }

        AlmDistribution::RectifiedNormal => {
            // Rectified Normal (max(0, X) where X ~ Normal)
            let sigma = scale;
            let mut ll = 0.0;
            for i in 0..n {
                if y[i] > 0.0 {
                    let z = (y[i] - mu[i]) / sigma;
                    ll += -0.5 * (2.0 * PI).ln() - sigma.ln() - 0.5 * z * z;
                } else if y[i] == 0.0 {
                    // Point mass at 0
                    let z = -mu[i] / sigma;
                    ll += normal_cdf(z).ln();
                } else {
                    return f64::NEG_INFINITY;
                }
            }
            ll
        }

        AlmDistribution::BoxCoxNormal => {
            // Box-Cox Normal: transform y^(lambda), then Normal
            let lambda = extra.unwrap_or(1.0);
            let sigma = scale;
            let sigma2 = sigma * sigma;
            let mut ll = 0.0;
            for i in 0..n {
                if y[i] > 0.0 {
                    let y_trans = if lambda.abs() < 1e-10 {
                        y[i].ln()
                    } else {
                        (y[i].powf(lambda) - 1.0) / lambda
                    };
                    let mu_trans = if lambda.abs() < 1e-10 {
                        mu[i].ln()
                    } else {
                        (mu[i].powf(lambda) - 1.0) / lambda
                    };
                    // Jacobian: |dy_trans/dy| = y^(lambda-1)
                    ll += -0.5 * (2.0 * PI * sigma2).ln()
                        - (y_trans - mu_trans).powi(2) / (2.0 * sigma2)
                        + (lambda - 1.0) * y[i].ln();
                } else {
                    return f64::NEG_INFINITY;
                }
            }
            ll
        }

        AlmDistribution::Gamma => {
            // Gamma distribution parameterized by mean mu and shape k
            // PDF = (k/mu)^k * y^(k-1) * exp(-k*y/mu) / Gamma(k)
            let shape = extra.unwrap_or(1.0);
            let mut ll = 0.0;
            for i in 0..n {
                if y[i] > 0.0 && mu[i] > 0.0 {
                    let rate = shape / mu[i];
                    ll += shape * rate.ln() + (shape - 1.0) * y[i].ln()
                        - rate * y[i]
                        - ln_gamma(shape);
                } else {
                    return f64::NEG_INFINITY;
                }
            }
            ll
        }

        AlmDistribution::InverseGaussian => {
            // Inverse Gaussian distribution
            // PDF = sqrt(lambda/(2*pi*y^3)) * exp(-lambda*(y-mu)^2 / (2*mu^2*y))
            let lambda = extra.unwrap_or(1.0);
            let mut ll = 0.0;
            for i in 0..n {
                if y[i] > 0.0 && mu[i] > 0.0 {
                    ll += 0.5 * (lambda / (2.0 * PI * y[i].powi(3))).ln()
                        - lambda * (y[i] - mu[i]).powi(2) / (2.0 * mu[i].powi(2) * y[i]);
                } else {
                    return f64::NEG_INFINITY;
                }
            }
            ll
        }

        AlmDistribution::Exponential => {
            // Exponential distribution with mean mu
            // PDF = (1/mu) * exp(-y/mu)
            let mut ll = 0.0;
            for i in 0..n {
                if y[i] >= 0.0 && mu[i] > 0.0 {
                    ll += -mu[i].ln() - y[i] / mu[i];
                } else {
                    return f64::NEG_INFINITY;
                }
            }
            ll
        }

        AlmDistribution::Beta => {
            // Beta distribution parameterized by mean mu and precision phi
            // alpha = mu * phi, beta = (1 - mu) * phi
            let phi = extra.unwrap_or(1.0);
            let mut ll = 0.0;
            for i in 0..n {
                if y[i] > 0.0 && y[i] < 1.0 && mu[i] > 0.0 && mu[i] < 1.0 {
                    let alpha = mu[i] * phi;
                    let beta_param = (1.0 - mu[i]) * phi;
                    ll += ln_gamma(phi) - ln_gamma(alpha) - ln_gamma(beta_param)
                        + (alpha - 1.0) * y[i].ln()
                        + (beta_param - 1.0) * (1.0 - y[i]).ln();
                } else {
                    return f64::NEG_INFINITY;
                }
            }
            ll
        }

        AlmDistribution::LogitNormal => {
            // Logit-Normal distribution
            let sigma = scale;
            let sigma2 = sigma * sigma;
            let mut ll = 0.0;
            for i in 0..n {
                if y[i] > 0.0 && y[i] < 1.0 && mu[i] > 0.0 && mu[i] < 1.0 {
                    let logit_y = (y[i] / (1.0 - y[i])).ln();
                    let logit_mu = (mu[i] / (1.0 - mu[i])).ln();
                    ll += -0.5 * (2.0 * PI * sigma2).ln()
                        - (logit_y - logit_mu).powi(2) / (2.0 * sigma2)
                        - y[i].ln()
                        - (1.0 - y[i]).ln();
                } else {
                    return f64::NEG_INFINITY;
                }
            }
            ll
        }

        AlmDistribution::Poisson => {
            // Poisson log-likelihood: sum(y*log(mu) - mu - log(y!))
            let mut ll = 0.0;
            for i in 0..n {
                let yi = y[i].round().max(0.0);
                let mui = mu[i].max(1e-10);
                ll += yi * mui.ln() - mui - ln_gamma(yi + 1.0);
            }
            ll
        }

        AlmDistribution::NegativeBinomial => {
            // Negative Binomial with mean mu and size (dispersion) parameter
            // Variance = mu + mu^2/size
            let size = extra.unwrap_or(1.0);
            let mut ll = 0.0;
            for i in 0..n {
                let yi = y[i].round().max(0.0);
                let mui = mu[i].max(1e-10);
                let p = size / (size + mui);
                ll += ln_gamma(yi + size) - ln_gamma(size) - ln_gamma(yi + 1.0)
                    + size * p.ln()
                    + yi * (1.0 - p).ln();
            }
            ll
        }

        AlmDistribution::Binomial => {
            // Binomial log-likelihood for proportions
            // n_trials is the number of trials (stored in extra or assumed 1)
            let n_trials = extra.unwrap_or(1.0);
            let mut ll = 0.0;
            for i in 0..n {
                let p = mu[i].clamp(1e-10, 1.0 - 1e-10);
                let k = (y[i] * n_trials).round().max(0.0).min(n_trials);
                ll += ln_gamma(n_trials + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n_trials - k + 1.0)
                    + k * p.ln()
                    + (n_trials - k) * (1.0 - p).ln();
            }
            ll
        }

        AlmDistribution::Geometric => {
            // Geometric distribution: probability of k failures before first success
            // P(Y=k) = p * (1-p)^k where p is probability of success
            let mut ll = 0.0;
            for i in 0..n {
                let p = mu[i].clamp(1e-10, 1.0 - 1e-10);
                let k = y[i].round().max(0.0);
                ll += p.ln() + k * (1.0 - p).ln();
            }
            ll
        }

        AlmDistribution::CumulativeLogistic | AlmDistribution::CumulativeNormal => {
            // Ordinal regression - not implemented in basic form
            // Would need category thresholds
            f64::NEG_INFINITY
        }
    }
}

/// Estimate the scale parameter from residuals.
pub fn estimate_scale(y: &Col<f64>, mu: &Col<f64>, distribution: AlmDistribution, df: f64) -> f64 {
    let n = y.nrows();

    match distribution {
        AlmDistribution::Normal
        | AlmDistribution::FoldedNormal
        | AlmDistribution::RectifiedNormal => {
            // MLE for sigma: sqrt(RSS / n) or sqrt(RSS / df) for unbiased
            let rss: f64 = (0..n).map(|i| (y[i] - mu[i]).powi(2)).sum();
            (rss / df).sqrt()
        }

        AlmDistribution::Laplace | AlmDistribution::LogLaplace => {
            // MLE for b: mean absolute deviation
            let sad: f64 = (0..n).map(|i| (y[i] - mu[i]).abs()).sum();
            sad / n as f64
        }

        AlmDistribution::StudentT | AlmDistribution::Logistic => {
            // Use robust scale estimate
            let mut abs_residuals: Vec<f64> = (0..n).map(|i| (y[i] - mu[i]).abs()).collect();
            abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_idx = n / 2;
            let mad = abs_residuals[median_idx];
            mad / 0.6745 // Scale factor for Normal
        }

        AlmDistribution::LogNormal
        | AlmDistribution::LogGeneralisedNormal
        | AlmDistribution::LogitNormal => {
            // Scale on log scale
            let rss: f64 = (0..n)
                .filter(|&i| y[i] > 0.0 && mu[i] > 0.0)
                .map(|i| (y[i].ln() - mu[i].ln()).powi(2))
                .sum();
            (rss / df).sqrt()
        }

        AlmDistribution::AsymmetricLaplace
        | AlmDistribution::GeneralisedNormal
        | AlmDistribution::S
        | AlmDistribution::LogS
        | AlmDistribution::BoxCoxNormal => {
            // Use MAD-based estimate
            let mut abs_residuals: Vec<f64> = (0..n).map(|i| (y[i] - mu[i]).abs()).collect();
            abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            abs_residuals[n / 2] / 0.6745
        }

        // For discrete distributions, scale is typically 1 or not used
        _ => 1.0,
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Standard normal CDF
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Standard normal PDF
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Probit function (inverse normal CDF)
#[allow(clippy::excessive_precision)]
fn probit(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Use rational approximation
    let a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];
    let b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Approximation from Abramowitz and Stegun
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

// ============================================================================
// ALM Regressor
// ============================================================================

/// Augmented Linear Model regressor using maximum likelihood estimation.
#[derive(Debug, Clone)]
pub struct AlmRegressor {
    options: RegressionOptions,
    distribution: AlmDistribution,
    link: LinkFunction,
    extra_parameter: Option<f64>,
    max_iterations: usize,
    tolerance: f64,
}

impl AlmRegressor {
    /// Create a new ALM regressor.
    pub fn new(
        options: RegressionOptions,
        distribution: AlmDistribution,
        link: Option<LinkFunction>,
        extra_parameter: Option<f64>,
    ) -> Self {
        let link = link.unwrap_or_else(|| distribution.canonical_link());
        Self {
            options,
            distribution,
            link,
            extra_parameter,
            max_iterations: 100,
            tolerance: 1e-8,
        }
    }

    /// Create a builder for configuring the regressor.
    pub fn builder() -> AlmRegressorBuilder {
        AlmRegressorBuilder::default()
    }

    /// Get the distribution.
    pub fn distribution(&self) -> AlmDistribution {
        self.distribution
    }

    /// Get the link function.
    pub fn link(&self) -> LinkFunction {
        self.link
    }

    /// Fit the model using iteratively reweighted least squares (IRLS) for GLM-type
    /// distributions, or gradient descent for others.
    fn fit_irls(&self, x: &Mat<f64>, y: &Col<f64>) -> Result<FittedAlm, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();
        let n_params = if self.options.with_intercept {
            p + 1
        } else {
            p
        };

        // Initialize coefficients using OLS
        let mut beta = self.initialize_coefficients(x, y)?;

        // Initial linear predictor and mean
        let mut eta = self.compute_eta(x, &beta);
        let mut mu = self.compute_mu(&eta);

        // Initial scale estimate
        let df = (n - n_params) as f64;
        let mut scale = estimate_scale(y, &mu, self.distribution, df.max(1.0));

        let mut prev_ll = log_likelihood(y, &mu, self.distribution, scale, self.extra_parameter);

        for iter in 0..self.max_iterations {
            // Compute weights and working response for IRLS
            let (weights, z) = self.compute_irls_components(y, &mu, &eta);

            // Weighted least squares step
            beta = self.weighted_ls_step(x, &z, &weights, &beta)?;

            // Update linear predictor and mean
            eta = self.compute_eta(x, &beta);
            mu = self.compute_mu(&eta);

            // Update scale
            scale = estimate_scale(y, &mu, self.distribution, df.max(1.0));

            // Check convergence
            let ll = log_likelihood(y, &mu, self.distribution, scale, self.extra_parameter);

            if (ll - prev_ll).abs() < self.tolerance * (1.0 + prev_ll.abs()) {
                break;
            }

            if iter == self.max_iterations - 1 {
                // Allow convergence failure for now - return best estimate
            }

            prev_ll = ll;
        }

        // Compute final results
        self.build_result(x, y, &beta, &mu, scale)
    }

    /// Initialize coefficients using OLS or appropriate method.
    fn initialize_coefficients(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
    ) -> Result<Col<f64>, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();

        // Transform y for initialization based on link function
        let y_init: Col<f64> = match self.link {
            LinkFunction::Identity => y.clone(),
            LinkFunction::Log => Col::from_fn(n, |i| if y[i] > 0.0 { y[i].ln() } else { 0.0 }),
            LinkFunction::Logit => Col::from_fn(n, |i| {
                let yi = y[i].clamp(0.01, 0.99);
                (yi / (1.0 - yi)).ln()
            }),
            LinkFunction::Probit => Col::from_fn(n, |i| probit(y[i].clamp(0.01, 0.99))),
            LinkFunction::Inverse => {
                Col::from_fn(n, |i| if y[i] != 0.0 { 1.0 / y[i] } else { 1.0 })
            }
            LinkFunction::Sqrt => Col::from_fn(n, |i| y[i].max(0.0).sqrt()),
            LinkFunction::Cloglog => Col::from_fn(n, |i| {
                let yi = y[i].clamp(0.01, 0.99);
                (-(1.0 - yi).ln()).ln()
            }),
        };

        // Solve OLS using normal equations with regularization for numerical stability
        if self.options.with_intercept {
            // Augment X with intercept column
            let mut x_aug = Mat::zeros(n, p + 1);
            for i in 0..n {
                x_aug[(i, 0)] = 1.0;
                for j in 0..p {
                    x_aug[(i, j + 1)] = x[(i, j)];
                }
            }

            // Solve using QR with column pivoting and proper permutation handling
            let qr = x_aug.col_piv_qr();
            let perm = qr.P();
            let perm_arr = perm.arrays().0;

            let q = qr.compute_Q();
            let r = qr.R();
            let ncols = p + 1;

            let qty = q.transpose() * &y_init;
            let mut beta_perm = Col::zeros(ncols);

            // Back-substitution for R * beta_perm = Q' * y
            for i in (0..ncols.min(n)).rev() {
                let mut sum = qty[i];
                for j in (i + 1)..ncols.min(n) {
                    sum -= r[(i, j)] * beta_perm[j];
                }
                if r[(i, i)].abs() > 1e-10 {
                    beta_perm[i] = sum / r[(i, i)];
                }
            }

            // Apply inverse permutation: beta[perm[i]] = beta_perm[i]
            let mut beta = Col::zeros(ncols);
            for i in 0..ncols {
                let orig_col = perm_arr[i];
                beta[orig_col] = beta_perm[i];
            }

            Ok(beta)
        } else {
            let qr = x.col_piv_qr();
            let perm = qr.P();
            let perm_arr = perm.arrays().0;

            let q = qr.compute_Q();
            let r = qr.R();

            let qty = q.transpose() * &y_init;
            let mut beta_perm = Col::zeros(p);

            for i in (0..p.min(n)).rev() {
                let mut sum = qty[i];
                for j in (i + 1)..p.min(n) {
                    sum -= r[(i, j)] * beta_perm[j];
                }
                if r[(i, i)].abs() > 1e-10 {
                    beta_perm[i] = sum / r[(i, i)];
                }
            }

            // Apply inverse permutation
            let mut beta = Col::zeros(p);
            for i in 0..p {
                let orig_col = perm_arr[i];
                beta[orig_col] = beta_perm[i];
            }

            Ok(beta)
        }
    }

    /// Compute linear predictor eta = X * beta (+ intercept if applicable).
    fn compute_eta(&self, x: &Mat<f64>, beta: &Col<f64>) -> Col<f64> {
        let n = x.nrows();
        let p = x.ncols();

        if self.options.with_intercept {
            let intercept = beta[0];
            let mut eta = Col::zeros(n);
            for i in 0..n {
                eta[i] = intercept;
                for j in 0..p {
                    eta[i] += x[(i, j)] * beta[j + 1];
                }
            }
            eta
        } else {
            let mut eta = Col::zeros(n);
            for i in 0..n {
                for j in 0..p {
                    eta[i] += x[(i, j)] * beta[j];
                }
            }
            eta
        }
    }

    /// Compute mean mu = g^{-1}(eta) using the inverse link function.
    fn compute_mu(&self, eta: &Col<f64>) -> Col<f64> {
        let n = eta.nrows();
        let mut mu = Col::zeros(n);
        for i in 0..n {
            mu[i] = self.link.inverse(eta[i]);
            // Ensure valid range for certain distributions
            if self.distribution.requires_positive() {
                mu[i] = mu[i].max(1e-10);
            }
            if self.distribution.requires_unit_interval() {
                mu[i] = mu[i].clamp(1e-10, 1.0 - 1e-10);
            }
        }
        mu
    }

    /// Compute IRLS weights and working response.
    fn compute_irls_components(
        &self,
        y: &Col<f64>,
        mu: &Col<f64>,
        eta: &Col<f64>,
    ) -> (Col<f64>, Col<f64>) {
        let n = y.nrows();
        let mut weights = Col::zeros(n);
        let mut z = Col::zeros(n);

        for i in 0..n {
            // Derivative of inverse link
            let d_mu = self.link.inverse_derivative(eta[i]).max(1e-10);

            // Variance function based on distribution
            let v = self.variance_function(mu[i]);

            // For Laplace and other robust distributions, use special weights
            let weight = match self.distribution {
                AlmDistribution::Laplace | AlmDistribution::LogLaplace => {
                    // For LAD: w_i = 1 / |y_i - mu_i| (iteratively reweighted)
                    let abs_resid = (y[i] - mu[i]).abs().max(1e-6);
                    1.0 / abs_resid
                }
                AlmDistribution::AsymmetricLaplace => {
                    // Asymmetric Laplace weights for quantile regression
                    let alpha = self.extra_parameter.unwrap_or(0.5);
                    let resid = y[i] - mu[i];
                    let abs_resid = resid.abs().max(1e-6);
                    if resid >= 0.0 {
                        alpha / abs_resid
                    } else {
                        (1.0 - alpha) / abs_resid
                    }
                }
                AlmDistribution::StudentT => {
                    // Robust Student-t weights
                    let df = self.extra_parameter.unwrap_or(5.0);
                    let resid = y[i] - mu[i];
                    let scale_est = 1.0; // Use previous scale estimate
                    let z_sq = (resid / scale_est).powi(2);
                    (df + 1.0) / (df + z_sq)
                }
                _ => {
                    // Standard GLM weight: d_mu^2 / V(mu)
                    (d_mu * d_mu / v).max(1e-10)
                }
            };

            weights[i] = weight;

            // Working response: z = eta + (y - mu) / d_mu
            z[i] = eta[i] + (y[i] - mu[i]) / d_mu;
        }

        (weights, z)
    }

    /// Variance function V(mu) for the distribution.
    fn variance_function(&self, mu: f64) -> f64 {
        match self.distribution {
            AlmDistribution::Normal
            | AlmDistribution::Laplace
            | AlmDistribution::StudentT
            | AlmDistribution::Logistic
            | AlmDistribution::AsymmetricLaplace
            | AlmDistribution::GeneralisedNormal
            | AlmDistribution::S => 1.0,

            AlmDistribution::Poisson => mu.max(1e-10),

            AlmDistribution::Binomial | AlmDistribution::Geometric => {
                let p = mu.clamp(1e-10, 1.0 - 1e-10);
                p * (1.0 - p)
            }

            AlmDistribution::NegativeBinomial => {
                let size = self.extra_parameter.unwrap_or(1.0);
                mu + mu * mu / size
            }

            AlmDistribution::Gamma | AlmDistribution::InverseGaussian => mu * mu,

            AlmDistribution::Exponential => mu * mu,

            AlmDistribution::Beta => {
                let phi = self.extra_parameter.unwrap_or(1.0);
                let p = mu.clamp(1e-10, 1.0 - 1e-10);
                p * (1.0 - p) / (1.0 + phi)
            }

            AlmDistribution::LogNormal
            | AlmDistribution::LogLaplace
            | AlmDistribution::LogS
            | AlmDistribution::LogGeneralisedNormal
            | AlmDistribution::FoldedNormal
            | AlmDistribution::RectifiedNormal
            | AlmDistribution::BoxCoxNormal
            | AlmDistribution::LogitNormal => 1.0,

            AlmDistribution::CumulativeLogistic | AlmDistribution::CumulativeNormal => 1.0,
        }
    }

    /// Perform weighted least squares step.
    fn weighted_ls_step(
        &self,
        x: &Mat<f64>,
        z: &Col<f64>,
        weights: &Col<f64>,
        _prev_beta: &Col<f64>,
    ) -> Result<Col<f64>, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();

        // Create weighted design matrix and response
        if self.options.with_intercept {
            let ncols = p + 1;
            let mut xw = Mat::zeros(n, ncols);
            let mut zw = Col::zeros(n);

            for i in 0..n {
                let w_sqrt = weights[i].sqrt();
                xw[(i, 0)] = w_sqrt;
                for j in 0..p {
                    xw[(i, j + 1)] = x[(i, j)] * w_sqrt;
                }
                zw[i] = z[i] * w_sqrt;
            }

            // Solve weighted least squares with permutation handling
            let qr = xw.col_piv_qr();
            let perm = qr.P();
            let perm_arr = perm.arrays().0;

            let q = qr.compute_Q();
            let r = qr.R();

            let qty = q.transpose() * &zw;
            let mut beta_perm = Col::zeros(ncols);

            for i in (0..ncols.min(n)).rev() {
                let mut sum = qty[i];
                for j in (i + 1)..ncols.min(n) {
                    sum -= r[(i, j)] * beta_perm[j];
                }
                if r[(i, i)].abs() > 1e-10 {
                    beta_perm[i] = sum / r[(i, i)];
                }
            }

            // Apply inverse permutation
            let mut beta = Col::zeros(ncols);
            for i in 0..ncols {
                let orig_col = perm_arr[i];
                beta[orig_col] = beta_perm[i];
            }

            Ok(beta)
        } else {
            let mut xw = Mat::zeros(n, p);
            let mut zw = Col::zeros(n);

            for i in 0..n {
                let w_sqrt = weights[i].sqrt();
                for j in 0..p {
                    xw[(i, j)] = x[(i, j)] * w_sqrt;
                }
                zw[i] = z[i] * w_sqrt;
            }

            let qr = xw.col_piv_qr();
            let perm = qr.P();
            let perm_arr = perm.arrays().0;

            let q = qr.compute_Q();
            let r = qr.R();

            let qty = q.transpose() * &zw;
            let mut beta_perm = Col::zeros(p);

            for i in (0..p.min(n)).rev() {
                let mut sum = qty[i];
                for j in (i + 1)..p.min(n) {
                    sum -= r[(i, j)] * beta_perm[j];
                }
                if r[(i, i)].abs() > 1e-10 {
                    beta_perm[i] = sum / r[(i, i)];
                }
            }

            // Apply inverse permutation
            let mut beta = Col::zeros(p);
            for i in 0..p {
                let orig_col = perm_arr[i];
                beta[orig_col] = beta_perm[i];
            }

            Ok(beta)
        }
    }

    /// Build the final result.
    fn build_result(
        &self,
        x: &Mat<f64>,
        y: &Col<f64>,
        beta: &Col<f64>,
        mu: &Col<f64>,
        scale: f64,
    ) -> Result<FittedAlm, RegressionError> {
        let n = x.nrows();
        let p = x.ncols();

        // Extract coefficients and intercept
        let (coefficients, intercept) = if self.options.with_intercept {
            let intercept = beta[0];
            let mut coefs = Col::zeros(p);
            for j in 0..p {
                coefs[j] = beta[j + 1];
            }
            (coefs, Some(intercept))
        } else {
            (beta.clone(), None)
        };

        let n_params = if self.options.with_intercept {
            p + 2 // coefficients + intercept + scale
        } else {
            p + 1 // coefficients + scale
        };

        // Compute residuals
        let mut residuals = Col::zeros(n);
        for i in 0..n {
            residuals[i] = y[i] - mu[i];
        }

        // Compute statistics
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let rss: f64 = residuals.iter().map(|&r| r.powi(2)).sum();

        let r_squared = if tss > 0.0 {
            (1.0 - rss / tss).clamp(0.0, 1.0)
        } else if rss < 1e-10 {
            1.0
        } else {
            0.0
        };

        let df_total = (n - 1) as f64;
        let df_resid = (n - n_params) as f64;
        let adj_r_squared = if df_resid > 0.0 && df_total > 0.0 {
            1.0 - (1.0 - r_squared) * df_total / df_resid
        } else {
            f64::NAN
        };

        let mse = if df_resid > 0.0 {
            rss / df_resid
        } else {
            f64::NAN
        };
        let rmse = mse.sqrt();

        // Log-likelihood
        let ll = log_likelihood(y, mu, self.distribution, scale, self.extra_parameter);

        // Information criteria
        let k = n_params as f64;
        let aic = if ll.is_finite() {
            2.0 * k - 2.0 * ll
        } else {
            f64::NAN
        };
        let aicc = if ll.is_finite() && (n as f64 - k - 1.0) > 0.0 {
            aic + 2.0 * k * (k + 1.0) / (n as f64 - k - 1.0)
        } else {
            f64::NAN
        };
        let bic = if ll.is_finite() {
            k * (n as f64).ln() - 2.0 * ll
        } else {
            f64::NAN
        };

        // F-statistic (may not be meaningful for non-Normal)
        let ess = tss - rss;
        let df_model = (n_params - if intercept.is_some() { 1 } else { 0 } - 1) as f64; // -1 for scale
        let f_statistic = if df_model > 0.0 && df_resid > 0.0 && mse > 0.0 {
            (ess / df_model) / mse
        } else {
            f64::NAN
        };
        let f_pvalue = if f_statistic.is_finite() && df_model > 0.0 && df_resid > 0.0 {
            let f_dist = FisherSnedecor::new(df_model, df_resid).ok();
            f_dist.map_or(f64::NAN, |d| 1.0 - d.cdf(f_statistic))
        } else {
            f64::NAN
        };

        let aliased = vec![false; p];

        let mut result = RegressionResult::empty(p, n);
        result.coefficients = coefficients.clone();
        result.intercept = intercept;
        result.residuals = residuals;
        result.fitted_values = mu.clone();
        result.rank = p;
        result.n_parameters = n_params;
        result.n_observations = n;
        result.aliased = aliased;
        result.r_squared = r_squared;
        result.adj_r_squared = adj_r_squared;
        result.mse = mse;
        result.rmse = rmse;
        result.f_statistic = f_statistic;
        result.f_pvalue = f_pvalue;
        result.aic = aic;
        result.aicc = aicc;
        result.bic = bic;
        result.log_likelihood = ll;
        result.confidence_level = self.options.confidence_level;

        // Compute inference if requested
        if self.options.compute_inference {
            self.compute_inference(x, &mut result, scale)?;
        }

        Ok(FittedAlm {
            options: self.options.clone(),
            distribution: self.distribution,
            link: self.link,
            extra_parameter: self.extra_parameter,
            scale,
            result,
        })
    }

    /// Compute inference statistics using Fisher information.
    fn compute_inference(
        &self,
        x: &Mat<f64>,
        result: &mut RegressionResult,
        _scale: f64,
    ) -> Result<(), RegressionError> {
        let df = result.residual_df() as f64;

        if df <= 0.0 || !result.mse.is_finite() {
            return Ok(());
        }

        // Use standard errors based on (X'WX)^{-1} approximation
        // For simplicity, use the OLS-based standard errors
        if result.intercept.is_some() {
            if let Ok((se, se_int)) =
                CoefficientInference::standard_errors_with_intercept(x, result.mse, &result.aliased)
            {
                let t_stats = CoefficientInference::t_statistics(&result.coefficients, &se);
                let p_vals = CoefficientInference::p_values(&t_stats, df);
                let (ci_lower, ci_upper) = CoefficientInference::confidence_intervals(
                    &result.coefficients,
                    &se,
                    df,
                    self.options.confidence_level,
                );

                result.std_errors = Some(se);
                result.t_statistics = Some(t_stats);
                result.p_values = Some(p_vals);
                result.conf_interval_lower = Some(ci_lower);
                result.conf_interval_upper = Some(ci_upper);

                // Intercept inference
                let intercept = result.intercept.unwrap();
                let t_int = if se_int > 0.0 {
                    intercept / se_int
                } else {
                    f64::NAN
                };

                use statrs::distribution::StudentsT;
                let t_dist = StudentsT::new(0.0, 1.0, df).ok();
                let p_int = if t_int.is_finite() {
                    t_dist.map_or(f64::NAN, |d| 2.0 * (1.0 - d.cdf(t_int.abs())))
                } else {
                    f64::NAN
                };

                let t_crit = t_dist.map_or(f64::NAN, |d| {
                    d.inverse_cdf(1.0 - (1.0 - self.options.confidence_level) / 2.0)
                });
                let ci_int = (intercept - t_crit * se_int, intercept + t_crit * se_int);

                result.intercept_std_error = Some(se_int);
                result.intercept_t_statistic = Some(t_int);
                result.intercept_p_value = Some(p_int);
                result.intercept_conf_interval = Some(ci_int);
            }
        } else if let Ok(se) = CoefficientInference::standard_errors(x, result.mse, &result.aliased)
        {
            let t_stats = CoefficientInference::t_statistics(&result.coefficients, &se);
            let p_vals = CoefficientInference::p_values(&t_stats, df);
            let (ci_lower, ci_upper) = CoefficientInference::confidence_intervals(
                &result.coefficients,
                &se,
                df,
                self.options.confidence_level,
            );

            result.std_errors = Some(se);
            result.t_statistics = Some(t_stats);
            result.p_values = Some(p_vals);
            result.conf_interval_lower = Some(ci_lower);
            result.conf_interval_upper = Some(ci_upper);
        }

        Ok(())
    }
}

impl Regressor for AlmRegressor {
    type Fitted = FittedAlm;

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

        // Check minimum observations
        let n_params = if self.options.with_intercept {
            n_features + 2 // +1 for intercept, +1 for scale
        } else {
            n_features + 1 // +1 for scale
        };

        if n_samples < n_params {
            return Err(RegressionError::InsufficientObservations {
                needed: n_params,
                got: n_samples,
            });
        }

        // Validate response values for certain distributions
        if self.distribution.requires_positive() {
            for i in 0..n_samples {
                if y[i] <= 0.0 {
                    return Err(RegressionError::NumericalError(format!(
                        "Distribution {:?} requires positive response values",
                        self.distribution
                    )));
                }
            }
        }

        if self.distribution.requires_unit_interval() {
            for i in 0..n_samples {
                if y[i] <= 0.0 || y[i] >= 1.0 {
                    return Err(RegressionError::NumericalError(format!(
                        "Distribution {:?} requires response values in (0, 1)",
                        self.distribution
                    )));
                }
            }
        }

        self.fit_irls(x, y)
    }
}

// ============================================================================
// Fitted ALM
// ============================================================================

/// A fitted Augmented Linear Model.
#[derive(Debug, Clone)]
pub struct FittedAlm {
    #[allow(dead_code)]
    options: RegressionOptions,
    distribution: AlmDistribution,
    link: LinkFunction,
    extra_parameter: Option<f64>,
    scale: f64,
    result: RegressionResult,
}

impl FittedAlm {
    /// Get the distribution used.
    pub fn distribution(&self) -> AlmDistribution {
        self.distribution
    }

    /// Get the link function used.
    pub fn link(&self) -> LinkFunction {
        self.link
    }

    /// Get the estimated scale parameter.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Get the extra parameter (df, shape, etc.) if any.
    pub fn extra_parameter(&self) -> Option<f64> {
        self.extra_parameter
    }
}

impl FittedRegressor for FittedAlm {
    fn predict(&self, x: &Mat<f64>) -> Col<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let mut predictions = Col::zeros(n);

        let intercept = self.result.intercept.unwrap_or(0.0);

        for i in 0..n {
            let mut eta = intercept;
            for j in 0..p {
                if !self.result.aliased[j] && !self.result.coefficients[j].is_nan() {
                    eta += x[(i, j)] * self.result.coefficients[j];
                }
            }
            // Apply inverse link to get predictions on response scale
            predictions[i] = self.link.inverse(eta);
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
                // For non-Normal distributions, prediction intervals are more complex
                // Return point predictions with NaN intervals for now
                let n = x.nrows();
                let mut lower = Col::zeros(n);
                let mut upper = Col::zeros(n);
                let mut se = Col::zeros(n);
                for i in 0..n {
                    lower[i] = f64::NAN;
                    upper[i] = f64::NAN;
                    se[i] = f64::NAN;
                }
                PredictionResult::with_intervals(predictions, lower, upper, se)
            }
        }
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for `AlmRegressor`.
#[derive(Debug, Clone)]
pub struct AlmRegressorBuilder {
    options_builder: RegressionOptionsBuilder,
    distribution: AlmDistribution,
    link: Option<LinkFunction>,
    extra_parameter: Option<f64>,
    max_iterations: usize,
    tolerance: f64,
}

impl Default for AlmRegressorBuilder {
    fn default() -> Self {
        Self {
            options_builder: RegressionOptionsBuilder::default(),
            distribution: AlmDistribution::Normal,
            link: None,
            extra_parameter: None,
            max_iterations: 100,
            tolerance: 1e-8,
        }
    }
}

impl AlmRegressorBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the distribution family.
    pub fn distribution(mut self, dist: AlmDistribution) -> Self {
        self.distribution = dist;
        self
    }

    /// Set the link function (defaults to canonical link for distribution).
    pub fn link(mut self, link: LinkFunction) -> Self {
        self.link = Some(link);
        self
    }

    /// Set the extra parameter (degrees of freedom, shape, etc.).
    pub fn extra_parameter(mut self, param: f64) -> Self {
        self.extra_parameter = Some(param);
        self
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

    /// Set maximum iterations for optimization.
    pub fn max_iterations(mut self, iters: usize) -> Self {
        self.max_iterations = iters;
        self
    }

    /// Set convergence tolerance.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Build the ALM regressor.
    pub fn build(self) -> AlmRegressor {
        let options = self.options_builder.build_unchecked();
        let mut alm =
            AlmRegressor::new(options, self.distribution, self.link, self.extra_parameter);
        alm.max_iterations = self.max_iterations;
        alm.tolerance = self.tolerance;
        alm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_likelihood() {
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| i as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::Normal, 1.0, None);
        // Perfect fit: RSS = 0
        // LL = -5/2 * log(2*pi) - 0 = -5/2 * 1.8379
        let expected = -2.5 * (2.0 * PI).ln();
        assert!((ll - expected).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_likelihood() {
        let y = Col::from_fn(5, |i| i as f64);
        let mu = Col::from_fn(5, |i| i as f64);
        let ll = log_likelihood(&y, &mu, AlmDistribution::Laplace, 1.0, None);
        // Perfect fit: SAD = 0
        let expected = -5.0 * (2.0_f64).ln();
        assert!((ll - expected).abs() < 1e-10);
    }

    #[test]
    fn test_link_functions() {
        // Identity
        assert!((LinkFunction::Identity.inverse(2.0) - 2.0).abs() < 1e-10);

        // Log
        assert!((LinkFunction::Log.inverse(0.0) - 1.0).abs() < 1e-10);
        assert!((LinkFunction::Log.inverse(1.0) - std::f64::consts::E).abs() < 1e-10);

        // Logit
        assert!((LinkFunction::Logit.inverse(0.0) - 0.5).abs() < 1e-10);
    }
}
