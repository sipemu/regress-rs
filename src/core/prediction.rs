//! Prediction types for interval estimation.

use faer::Col;

/// Type of interval to compute for predictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntervalType {
    /// Confidence interval for the mean response E[Y|X=x₀].
    /// Narrower - only accounts for uncertainty in coefficient estimates.
    Confidence,

    /// Prediction interval for a new observation Y|X=x₀.
    /// Wider - also accounts for residual variance (irreducible error).
    #[default]
    Prediction,
}

/// Result of prediction with optional intervals.
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Point predictions (fitted values).
    pub fit: Col<f64>,
    /// Lower bounds of the interval.
    pub lower: Col<f64>,
    /// Upper bounds of the interval.
    pub upper: Col<f64>,
    /// Standard errors of predictions.
    pub se: Col<f64>,
}

impl PredictionResult {
    /// Create a new prediction result with only point predictions (no intervals).
    pub fn point_only(fit: Col<f64>) -> Self {
        let n = fit.nrows();
        Self {
            fit,
            lower: Col::zeros(n),
            upper: Col::zeros(n),
            se: Col::zeros(n),
        }
    }

    /// Create a new prediction result with intervals.
    pub fn with_intervals(fit: Col<f64>, lower: Col<f64>, upper: Col<f64>, se: Col<f64>) -> Self {
        Self {
            fit,
            lower,
            upper,
            se,
        }
    }

    /// Number of predictions.
    pub fn len(&self) -> usize {
        self.fit.nrows()
    }

    /// Returns true if there are no predictions.
    pub fn is_empty(&self) -> bool {
        self.fit.nrows() == 0
    }
}
