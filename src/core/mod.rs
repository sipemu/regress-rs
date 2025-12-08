//! Core types for regression analysis.

mod family;
mod na_action;
mod options;
mod prediction;
mod result;

pub use family::TweedieFamily;
pub use na_action::{NaAction, NaError, NaHandler, NaInfo, NaResult};
pub use options::{
    LambdaScaling, OptionsError, RegressionOptions, RegressionOptionsBuilder, SolverType,
};
pub use prediction::{IntervalType, PredictionResult};
pub use result::RegressionResult;
