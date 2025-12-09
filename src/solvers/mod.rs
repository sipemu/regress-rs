//! Regression solvers implementing various estimation methods.

pub mod alm;
mod binomial;
mod bls;
mod elastic_net;
mod negative_binomial;
mod ols;
mod poisson;
mod ridge;
mod rls;
mod traits;
mod tweedie;
mod wls;

pub use alm::{AlmDistribution, AlmRegressor, AlmRegressorBuilder, FittedAlm, LinkFunction};
pub use binomial::{BinomialRegressor, FittedBinomial};
pub use bls::{BlsRegressor, FittedBls};
pub use elastic_net::{ElasticNetRegressor, FittedElasticNet};
pub use negative_binomial::{FittedNegativeBinomial, NegativeBinomialRegressor};
pub use ols::{FittedOls, OlsRegressor};
pub use poisson::{FittedPoisson, PoissonRegressor};
pub use ridge::{FittedRidge, RidgeRegressor};
pub use rls::{FittedRls, RlsRegressor};
pub use traits::{FittedRegressor, RegressionError, Regressor};
pub use tweedie::{FittedTweedie, TweedieRegressor};
pub use wls::{FittedWls, WlsRegressor};
