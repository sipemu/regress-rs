//! Regression solvers implementing various estimation methods.

mod bls;
mod elastic_net;
mod ols;
mod ridge;
mod rls;
mod traits;
mod tweedie;
mod wls;

pub use bls::{BlsRegressor, FittedBls};
pub use elastic_net::{ElasticNetRegressor, FittedElasticNet};
pub use ols::{FittedOls, OlsRegressor};
pub use ridge::{FittedRidge, RidgeRegressor};
pub use rls::{FittedRls, RlsRegressor};
pub use traits::{FittedRegressor, RegressionError, Regressor};
pub use tweedie::{FittedTweedie, TweedieRegressor};
pub use wls::{FittedWls, WlsRegressor};
