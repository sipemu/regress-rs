//! Regression solvers implementing various estimation methods.

mod traits;
mod ols;
mod ridge;
mod elastic_net;
mod wls;
mod rls;

pub use traits::{FittedRegressor, Regressor, RegressionError};
pub use ols::{FittedOls, OlsRegressor};
pub use ridge::{FittedRidge, RidgeRegressor};
pub use elastic_net::{ElasticNetRegressor, FittedElasticNet};
pub use wls::{FittedWls, WlsRegressor};
pub use rls::{FittedRls, RlsRegressor};
