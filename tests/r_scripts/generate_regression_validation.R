#!/usr/bin/env Rscript
# Generate validation data for WLS, Ridge, Elastic Net, and Tweedie
# This script creates test cases with known outputs from R

library(glmnet)
library(statmod)

set.seed(42)

cat("// =============================================================================\n")
cat("// Regression Validation Data Generated from R\n")
cat("// Generated on:", format(Sys.time()), "\n")
cat("// R version:", R.version.string, "\n")
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# WLS (Weighted Least Squares)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Weighted Least Squares\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 20
x_wls <- 1:n
y_wls <- 2.5 + 1.8 * x_wls + rnorm(n, sd = 0.5 * x_wls)  # Heteroscedastic
weights_wls <- 1 / (x_wls^2)  # Inverse variance weights

fit_wls <- lm(y_wls ~ x_wls, weights = weights_wls)
summary_wls <- summary(fit_wls)

cat(sprintf("// R Code: lm(y ~ x, weights = w)\n"))
cat(sprintf("const X_WLS: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_wls), collapse=", ")))
cat(sprintf("const Y_WLS: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_wls), collapse=", ")))
cat(sprintf("const W_WLS: [f64; %d] = [%s];\n", n, paste(sprintf("%.10f", weights_wls), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_WLS: f64 = %.10f;\n", coef(fit_wls)[1]))
cat(sprintf("const EXPECTED_COEF_WLS: f64 = %.10f;\n", coef(fit_wls)[2]))
cat(sprintf("const EXPECTED_SE_INTERCEPT_WLS: f64 = %.10f;\n", summary_wls$coefficients[1,2]))
cat(sprintf("const EXPECTED_SE_COEF_WLS: f64 = %.10f;\n", summary_wls$coefficients[2,2]))
cat(sprintf("const EXPECTED_R_SQUARED_WLS: f64 = %.10f;\n", summary_wls$r.squared))
cat("\n")

# -----------------------------------------------------------------------------
# Ridge Regression (glmnet)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Ridge Regression (glmnet)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 50
p <- 3
X_ridge <- matrix(rnorm(n * p), n, p)
beta_true <- c(1.5, -2.0, 0.5)
y_ridge <- X_ridge %*% beta_true + rnorm(n, sd = 0.5)

# glmnet with alpha = 0 (ridge)
# Note: glmnet standardizes by default, we use standardize=FALSE for comparison
lambda <- 0.5
fit_ridge <- glmnet(X_ridge, y_ridge, alpha = 0, lambda = lambda,
                     standardize = FALSE, intercept = TRUE)

cat(sprintf("// R Code: glmnet(X, y, alpha = 0, lambda = %.1f, standardize = FALSE)\n", lambda))
cat(sprintf("const N_RIDGE: usize = %d;\n", n))
cat(sprintf("const P_RIDGE: usize = %d;\n", p))
cat(sprintf("const X_RIDGE: [f64; %d] = [%s];\n", n*p, paste(sprintf("%.6f", as.vector(X_ridge)), collapse=", ")))
cat(sprintf("const Y_RIDGE: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_ridge), collapse=", ")))
cat(sprintf("const LAMBDA_RIDGE: f64 = %.10f;\n", lambda))
cat(sprintf("const EXPECTED_INTERCEPT_RIDGE: f64 = %.10f;\n", as.numeric(fit_ridge$a0)))
cat(sprintf("const EXPECTED_COEFS_RIDGE: [f64; %d] = [%s];\n", p,
            paste(sprintf("%.10f", as.numeric(fit_ridge$beta)), collapse=", ")))
cat("\n")

# -----------------------------------------------------------------------------
# Elastic Net (glmnet)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Elastic Net (glmnet)\n")
cat("// -----------------------------------------------------------------------------\n")

# Use same data as ridge
alpha <- 0.5  # Mix of L1 and L2
lambda_enet <- 0.3
fit_enet <- glmnet(X_ridge, y_ridge, alpha = alpha, lambda = lambda_enet,
                    standardize = FALSE, intercept = TRUE)

cat(sprintf("// R Code: glmnet(X, y, alpha = %.1f, lambda = %.1f, standardize = FALSE)\n", alpha, lambda_enet))
cat(sprintf("const ALPHA_ENET: f64 = %.10f;\n", alpha))
cat(sprintf("const LAMBDA_ENET: f64 = %.10f;\n", lambda_enet))
cat(sprintf("const EXPECTED_INTERCEPT_ENET: f64 = %.10f;\n", as.numeric(fit_enet$a0)))
cat(sprintf("const EXPECTED_COEFS_ENET: [f64; %d] = [%s];\n", p,
            paste(sprintf("%.10f", as.numeric(fit_enet$beta)), collapse=", ")))
cat("\n")

# -----------------------------------------------------------------------------
# Tweedie GLM (Gamma - var.power = 2)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Tweedie GLM - Gamma (var.power = 2)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 30
x_tweedie <- seq(0.5, 3.0, length.out = n)
mu_tweedie <- exp(0.5 + 0.4 * x_tweedie)
set.seed(42)
y_tweedie <- rgamma(n, shape = 2, rate = 2 / mu_tweedie)

fit_tweedie_gamma <- glm(y_tweedie ~ x_tweedie, family = tweedie(var.power = 2, link.power = 0))
summary_tweedie_gamma <- summary(fit_tweedie_gamma)

cat(sprintf("// R Code: glm(y ~ x, family = tweedie(var.power = 2, link.power = 0))\n"))
cat(sprintf("const X_TWEEDIE_GAMMA: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_tweedie), collapse=", ")))
cat(sprintf("const Y_TWEEDIE_GAMMA: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_tweedie), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_TWEEDIE_GAMMA: f64 = %.10f;\n", coef(fit_tweedie_gamma)[1]))
cat(sprintf("const EXPECTED_COEF_TWEEDIE_GAMMA: f64 = %.10f;\n", coef(fit_tweedie_gamma)[2]))
cat(sprintf("const EXPECTED_DEVIANCE_TWEEDIE_GAMMA: f64 = %.10f;\n", deviance(fit_tweedie_gamma)))
cat(sprintf("const EXPECTED_DISPERSION_TWEEDIE_GAMMA: f64 = %.10f;\n", summary_tweedie_gamma$dispersion))
cat("\n")

# -----------------------------------------------------------------------------
# Tweedie GLM - Poisson (var.power = 1)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Tweedie GLM - Poisson (var.power = 1)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 30
x_pois_tw <- seq(0.1, 3.0, length.out = n)
set.seed(42)
lambda_pois <- exp(0.5 + 0.8 * x_pois_tw)
y_pois_tw <- rpois(n, lambda_pois)

fit_tweedie_pois <- glm(y_pois_tw ~ x_pois_tw, family = tweedie(var.power = 1, link.power = 0))
summary_tweedie_pois <- summary(fit_tweedie_pois)

cat(sprintf("// R Code: glm(y ~ x, family = tweedie(var.power = 1, link.power = 0))\n"))
cat(sprintf("const X_TWEEDIE_POISSON: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_pois_tw), collapse=", ")))
cat(sprintf("const Y_TWEEDIE_POISSON: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_pois_tw), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_TWEEDIE_POISSON: f64 = %.10f;\n", coef(fit_tweedie_pois)[1]))
cat(sprintf("const EXPECTED_COEF_TWEEDIE_POISSON: f64 = %.10f;\n", coef(fit_tweedie_pois)[2]))
cat(sprintf("const EXPECTED_DEVIANCE_TWEEDIE_POISSON: f64 = %.10f;\n", deviance(fit_tweedie_pois)))
cat("\n")

# -----------------------------------------------------------------------------
# Tweedie GLM - Inverse Gaussian (var.power = 3)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Tweedie GLM - Inverse Gaussian (var.power = 3)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 30
x_ig_tw <- seq(0.5, 3.0, length.out = n)
mu_ig <- exp(0.3 + 0.5 * x_ig_tw)
set.seed(42)
y_ig_tw <- statmod::rinvgauss(n, mean = mu_ig, shape = 2)

fit_tweedie_ig <- glm(y_ig_tw ~ x_ig_tw, family = tweedie(var.power = 3, link.power = 0))
summary_tweedie_ig <- summary(fit_tweedie_ig)

cat(sprintf("// R Code: glm(y ~ x, family = tweedie(var.power = 3, link.power = 0))\n"))
cat(sprintf("const X_TWEEDIE_INVGAUSS: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_ig_tw), collapse=", ")))
cat(sprintf("const Y_TWEEDIE_INVGAUSS: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_ig_tw), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_TWEEDIE_INVGAUSS: f64 = %.10f;\n", coef(fit_tweedie_ig)[1]))
cat(sprintf("const EXPECTED_COEF_TWEEDIE_INVGAUSS: f64 = %.10f;\n", coef(fit_tweedie_ig)[2]))
cat(sprintf("const EXPECTED_DEVIANCE_TWEEDIE_INVGAUSS: f64 = %.10f;\n", deviance(fit_tweedie_ig)))
cat("\n")

# -----------------------------------------------------------------------------
# Tweedie GLM - Compound Poisson-Gamma (var.power = 1.5)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Tweedie GLM - Compound Poisson-Gamma (var.power = 1.5)\n")
cat("// -----------------------------------------------------------------------------\n")

# Generate CPG-like data (with some zeros)
n <- 30
x_cpg <- seq(0.5, 3.0, length.out = n)
mu_cpg <- exp(0.3 + 0.4 * x_cpg)
set.seed(42)
# Simulate CPG: some zeros, rest positive
y_cpg <- ifelse(runif(n) < 0.3, 0, rgamma(n, shape = 2, rate = 2 / mu_cpg))
# Replace zeros with small positive values for GLM
y_cpg[y_cpg == 0] <- 0.001

fit_tweedie_cpg <- glm(y_cpg ~ x_cpg, family = tweedie(var.power = 1.5, link.power = 0))
summary_tweedie_cpg <- summary(fit_tweedie_cpg)

cat(sprintf("// R Code: glm(y ~ x, family = tweedie(var.power = 1.5, link.power = 0))\n"))
cat(sprintf("const X_TWEEDIE_CPG: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_cpg), collapse=", ")))
cat(sprintf("const Y_TWEEDIE_CPG: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_cpg), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_TWEEDIE_CPG: f64 = %.10f;\n", coef(fit_tweedie_cpg)[1]))
cat(sprintf("const EXPECTED_COEF_TWEEDIE_CPG: f64 = %.10f;\n", coef(fit_tweedie_cpg)[2]))
cat(sprintf("const EXPECTED_DEVIANCE_TWEEDIE_CPG: f64 = %.10f;\n", deviance(fit_tweedie_cpg)))
cat("\n")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat("// =============================================================================\n")
cat("// Summary of expected values\n")
cat("// =============================================================================\n")
cat(sprintf("// WLS:                intercept=%.6f, coef=%.6f, R²=%.6f\n",
            coef(fit_wls)[1], coef(fit_wls)[2], summary_wls$r.squared))
cat(sprintf("// Ridge (λ=%.1f):     intercept=%.6f, coefs=[%.6f, %.6f, %.6f]\n",
            lambda, as.numeric(fit_ridge$a0),
            as.numeric(fit_ridge$beta)[1], as.numeric(fit_ridge$beta)[2], as.numeric(fit_ridge$beta)[3]))
cat(sprintf("// Elastic Net:        intercept=%.6f, coefs=[%.6f, %.6f, %.6f]\n",
            as.numeric(fit_enet$a0),
            as.numeric(fit_enet$beta)[1], as.numeric(fit_enet$beta)[2], as.numeric(fit_enet$beta)[3]))
cat(sprintf("// Tweedie Gamma:      intercept=%.6f, coef=%.6f, deviance=%.6f\n",
            coef(fit_tweedie_gamma)[1], coef(fit_tweedie_gamma)[2], deviance(fit_tweedie_gamma)))
cat(sprintf("// Tweedie Poisson:    intercept=%.6f, coef=%.6f, deviance=%.6f\n",
            coef(fit_tweedie_pois)[1], coef(fit_tweedie_pois)[2], deviance(fit_tweedie_pois)))
cat(sprintf("// Tweedie InvGauss:   intercept=%.6f, coef=%.6f, deviance=%.6f\n",
            coef(fit_tweedie_ig)[1], coef(fit_tweedie_ig)[2], deviance(fit_tweedie_ig)))
cat(sprintf("// Tweedie CPG:        intercept=%.6f, coef=%.6f, deviance=%.6f\n",
            coef(fit_tweedie_cpg)[1], coef(fit_tweedie_cpg)[2], deviance(fit_tweedie_cpg)))
