#!/usr/bin/env Rscript
# Generate validation data for GLM implementations
# This script creates test cases with known outputs from R's glm() function

library(MASS)  # for negative binomial

# Set seed for reproducibility
set.seed(42)

cat("// =============================================================================\n")
cat("// GLM Validation Data Generated from R glm() function\n")
cat("// Generated on:", format(Sys.time()), "\n")
cat("// R version:", R.version.string, "\n")
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Test Case 1: Poisson GLM with log link
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Poisson GLM with log link\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 30
x_pois <- seq(0.1, 3.0, length.out = n)
# Generate Poisson data: lambda = exp(0.5 + 0.8 * x)
set.seed(42)
lambda <- exp(0.5 + 0.8 * x_pois)
y_pois <- rpois(n, lambda)

fit_pois <- glm(y_pois ~ x_pois, family = poisson(link = "log"))
summary_pois <- summary(fit_pois)

cat(sprintf("// R Code: glm(y ~ x, family = poisson(link = \"log\"))\n"))
cat(sprintf("const X_POISSON_LOG: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_pois), collapse=", ")))
cat(sprintf("const Y_POISSON_LOG: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_pois), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_POISSON_LOG: f64 = %.10f;\n", coef(fit_pois)[1]))
cat(sprintf("const EXPECTED_COEF_POISSON_LOG: f64 = %.10f;\n", coef(fit_pois)[2]))
cat(sprintf("const EXPECTED_DEVIANCE_POISSON_LOG: f64 = %.10f;\n", deviance(fit_pois)))
cat(sprintf("const EXPECTED_NULL_DEVIANCE_POISSON_LOG: f64 = %.10f;\n", fit_pois$null.deviance))
cat(sprintf("const EXPECTED_AIC_POISSON_LOG: f64 = %.10f;\n", AIC(fit_pois)))
cat(sprintf("const EXPECTED_SE_INTERCEPT_POISSON_LOG: f64 = %.10f;\n", summary_pois$coefficients[1,2]))
cat(sprintf("const EXPECTED_SE_COEF_POISSON_LOG: f64 = %.10f;\n", summary_pois$coefficients[2,2]))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 2: Poisson GLM with identity link
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Poisson GLM with identity link\n")
cat("// -----------------------------------------------------------------------------\n")

x_pois_id <- 1:20
y_pois_id <- 2 + 3 * x_pois_id + rpois(20, 2)  # Linear with Poisson noise

fit_pois_id <- glm(y_pois_id ~ x_pois_id, family = poisson(link = "identity"))
summary_pois_id <- summary(fit_pois_id)

cat(sprintf("// R Code: glm(y ~ x, family = poisson(link = \"identity\"))\n"))
cat(sprintf("const X_POISSON_IDENTITY: [f64; %d] = [%s];\n", length(x_pois_id), paste(sprintf("%.6f", x_pois_id), collapse=", ")))
cat(sprintf("const Y_POISSON_IDENTITY: [f64; %d] = [%s];\n", length(y_pois_id), paste(sprintf("%.6f", y_pois_id), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_POISSON_IDENTITY: f64 = %.10f;\n", coef(fit_pois_id)[1]))
cat(sprintf("const EXPECTED_COEF_POISSON_IDENTITY: f64 = %.10f;\n", coef(fit_pois_id)[2]))
cat(sprintf("const EXPECTED_DEVIANCE_POISSON_IDENTITY: f64 = %.10f;\n", deviance(fit_pois_id)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 3: Binomial GLM with logit link (logistic regression)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Binomial GLM with logit link (logistic regression)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 30
x_binom <- seq(-3, 3, length.out = n)
# Generate binary data: P(Y=1) = logistic(0.5 + 1.2 * x)
prob <- 1 / (1 + exp(-(0.5 + 1.2 * x_binom)))
set.seed(42)
y_binom <- rbinom(n, 1, prob)

fit_binom <- glm(y_binom ~ x_binom, family = binomial(link = "logit"))
summary_binom <- summary(fit_binom)

cat(sprintf("// R Code: glm(y ~ x, family = binomial(link = \"logit\"))\n"))
cat(sprintf("const X_BINOMIAL_LOGIT: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_binom), collapse=", ")))
cat(sprintf("const Y_BINOMIAL_LOGIT: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_binom), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_BINOMIAL_LOGIT: f64 = %.10f;\n", coef(fit_binom)[1]))
cat(sprintf("const EXPECTED_COEF_BINOMIAL_LOGIT: f64 = %.10f;\n", coef(fit_binom)[2]))
cat(sprintf("const EXPECTED_DEVIANCE_BINOMIAL_LOGIT: f64 = %.10f;\n", deviance(fit_binom)))
cat(sprintf("const EXPECTED_NULL_DEVIANCE_BINOMIAL_LOGIT: f64 = %.10f;\n", fit_binom$null.deviance))
cat(sprintf("const EXPECTED_AIC_BINOMIAL_LOGIT: f64 = %.10f;\n", AIC(fit_binom)))
cat(sprintf("const EXPECTED_SE_INTERCEPT_BINOMIAL_LOGIT: f64 = %.10f;\n", summary_binom$coefficients[1,2]))
cat(sprintf("const EXPECTED_SE_COEF_BINOMIAL_LOGIT: f64 = %.10f;\n", summary_binom$coefficients[2,2]))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 4: Binomial GLM with probit link
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Binomial GLM with probit link\n")
cat("// -----------------------------------------------------------------------------\n")

# Use same x, generate different y for probit
prob_probit <- pnorm(0.3 + 0.8 * x_binom)
set.seed(43)
y_probit <- rbinom(n, 1, prob_probit)

fit_probit <- glm(y_probit ~ x_binom, family = binomial(link = "probit"))
summary_probit <- summary(fit_probit)

cat(sprintf("// R Code: glm(y ~ x, family = binomial(link = \"probit\"))\n"))
cat(sprintf("const Y_BINOMIAL_PROBIT: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_probit), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_BINOMIAL_PROBIT: f64 = %.10f;\n", coef(fit_probit)[1]))
cat(sprintf("const EXPECTED_COEF_BINOMIAL_PROBIT: f64 = %.10f;\n", coef(fit_probit)[2]))
cat(sprintf("const EXPECTED_DEVIANCE_BINOMIAL_PROBIT: f64 = %.10f;\n", deviance(fit_probit)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 5: Negative Binomial GLM
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Negative Binomial GLM\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 30
x_nb <- seq(0.5, 3.0, length.out = n)
# Generate overdispersed count data
mu_nb <- exp(0.3 + 0.6 * x_nb)
set.seed(42)
y_nb <- rnbinom(n, size = 2, mu = mu_nb)

fit_nb <- glm.nb(y_nb ~ x_nb)
summary_nb <- summary(fit_nb)

cat(sprintf("// R Code: glm.nb(y ~ x)  # from MASS package\n"))
cat(sprintf("const X_NEGBINOM: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_nb), collapse=", ")))
cat(sprintf("const Y_NEGBINOM: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_nb), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_NEGBINOM: f64 = %.10f;\n", coef(fit_nb)[1]))
cat(sprintf("const EXPECTED_COEF_NEGBINOM: f64 = %.10f;\n", coef(fit_nb)[2]))
cat(sprintf("const EXPECTED_THETA_NEGBINOM: f64 = %.10f;\n", fit_nb$theta))
cat(sprintf("const EXPECTED_DEVIANCE_NEGBINOM: f64 = %.10f;\n", deviance(fit_nb)))
cat(sprintf("const EXPECTED_AIC_NEGBINOM: f64 = %.10f;\n", AIC(fit_nb)))
cat(sprintf("const EXPECTED_SE_INTERCEPT_NEGBINOM: f64 = %.10f;\n", summary_nb$coefficients[1,2]))
cat(sprintf("const EXPECTED_SE_COEF_NEGBINOM: f64 = %.10f;\n", summary_nb$coefficients[2,2]))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 6: Gamma GLM with log link
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Gamma GLM with log link\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 30
x_gamma <- seq(0.5, 3.0, length.out = n)
# Generate Gamma data: mean = exp(0.5 + 0.4 * x), shape = 2
mu_gamma <- exp(0.5 + 0.4 * x_gamma)
set.seed(42)
y_gamma <- rgamma(n, shape = 2, rate = 2 / mu_gamma)

fit_gamma <- glm(y_gamma ~ x_gamma, family = Gamma(link = "log"))
summary_gamma <- summary(fit_gamma)

cat(sprintf("// R Code: glm(y ~ x, family = Gamma(link = \"log\"))\n"))
cat(sprintf("const X_GAMMA_LOG: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_gamma), collapse=", ")))
cat(sprintf("const Y_GAMMA_LOG: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_gamma), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_GAMMA_LOG: f64 = %.10f;\n", coef(fit_gamma)[1]))
cat(sprintf("const EXPECTED_COEF_GAMMA_LOG: f64 = %.10f;\n", coef(fit_gamma)[2]))
cat(sprintf("const EXPECTED_DEVIANCE_GAMMA_LOG: f64 = %.10f;\n", deviance(fit_gamma)))
cat(sprintf("const EXPECTED_DISPERSION_GAMMA: f64 = %.10f;\n", summary_gamma$dispersion))
cat(sprintf("const EXPECTED_SE_INTERCEPT_GAMMA_LOG: f64 = %.10f;\n", summary_gamma$coefficients[1,2]))
cat(sprintf("const EXPECTED_SE_COEF_GAMMA_LOG: f64 = %.10f;\n", summary_gamma$coefficients[2,2]))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 7: Inverse Gaussian GLM with log link
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Inverse Gaussian GLM with log link\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 30
x_ig <- seq(0.5, 3.0, length.out = n)
# Generate Inverse Gaussian data
mu_ig <- exp(0.3 + 0.5 * x_ig)
set.seed(42)
y_ig <- statmod::rinvgauss(n, mean = mu_ig, shape = 2)

fit_ig <- glm(y_ig ~ x_ig, family = inverse.gaussian(link = "log"))
summary_ig <- summary(fit_ig)

cat(sprintf("// R Code: glm(y ~ x, family = inverse.gaussian(link = \"log\"))\n"))
cat(sprintf("const X_INVGAUSS_LOG: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_ig), collapse=", ")))
cat(sprintf("const Y_INVGAUSS_LOG: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_ig), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_INVGAUSS_LOG: f64 = %.10f;\n", coef(fit_ig)[1]))
cat(sprintf("const EXPECTED_COEF_INVGAUSS_LOG: f64 = %.10f;\n", coef(fit_ig)[2]))
cat(sprintf("const EXPECTED_DEVIANCE_INVGAUSS_LOG: f64 = %.10f;\n", deviance(fit_ig)))
cat(sprintf("const EXPECTED_DISPERSION_INVGAUSS: f64 = %.10f;\n", summary_ig$dispersion))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 8: Poisson GLM with offset (exposure)
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Poisson GLM with offset (exposure modeling)\n")
cat("// -----------------------------------------------------------------------------\n")

n <- 20
x_offset <- 1:n
exposure <- rep(c(10, 20, 30, 40), 5)  # varying exposure
# Rate = exp(0.2 + 0.1 * x), count = rate * exposure
rate <- exp(0.2 + 0.1 * x_offset)
set.seed(42)
y_offset <- rpois(n, rate * exposure)

fit_offset <- glm(y_offset ~ x_offset + offset(log(exposure)), family = poisson(link = "log"))
summary_offset <- summary(fit_offset)

cat(sprintf("// R Code: glm(y ~ x + offset(log(exposure)), family = poisson(link = \"log\"))\n"))
cat(sprintf("const X_POISSON_OFFSET: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", x_offset), collapse=", ")))
cat(sprintf("const Y_POISSON_OFFSET: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_offset), collapse=", ")))
cat(sprintf("const EXPOSURE_POISSON_OFFSET: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", exposure), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_POISSON_OFFSET: f64 = %.10f;\n", coef(fit_offset)[1]))
cat(sprintf("const EXPECTED_COEF_POISSON_OFFSET: f64 = %.10f;\n", coef(fit_offset)[2]))
cat(sprintf("const EXPECTED_DEVIANCE_POISSON_OFFSET: f64 = %.10f;\n", deviance(fit_offset)))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 9: Binomial GLM with cloglog link
# -----------------------------------------------------------------------------
cat("// -----------------------------------------------------------------------------\n")
cat("// Binomial GLM with complementary log-log link\n")
cat("// -----------------------------------------------------------------------------\n")

# cloglog: P = 1 - exp(-exp(eta))
n <- 30  # Reset n to match x_binom length
prob_cloglog <- 1 - exp(-exp(-0.5 + 0.6 * x_binom))
set.seed(44)
y_cloglog <- rbinom(n, 1, prob_cloglog)

fit_cloglog <- glm(y_cloglog ~ x_binom, family = binomial(link = "cloglog"))
summary_cloglog <- summary(fit_cloglog)

cat(sprintf("// R Code: glm(y ~ x, family = binomial(link = \"cloglog\"))\n"))
cat(sprintf("const Y_BINOMIAL_CLOGLOG: [f64; %d] = [%s];\n", n, paste(sprintf("%.6f", y_cloglog), collapse=", ")))
cat(sprintf("const EXPECTED_INTERCEPT_BINOMIAL_CLOGLOG: f64 = %.10f;\n", coef(fit_cloglog)[1]))
cat(sprintf("const EXPECTED_COEF_BINOMIAL_CLOGLOG: f64 = %.10f;\n", coef(fit_cloglog)[2]))
cat(sprintf("const EXPECTED_DEVIANCE_BINOMIAL_CLOGLOG: f64 = %.10f;\n", deviance(fit_cloglog)))
cat("\n")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat("// =============================================================================\n")
cat("// Summary of expected values\n")
cat("// =============================================================================\n")
cat(sprintf("// Poisson (log):      intercept=%.6f, coef=%.6f, deviance=%.6f\n",
            coef(fit_pois)[1], coef(fit_pois)[2], deviance(fit_pois)))
cat(sprintf("// Poisson (identity): intercept=%.6f, coef=%.6f, deviance=%.6f\n",
            coef(fit_pois_id)[1], coef(fit_pois_id)[2], deviance(fit_pois_id)))
cat(sprintf("// Binomial (logit):   intercept=%.6f, coef=%.6f, deviance=%.6f\n",
            coef(fit_binom)[1], coef(fit_binom)[2], deviance(fit_binom)))
cat(sprintf("// Binomial (probit):  intercept=%.6f, coef=%.6f, deviance=%.6f\n",
            coef(fit_probit)[1], coef(fit_probit)[2], deviance(fit_probit)))
cat(sprintf("// NegBinom:           intercept=%.6f, coef=%.6f, theta=%.6f\n",
            coef(fit_nb)[1], coef(fit_nb)[2], fit_nb$theta))
cat(sprintf("// Gamma (log):        intercept=%.6f, coef=%.6f, deviance=%.6f\n",
            coef(fit_gamma)[1], coef(fit_gamma)[2], deviance(fit_gamma)))
cat(sprintf("// InvGauss (log):     intercept=%.6f, coef=%.6f, deviance=%.6f\n",
            coef(fit_ig)[1], coef(fit_ig)[2], deviance(fit_ig)))
cat(sprintf("// Poisson (offset):   intercept=%.6f, coef=%.6f, deviance=%.6f\n",
            coef(fit_offset)[1], coef(fit_offset)[2], deviance(fit_offset)))
cat(sprintf("// Binomial (cloglog): intercept=%.6f, coef=%.6f, deviance=%.6f\n",
            coef(fit_cloglog)[1], coef(fit_cloglog)[2], deviance(fit_cloglog)))
