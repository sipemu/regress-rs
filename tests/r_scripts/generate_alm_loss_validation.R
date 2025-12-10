#!/usr/bin/env Rscript
# Generate validation data for ALM loss functions
# This script creates test cases with known outputs from R's greybox package

library(greybox)

# Set seed for reproducibility
set.seed(42)

cat("// =============================================================================\n")
cat("// ALM Loss Function Validation Data Generated from R greybox package\n")
cat("// Generated on:", format(Sys.time()), "\n")
cat("// greybox version:", as.character(packageVersion("greybox")), "\n")
cat("// =============================================================================\n\n")

# -----------------------------------------------------------------------------
# Generate test data
# -----------------------------------------------------------------------------
n <- 50
x <- seq(1, 50, length.out = n)
y_clean <- 2.5 + 1.5 * x + rnorm(n, sd = 3)

# Data with outliers for robust loss testing
y_outliers <- y_clean
y_outliers[10] <- y_outliers[10] + 80   # Large positive outlier
y_outliers[40] <- y_outliers[40] - 80   # Large negative outlier

cat("// Test data dimensions: n =", n, "\n")
cat(sprintf("const X_DATA: [f64; %d] = [%s];\n\n", n, paste(sprintf("%.6f", x), collapse=", ")))
cat(sprintf("const Y_CLEAN: [f64; %d] = [%s];\n\n", n, paste(sprintf("%.6f", y_clean), collapse=", ")))
cat(sprintf("const Y_OUTLIERS: [f64; %d] = [%s];\n\n", n, paste(sprintf("%.6f", y_outliers), collapse=", ")))

# -----------------------------------------------------------------------------
# Test Case 1: Likelihood loss (default)
# -----------------------------------------------------------------------------
cat("// =============================================================================\n")
cat("// Test Case 1: Likelihood loss (default)\n")
cat("// =============================================================================\n")

model_ll <- alm(y_clean ~ x, distribution = "dnorm", loss = "likelihood")

cat(sprintf("// Intercept: %.10f\n", coef(model_ll)[1]))
cat(sprintf("// Coefficient: %.10f\n", coef(model_ll)[2]))
cat(sprintf("// Scale: %.10f\n", model_ll$scale))
cat(sprintf("// Log-likelihood: %.10f\n", as.numeric(logLik(model_ll))))

cat("\n// Rust constants:\n")
cat(sprintf("const EXPECTED_INTERCEPT_LL: f64 = %.10f;\n", coef(model_ll)[1]))
cat(sprintf("const EXPECTED_COEF_LL: f64 = %.10f;\n", coef(model_ll)[2]))
cat(sprintf("const EXPECTED_SCALE_LL: f64 = %.10f;\n", model_ll$scale))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 2: MSE loss
# -----------------------------------------------------------------------------
cat("// =============================================================================\n")
cat("// Test Case 2: MSE loss\n")
cat("// =============================================================================\n")

model_mse <- alm(y_clean ~ x, distribution = "dnorm", loss = "MSE")

cat(sprintf("// Intercept: %.10f\n", coef(model_mse)[1]))
cat(sprintf("// Coefficient: %.10f\n", coef(model_mse)[2]))
cat(sprintf("// Scale: %.10f\n", model_mse$scale))

cat("\n// Rust constants:\n")
cat(sprintf("const EXPECTED_INTERCEPT_MSE: f64 = %.10f;\n", coef(model_mse)[1]))
cat(sprintf("const EXPECTED_COEF_MSE: f64 = %.10f;\n", coef(model_mse)[2]))
cat(sprintf("const EXPECTED_SCALE_MSE: f64 = %.10f;\n", model_mse$scale))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 3: MAE loss (robust to outliers)
# -----------------------------------------------------------------------------
cat("// =============================================================================\n")
cat("// Test Case 3: MAE loss on clean data\n")
cat("// =============================================================================\n")

model_mae_clean <- alm(y_clean ~ x, distribution = "dnorm", loss = "MAE")

cat(sprintf("// Intercept: %.10f\n", coef(model_mae_clean)[1]))
cat(sprintf("// Coefficient: %.10f\n", coef(model_mae_clean)[2]))
cat(sprintf("// Scale: %.10f\n", model_mae_clean$scale))

cat("\n// Rust constants:\n")
cat(sprintf("const EXPECTED_INTERCEPT_MAE_CLEAN: f64 = %.10f;\n", coef(model_mae_clean)[1]))
cat(sprintf("const EXPECTED_COEF_MAE_CLEAN: f64 = %.10f;\n", coef(model_mae_clean)[2]))
cat(sprintf("const EXPECTED_SCALE_MAE_CLEAN: f64 = %.10f;\n", model_mae_clean$scale))
cat("\n")

# MAE on data with outliers
cat("// =============================================================================\n")
cat("// Test Case 4: MAE loss on data with outliers\n")
cat("// =============================================================================\n")

model_mae_outliers <- alm(y_outliers ~ x, distribution = "dnorm", loss = "MAE")

cat(sprintf("// Intercept: %.10f\n", coef(model_mae_outliers)[1]))
cat(sprintf("// Coefficient: %.10f\n", coef(model_mae_outliers)[2]))
cat(sprintf("// Scale: %.10f\n", model_mae_outliers$scale))

cat("\n// Rust constants:\n")
cat(sprintf("const EXPECTED_INTERCEPT_MAE_OUTLIERS: f64 = %.10f;\n", coef(model_mae_outliers)[1]))
cat(sprintf("const EXPECTED_COEF_MAE_OUTLIERS: f64 = %.10f;\n", coef(model_mae_outliers)[2]))
cat(sprintf("const EXPECTED_SCALE_MAE_OUTLIERS: f64 = %.10f;\n", model_mae_outliers$scale))
cat("\n")

# Compare MSE on outlier data (should be affected more)
model_mse_outliers <- alm(y_outliers ~ x, distribution = "dnorm", loss = "MSE")

cat("// =============================================================================\n")
cat("// Test Case 5: MSE loss on data with outliers (for comparison)\n")
cat("// =============================================================================\n")

cat(sprintf("// Intercept: %.10f\n", coef(model_mse_outliers)[1]))
cat(sprintf("// Coefficient: %.10f\n", coef(model_mse_outliers)[2]))
cat(sprintf("// Scale: %.10f\n", model_mse_outliers$scale))

cat("\n// Rust constants:\n")
cat(sprintf("const EXPECTED_INTERCEPT_MSE_OUTLIERS: f64 = %.10f;\n", coef(model_mse_outliers)[1]))
cat(sprintf("const EXPECTED_COEF_MSE_OUTLIERS: f64 = %.10f;\n", coef(model_mse_outliers)[2]))
cat(sprintf("const EXPECTED_SCALE_MSE_OUTLIERS: f64 = %.10f;\n", model_mse_outliers$scale))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 6: HAM loss (Half Absolute Moment)
# -----------------------------------------------------------------------------
cat("// =============================================================================\n")
cat("// Test Case 6: HAM loss (Half Absolute Moment)\n")
cat("// =============================================================================\n")

model_ham <- alm(y_clean ~ x, distribution = "dnorm", loss = "HAM")

cat(sprintf("// Intercept: %.10f\n", coef(model_ham)[1]))
cat(sprintf("// Coefficient: %.10f\n", coef(model_ham)[2]))
cat(sprintf("// Scale: %.10f\n", model_ham$scale))

cat("\n// Rust constants:\n")
cat(sprintf("const EXPECTED_INTERCEPT_HAM: f64 = %.10f;\n", coef(model_ham)[1]))
cat(sprintf("const EXPECTED_COEF_HAM: f64 = %.10f;\n", coef(model_ham)[2]))
cat(sprintf("const EXPECTED_SCALE_HAM: f64 = %.10f;\n", model_ham$scale))
cat("\n")

# HAM on data with outliers
cat("// =============================================================================\n")
cat("// Test Case 7: HAM loss on data with outliers\n")
cat("// =============================================================================\n")

model_ham_outliers <- alm(y_outliers ~ x, distribution = "dnorm", loss = "HAM")

cat(sprintf("// Intercept: %.10f\n", coef(model_ham_outliers)[1]))
cat(sprintf("// Coefficient: %.10f\n", coef(model_ham_outliers)[2]))
cat(sprintf("// Scale: %.10f\n", model_ham_outliers$scale))

cat("\n// Rust constants:\n")
cat(sprintf("const EXPECTED_INTERCEPT_HAM_OUTLIERS: f64 = %.10f;\n", coef(model_ham_outliers)[1]))
cat(sprintf("const EXPECTED_COEF_HAM_OUTLIERS: f64 = %.10f;\n", coef(model_ham_outliers)[2]))
cat(sprintf("const EXPECTED_SCALE_HAM_OUTLIERS: f64 = %.10f;\n", model_ham_outliers$scale))
cat("\n")

# -----------------------------------------------------------------------------
# Test Case 8: ROLE loss (RObust Likelihood Estimator)
# -----------------------------------------------------------------------------
cat("// =============================================================================\n")
cat("// Test Case 8: ROLE loss with default trim (0.05)\n")
cat("// =============================================================================\n")

model_role <- alm(y_outliers ~ x, distribution = "dnorm", loss = "ROLE")

cat(sprintf("// Intercept: %.10f\n", coef(model_role)[1]))
cat(sprintf("// Coefficient: %.10f\n", coef(model_role)[2]))
cat(sprintf("// Scale: %.10f\n", model_role$scale))

cat("\n// Rust constants:\n")
cat(sprintf("const EXPECTED_INTERCEPT_ROLE: f64 = %.10f;\n", coef(model_role)[1]))
cat(sprintf("const EXPECTED_COEF_ROLE: f64 = %.10f;\n", coef(model_role)[2]))
cat(sprintf("const EXPECTED_SCALE_ROLE: f64 = %.10f;\n", model_role$scale))
cat("\n")

# ROLE with custom trim
cat("// =============================================================================\n")
cat("// Test Case 9: ROLE loss with trim = 0.10\n")
cat("// =============================================================================\n")

model_role_10 <- alm(y_outliers ~ x, distribution = "dnorm", loss = "ROLE", trim = 0.10)

cat(sprintf("// Intercept: %.10f\n", coef(model_role_10)[1]))
cat(sprintf("// Coefficient: %.10f\n", coef(model_role_10)[2]))
cat(sprintf("// Scale: %.10f\n", model_role_10$scale))

cat("\n// Rust constants:\n")
cat(sprintf("const EXPECTED_INTERCEPT_ROLE_10: f64 = %.10f;\n", coef(model_role_10)[1]))
cat(sprintf("const EXPECTED_COEF_ROLE_10: f64 = %.10f;\n", coef(model_role_10)[2]))
cat(sprintf("const EXPECTED_SCALE_ROLE_10: f64 = %.10f;\n", model_role_10$scale))
cat("\n")

# -----------------------------------------------------------------------------
# Summary comparison
# -----------------------------------------------------------------------------
cat("// =============================================================================\n")
cat("// Summary: Coefficient comparison across loss functions (data with outliers)\n")
cat("// True values: intercept = 2.5, coefficient = 1.5\n")
cat("// =============================================================================\n")

cat(sprintf("// MSE:  intercept=%.4f (err=%.4f), coef=%.4f (err=%.4f)\n",
            coef(model_mse_outliers)[1], abs(coef(model_mse_outliers)[1] - 2.5),
            coef(model_mse_outliers)[2], abs(coef(model_mse_outliers)[2] - 1.5)))
cat(sprintf("// MAE:  intercept=%.4f (err=%.4f), coef=%.4f (err=%.4f)\n",
            coef(model_mae_outliers)[1], abs(coef(model_mae_outliers)[1] - 2.5),
            coef(model_mae_outliers)[2], abs(coef(model_mae_outliers)[2] - 1.5)))
cat(sprintf("// HAM:  intercept=%.4f (err=%.4f), coef=%.4f (err=%.4f)\n",
            coef(model_ham_outliers)[1], abs(coef(model_ham_outliers)[1] - 2.5),
            coef(model_ham_outliers)[2], abs(coef(model_ham_outliers)[2] - 1.5)))
cat(sprintf("// ROLE: intercept=%.4f (err=%.4f), coef=%.4f (err=%.4f)\n",
            coef(model_role)[1], abs(coef(model_role)[1] - 2.5),
            coef(model_role)[2], abs(coef(model_role)[2] - 1.5)))
cat("\n")
cat("// Note: Robust loss functions (MAE, HAM, ROLE) should have smaller errors\n")
cat("//       when outliers are present compared to MSE.\n")
