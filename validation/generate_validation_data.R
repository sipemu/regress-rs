#!/usr/bin/env Rscript
# Validation data generation for statistics library
# This script generates test cases with known R results

# Install required packages if not present
required_packages <- c("glmnet", "MASS")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# Set seed for reproducibility
set.seed(42)

cat("=" , rep("=", 70), "\n", sep = "")
cat("VALIDATION DATA FOR RUST STATISTICS LIBRARY\n")
cat("=" , rep("=", 70), "\n", sep = "")

# ============================================================================
# Dataset 1: Simple Linear Regression (OLS)
# ============================================================================
cat("\n### DATASET 1: Simple Linear Regression ###\n")

n1 <- 20
x1 <- matrix(1:n1, ncol = 1)
y1 <- 2.5 + 3.0 * x1[,1] + rnorm(n1, sd = 0.5)

# Print data
cat("\n# X data (column-major for Rust):\n")
cat("X1 = [", paste(sprintf("%.10f", x1), collapse = ", "), "]\n")
cat("\n# Y data:\n")
cat("Y1 = [", paste(sprintf("%.10f", y1), collapse = ", "), "]\n")

# Fit OLS
ols1 <- lm(y1 ~ x1)
cat("\n# OLS Results:\n")
cat(sprintf("intercept = %.15f\n", coef(ols1)[1]))
cat(sprintf("coefficient = %.15f\n", coef(ols1)[2]))
cat(sprintf("r_squared = %.15f\n", summary(ols1)$r.squared))
cat(sprintf("adj_r_squared = %.15f\n", summary(ols1)$adj.r.squared))
cat(sprintf("residual_std_error = %.15f\n", summary(ols1)$sigma))

# Standard errors
cat(sprintf("se_intercept = %.15f\n", summary(ols1)$coefficients[1, 2]))
cat(sprintf("se_coefficient = %.15f\n", summary(ols1)$coefficients[2, 2]))

# t-statistics
cat(sprintf("t_intercept = %.15f\n", summary(ols1)$coefficients[1, 3]))
cat(sprintf("t_coefficient = %.15f\n", summary(ols1)$coefficients[2, 3]))

# p-values
cat(sprintf("p_intercept = %.15e\n", summary(ols1)$coefficients[1, 4]))
cat(sprintf("p_coefficient = %.15e\n", summary(ols1)$coefficients[2, 4]))

# F-statistic
f_stat <- summary(ols1)$fstatistic
cat(sprintf("f_statistic = %.15f\n", f_stat[1]))

# Log-likelihood, AIC, BIC
cat(sprintf("log_likelihood = %.15f\n", as.numeric(logLik(ols1))))
cat(sprintf("aic = %.15f\n", AIC(ols1)))
cat(sprintf("bic = %.15f\n", BIC(ols1)))

# Residuals (first 5)
cat("residuals_first5 = [", paste(sprintf("%.15f", residuals(ols1)[1:5]), collapse = ", "), "]\n")

# ============================================================================
# Dataset 2: Multiple Regression (OLS)
# ============================================================================
cat("\n### DATASET 2: Multiple Regression ###\n")

n2 <- 50
x2_1 <- seq(0, 10, length.out = n2)
x2_2 <- sin(x2_1) * 5
x2 <- cbind(x2_1, x2_2)
y2 <- 1.0 + 2.0 * x2_1 + 3.0 * x2_2 + rnorm(n2, sd = 1.0)

cat("\n# X data (row-major):\n")
cat("n2 =", n2, "\n")
cat("X2_col1 = [", paste(sprintf("%.10f", x2[,1]), collapse = ", "), "]\n")
cat("X2_col2 = [", paste(sprintf("%.10f", x2[,2]), collapse = ", "), "]\n")
cat("\n# Y data:\n")
cat("Y2 = [", paste(sprintf("%.10f", y2), collapse = ", "), "]\n")

ols2 <- lm(y2 ~ x2)
cat("\n# OLS Results:\n")
cat(sprintf("intercept = %.15f\n", coef(ols2)[1]))
cat(sprintf("coef1 = %.15f\n", coef(ols2)[2]))
cat(sprintf("coef2 = %.15f\n", coef(ols2)[3]))
cat(sprintf("r_squared = %.15f\n", summary(ols2)$r.squared))
cat(sprintf("adj_r_squared = %.15f\n", summary(ols2)$adj.r.squared))

# Standard errors
cat(sprintf("se_intercept = %.15f\n", summary(ols2)$coefficients[1, 2]))
cat(sprintf("se_coef1 = %.15f\n", summary(ols2)$coefficients[2, 2]))
cat(sprintf("se_coef2 = %.15f\n", summary(ols2)$coefficients[3, 2]))

# ============================================================================
# Dataset 3: Ridge Regression
# ============================================================================
cat("\n### DATASET 3: Ridge Regression ###\n")

# Use same data as Dataset 2 but with ridge
library(glmnet)

# glmnet expects standardized data by default, we'll use standardize=FALSE
# to match our implementation
x3 <- scale(x2, center = TRUE, scale = FALSE)
y3 <- y2 - mean(y2)

# Ridge with specific lambda values
lambdas <- c(0.0, 0.1, 1.0, 10.0)

for (lam in lambdas) {
  cat(sprintf("\n# Ridge lambda = %.1f:\n", lam))

  if (lam == 0) {
    # For lambda=0, use OLS
    ridge_fit <- lm(y2 ~ x2)
    cat(sprintf("intercept = %.15f\n", coef(ridge_fit)[1]))
    cat(sprintf("coef1 = %.15f\n", coef(ridge_fit)[2]))
    cat(sprintf("coef2 = %.15f\n", coef(ridge_fit)[3]))
  } else {
    # Use glmnet with alpha=0 (ridge)
    # Note: glmnet uses different lambda scaling, so we adjust
    ridge_fit <- glmnet(x2, y2, alpha = 0, lambda = lam / n2,
                        standardize = FALSE, intercept = TRUE)
    cat(sprintf("intercept = %.15f\n", coef(ridge_fit)[1]))
    cat(sprintf("coef1 = %.15f\n", coef(ridge_fit)[2]))
    cat(sprintf("coef2 = %.15f\n", coef(ridge_fit)[3]))
  }
}

# ============================================================================
# Dataset 4: Elastic Net
# ============================================================================
cat("\n### DATASET 4: Elastic Net ###\n")

# Create data with more features for sparsity demonstration
set.seed(42)
n4 <- 100
p4 <- 5
x4 <- matrix(rnorm(n4 * p4), n4, p4)
# True coefficients: only first 2 are non-zero
true_coef <- c(3.0, -2.0, 0, 0, 0)
y4 <- x4 %*% true_coef + rnorm(n4, sd = 0.5)

cat("\n# X data dimensions:", n4, "x", p4, "\n")
cat("# First 10 rows of X:\n")
for (i in 1:10) {
  cat(sprintf("X4_row%d = [%s]\n", i, paste(sprintf("%.10f", x4[i,]), collapse = ", ")))
}
cat("# Y first 10:\n")
cat("Y4_first10 = [", paste(sprintf("%.10f", y4[1:10]), collapse = ", "), "]\n")

# Full data for Rust tests
cat("\n# Full X4 data (flattened row-major):\n")
cat("X4_flat = [", paste(sprintf("%.10f", t(x4)), collapse = ", "), "]\n")
cat("\n# Full Y4 data:\n")
cat("Y4 = [", paste(sprintf("%.10f", y4), collapse = ", "), "]\n")

# Elastic Net with alpha = 0.5
cat("\n# Elastic Net (alpha=0.5, lambda=0.1):\n")
enet_fit <- glmnet(x4, y4, alpha = 0.5, lambda = 0.1 / n4,
                   standardize = FALSE, intercept = TRUE)
cat(sprintf("intercept = %.15f\n", coef(enet_fit)[1]))
for (j in 1:p4) {
  cat(sprintf("coef%d = %.15f\n", j, coef(enet_fit)[j+1]))
}

# Lasso (alpha = 1)
cat("\n# Lasso (alpha=1.0, lambda=0.1):\n")
lasso_fit <- glmnet(x4, y4, alpha = 1.0, lambda = 0.1 / n4,
                    standardize = FALSE, intercept = TRUE)
cat(sprintf("intercept = %.15f\n", coef(lasso_fit)[1]))
for (j in 1:p4) {
  cat(sprintf("coef%d = %.15f\n", j, coef(lasso_fit)[j+1]))
}

# ============================================================================
# Dataset 5: Weighted Least Squares
# ============================================================================
cat("\n### DATASET 5: Weighted Least Squares ###\n")

set.seed(42)
n5 <- 30
x5 <- 1:n5
# Heteroscedastic errors: variance increases with x
y5 <- 2.0 + 1.5 * x5 + rnorm(n5, sd = 0.1 * x5)

# Weights inversely proportional to variance
weights5 <- 1 / (x5^2)

cat("\n# X data:\n")
cat("X5 = [", paste(sprintf("%.10f", x5), collapse = ", "), "]\n")
cat("\n# Y data:\n")
cat("Y5 = [", paste(sprintf("%.10f", y5), collapse = ", "), "]\n")
cat("\n# Weights:\n")
cat("W5 = [", paste(sprintf("%.10f", weights5), collapse = ", "), "]\n")

# WLS fit
wls5 <- lm(y5 ~ x5, weights = weights5)
cat("\n# WLS Results:\n")
cat(sprintf("intercept = %.15f\n", coef(wls5)[1]))
cat(sprintf("coefficient = %.15f\n", coef(wls5)[2]))
cat(sprintf("r_squared = %.15f\n", summary(wls5)$r.squared))

# Compare with OLS
ols5 <- lm(y5 ~ x5)
cat("\n# OLS (for comparison):\n")
cat(sprintf("intercept = %.15f\n", coef(ols5)[1]))
cat(sprintf("coefficient = %.15f\n", coef(ols5)[2]))

# ============================================================================
# Dataset 6: Collinearity Test
# ============================================================================
cat("\n### DATASET 6: Collinearity Test ###\n")

set.seed(42)
n6 <- 50
x6_1 <- rnorm(n6)
x6_2 <- x6_1 + rnorm(n6, sd = 0.01)  # Nearly collinear
x6_3 <- rnorm(n6)  # Independent
x6 <- cbind(x6_1, x6_2, x6_3)
y6 <- 1 + 2 * x6_1 + 3 * x6_3 + rnorm(n6, sd = 0.5)

cat("\n# X6 columns:\n")
cat("X6_col1 = [", paste(sprintf("%.10f", x6[,1]), collapse = ", "), "]\n")
cat("X6_col2 = [", paste(sprintf("%.10f", x6[,2]), collapse = ", "), "]\n")
cat("X6_col3 = [", paste(sprintf("%.10f", x6[,3]), collapse = ", "), "]\n")
cat("\n# Y6:\n")
cat("Y6 = [", paste(sprintf("%.10f", y6), collapse = ", "), "]\n")

# VIF calculation
library(MASS)
ols6 <- lm(y6 ~ x6)

# Manual VIF calculation
vif1 <- 1 / (1 - summary(lm(x6[,1] ~ x6[,2] + x6[,3]))$r.squared)
vif2 <- 1 / (1 - summary(lm(x6[,2] ~ x6[,1] + x6[,3]))$r.squared)
vif3 <- 1 / (1 - summary(lm(x6[,3] ~ x6[,1] + x6[,2]))$r.squared)

cat("\n# VIF values:\n")
cat(sprintf("vif1 = %.15f\n", vif1))
cat(sprintf("vif2 = %.15f\n", vif2))
cat(sprintf("vif3 = %.15f\n", vif3))

cat("\n# OLS coefficients (may be unstable due to collinearity):\n")
cat(sprintf("intercept = %.15f\n", coef(ols6)[1]))
cat(sprintf("coef1 = %.15f\n", coef(ols6)[2]))
cat(sprintf("coef2 = %.15f\n", coef(ols6)[3]))
cat(sprintf("coef3 = %.15f\n", coef(ols6)[4]))

# Ridge handles collinearity better
cat("\n# Ridge (lambda=0.1) on collinear data:\n")
ridge6 <- glmnet(x6, y6, alpha = 0, lambda = 0.1 / n6,
                 standardize = FALSE, intercept = TRUE)
cat(sprintf("intercept = %.15f\n", coef(ridge6)[1]))
cat(sprintf("coef1 = %.15f\n", coef(ridge6)[2]))
cat(sprintf("coef2 = %.15f\n", coef(ridge6)[3]))
cat(sprintf("coef3 = %.15f\n", coef(ridge6)[4]))

# ============================================================================
# Dataset 7: Leverage and Cook's Distance
# ============================================================================
cat("\n### DATASET 7: Leverage and Diagnostics ###\n")

set.seed(42)
n7 <- 20
x7 <- c(1:19, 50)  # Point 20 has high leverage
y7 <- 2 + 3 * (1:20) + c(rnorm(19, sd = 1), 10)  # Point 20 also has unusual y

cat("\n# X7:\n")
cat("X7 = [", paste(sprintf("%.10f", x7), collapse = ", "), "]\n")
cat("\n# Y7:\n")
cat("Y7 = [", paste(sprintf("%.10f", y7), collapse = ", "), "]\n")

ols7 <- lm(y7 ~ x7)
inf7 <- influence(ols7)
cooks7 <- cooks.distance(ols7)
leverage7 <- hatvalues(ols7)

cat("\n# Leverage values:\n")
cat("leverage = [", paste(sprintf("%.15f", leverage7), collapse = ", "), "]\n")

cat("\n# Cook's distance:\n")
cat("cooks_d = [", paste(sprintf("%.15f", cooks7), collapse = ", "), "]\n")

cat("\n# Studentized residuals:\n")
stud_resid7 <- rstudent(ols7)
cat("studentized_resid = [", paste(sprintf("%.15f", stud_resid7), collapse = ", "), "]\n")

cat("\n" , rep("=", 70), "\n", sep = "")
cat("END OF VALIDATION DATA\n")
cat(rep("=", 70), "\n", sep = "")
