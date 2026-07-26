# Standalone check of the truncated-family weight w and its first TWO
# eta-derivatives, the ingredient has_curvature_2nd_derivative() currently
# withholds for truncated_poisson / truncated_neg_binomial_2.
#
# The working weight of a zero-truncated family is w = f(a, da) with
#     f(a, da) = da / p - q da^2 / p^2,   q = e^{-a},  p = 1 - q
# a two-variable function of the shape a(eta) and its eta-derivative da(eta).
# Because w depends on eta only through (a, da), its eta-derivatives are the
# standard composite:
#     dw/deta   = f_a da + f_da d2a
#     d2w/deta2 = f_aa da^2 + 2 f_ada da d2a + f_dada d2a^2 + f_a d2a + f_da d3a
# The point of this proto is to confirm those two closed forms against a central
# difference of w and of dw, for both truncated families, BEFORE porting the
# d3a and the d2w branch into laplace_family_link.h / laplace_family_curvature.h.

options(digits = 12)

# a(eta) and its first THREE eta-derivatives, mu = exp(eta).
shape <- function(family, eta, phi) {
  mu <- exp(eta)
  if (family == "truncated_poisson") {
    return(list(a = mu, da = mu, d2a = mu, d3a = mu))
  }
  s <- phi + mu
  list(a   = phi * log1p(mu / phi),
       da  = phi * mu / s,
       d2a = phi^2 * mu / s^2,
       d3a = phi^2 * mu * (phi - mu) / s^3)
}

# The weight itself: e_weight = da/p - q da^2/p^2 (truncation_term in the C++).
weight <- function(family, eta, phi) {
  sh <- shape(family, eta, phi)
  a <- sh$a; da <- sh$da
  q <- exp(-a); p <- -expm1(-a)
  da / p - q * da^2 / p^2
}

# Analytic dw/deta and d2w/deta2 from the composite formulas above.
dweight <- function(family, eta, phi) {
  sh <- shape(family, eta, phi)
  a <- sh$a; da <- sh$da; d2a <- sh$d2a
  q <- exp(-a); p <- -expm1(-a)
  f_a  <- -da * q / p^2 + da^2 * q * (p + 2 * q) / p^3
  f_da <- 1 / p - 2 * q * da / p^2
  f_a * da + f_da * d2a
}

d2weight <- function(family, eta, phi) {
  sh <- shape(family, eta, phi)
  a <- sh$a; da <- sh$da; d2a <- sh$d2a; d3a <- sh$d3a
  q <- exp(-a); p <- -expm1(-a)
  f_a    <- -da * q / p^2 + da^2 * q * (p + 2 * q) / p^3
  f_da   <- 1 / p - 2 * q * da / p^2
  f_dada <- -2 * q / p^2
  f_ada  <- -q / p^2 + 2 * da * q * (p + 2 * q) / p^3
  R_a    <- -q * (p + 2 * q) / p^3 - 2 * q^2 * (2 * p + 3 * q) / p^4
  f_aa   <- da * q * (p + 2 * q) / p^3 + da^2 * R_a
  f_aa * da^2 + 2 * f_ada * da * d2a + f_dada * d2a^2 + f_a * d2a + f_da * d3a
}

fd  <- function(f, eta, h = 1e-6) (f(eta + h) - f(eta - h)) / (2 * h)

cat("truncated-family weight derivative check\n")
cat("========================================\n\n")
grid_eta <- c(-1.5, -0.5, 0.2, 1.0, 1.8)
fams <- list(c("truncated_poisson", NA), c("truncated_neg_binomial_2", 2.5),
             c("truncated_neg_binomial_2", 0.7))
worst_dw <- 0; worst_d2w <- 0
for (fp in fams) {
  fam <- fp[1]; phi <- as.numeric(fp[2])
  e_dw <- e_d2w <- 0
  for (eta in grid_eta) {
    dw_an  <- dweight(fam, eta, phi)
    dw_fd  <- fd(function(e) weight(fam, e, phi), eta)
    d2w_an <- d2weight(fam, eta, phi)
    d2w_fd <- fd(function(e) dweight(fam, e, phi), eta)
    e_dw  <- max(e_dw,  abs(dw_an - dw_fd)   / max(abs(dw_fd), 1e-8))
    e_d2w <- max(e_d2w, abs(d2w_an - d2w_fd) / max(abs(d2w_fd), 1e-8))
  }
  worst_dw  <- max(worst_dw, e_dw); worst_d2w <- max(worst_d2w, e_d2w)
  cat(sprintf("  %-26s phi=%-4s  dw max|rel|=%.2e   d2w max|rel|=%.2e\n",
              fam, ifelse(is.na(phi), "-", phi), e_dw, e_d2w))
}
cat(sprintf("\nWORST  dw=%.2e  d2w=%.2e  %s\n", worst_dw, worst_d2w,
            if (max(worst_dw, worst_d2w) < 1e-5) "PASS" else "FAIL"))
