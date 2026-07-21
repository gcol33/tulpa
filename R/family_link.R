# family_link.R
# ------------------------------------------------------------------------------
# The link layer: what turns a base family into its `<family>_<link>` forms.
#
# `.FAMILY_OPS` (family_loglik.R) writes each family's log-likelihood, score and
# working weight directly in eta, with the canonical link already substituted and
# simplified -- poisson's score is `y - mu`, not `(y/mu - 1) * mu`. Those forms
# are the numerically preferred ones and stay exactly as they are.
#
# A non-canonical link cannot reuse them, because the simplification is what
# baked the link in. It needs the family in MU space -- the log-density, the
# score with respect to mu, and the variance function -- composed with the link's
# inverse and its derivative:
#
#     mu     = linkinv(eta)
#     dmu    = mu_eta(eta)
#     score  = (d log f / d mu) * dmu
#     weight = dmu^2 / V(mu)
#
# which is the standard GLM construction and mirrors how the compiled side is
# organised (grad_mu / variance_fn composed with linkinv / mu_eta in
# src/laplace_family_link.h).
#
# The two representations describe the same family, so they can drift. They are
# pinned against each other by test-family-link.R, which composes every base
# family with its own canonical link and requires the result to match the
# hand-simplified `.FAMILY_OPS` entry.
# ------------------------------------------------------------------------------

# --- links --------------------------------------------------------------------

# exp with the argument bounded, matching tulpa_linalg::safe_exp so the R and C++
# means agree in the far tail rather than one overflowing to Inf.
.EXP_ARG_MAX <- 700
.safe_exp <- function(x) exp(pmin(pmax(x, -.EXP_ARG_MAX), .EXP_ARG_MAX))

# Floor for the links whose inverse is singular at eta = 0. This is a NaN guard
# only: the domain check below is what actually keeps eta positive, exactly as
# safe_pos_eta and link_eta_in_domain divide the work in C++. Below the floor the
# returned mean is constant while mu_eta is not, so the pair is not a usable
# extension of the link -- it only has to be finite.
.ETA_FLOOR <- 1e-10
.safe_pos_eta <- function(eta) pmax(eta, .ETA_FLOOR)

# Each entry mirrors linkinv / mu_eta in src/laplace_family_link.h.
# `positive_eta` marks the links carried on the open half-line eta > 0: inverse
# and 1mu2 because mu is undefined otherwise, sqrt because mu = eta^2 makes eta
# and -eta observationally identical, so admitting both branches would make the
# mode a mirror pair with a singular Hessian between them.
.LINKS <- list(
  identity = list(
    linkfun = function(mu) mu,
    linkinv = function(eta) eta,
    mu_eta  = function(eta) rep(1, length(eta)),
    positive_eta = FALSE
  ),
  log = list(
    linkfun = function(mu) log(mu),
    linkinv = function(eta) .safe_exp(eta),
    mu_eta  = function(eta) .safe_exp(eta),
    positive_eta = FALSE
  ),
  inverse = list(
    linkfun = function(mu) 1 / mu,
    linkinv = function(eta) 1 / .safe_pos_eta(eta),
    mu_eta  = function(eta) -1 / .safe_pos_eta(eta)^2,
    positive_eta = TRUE
  ),
  logit = list(
    linkfun = function(mu) log(mu) - log1p(-mu),
    linkinv = function(eta) stats::plogis(eta),
    mu_eta  = function(eta) { p <- stats::plogis(eta); p * (1 - p) },
    positive_eta = FALSE
  ),
  probit = list(
    linkfun = function(mu) stats::qnorm(mu),
    linkinv = function(eta) stats::pnorm(eta),
    mu_eta  = function(eta) stats::dnorm(eta),
    positive_eta = FALSE
  ),
  cauchit = list(
    linkfun = function(mu) tan(pi * (mu - 0.5)),
    linkinv = function(eta) 0.5 + atan(eta) / pi,
    mu_eta  = function(eta) 1 / (pi * (1 + eta^2)),
    positive_eta = FALSE
  ),
  cloglog = list(
    linkfun = function(mu) log(-log1p(-mu)),
    linkinv = function(eta) -expm1(-exp(eta)),
    mu_eta  = function(eta) exp(eta - exp(eta)),
    positive_eta = FALSE
  ),
  sqrt = list(
    linkfun = function(mu) sqrt(mu),
    linkinv = function(eta) eta^2,
    mu_eta  = function(eta) 2 * eta,
    positive_eta = TRUE
  ),
  `1mu2` = list(
    linkfun = function(mu) 1 / mu^2,
    linkinv = function(eta) 1 / sqrt(.safe_pos_eta(eta)),
    mu_eta  = function(eta) -0.5 / .safe_pos_eta(eta)^1.5,
    positive_eta = TRUE
  )
)

#' Links available as a `<family>_<link>` suffix.
#' @keywords internal
link_names <- function() names(.LINKS)

# Canonical link per base family; a family absent here takes no link suffix.
# Mirrors the defaults table in parse_family_link (src/laplace_family_link.h), so
# the two sides agree on both which suffixes exist and what the bare name means.
.LINK_DEFAULTS <- c(
  gaussian         = "identity",
  binomial         = "logit",
  poisson          = "log",
  neg_binomial_2   = "log",
  gamma            = "log",
  inverse_gaussian = "log",
  beta             = "logit",
  lognormal        = "identity"
)

# Links admissible for each base family, mirroring the `okLinks` of the
# corresponding stats::family() where one exists, so a tulpa code means what the
# same code means to glm(). This is deliberately stricter than the compiled
# parser, which will resolve any of the nine links against any base: the range
# constraint on mu makes most of that cross product meaningless (binomial_sqrt
# would put mu = eta^2 outside (0, 1) and reach the likelihood only through a
# clamp), and the front door should not offer a combination it cannot fit.
.OK_LINKS <- list(
  gaussian         = c("identity", "log", "inverse"),
  binomial         = c("logit", "probit", "cauchit", "log", "cloglog"),
  poisson          = c("log", "identity", "sqrt"),
  neg_binomial_2   = c("log", "identity", "sqrt"),
  gamma            = c("inverse", "identity", "log"),
  inverse_gaussian = c("1mu2", "inverse", "identity", "log"),
  beta             = c("logit", "probit", "cauchit", "cloglog"),
  lognormal        = c("identity", "log")
)

# Split a family code into base and link. Returns NULL when the code is not a
# `<base>_<link>` form over a link-capable base, so the caller can fall through
# to the plain registry error rather than inventing a family.
.parse_family_link <- function(code) {
  if (!is.character(code) || length(code) != 1L || is.na(code)) return(NULL)
  # `[[` on a named character vector ERRORS for an absent name (unlike a list,
  # which returns NULL), so membership is tested before indexing.
  if (code %in% names(.LINK_DEFAULTS)) {
    return(list(base = code, link = unname(.LINK_DEFAULTS[[code]])))
  }
  # Longest base first: "beta_binomial" begins with "beta_", so a registration
  # order scan would read it as beta with a "binomial" link.
  bases <- names(.LINK_DEFAULTS)
  bases <- bases[order(nchar(bases), decreasing = TRUE)]
  for (base in bases) {
    prefix <- paste0(base, "_")
    if (!startsWith(code, prefix)) next
    link <- substring(code, nchar(prefix) + 1L)
    if (link %in% .OK_LINKS[[base]]) return(list(base = base, link = link))
  }
  NULL
}

# --- families in mu space -----------------------------------------------------

# Per base family:
#   loglik_mu(y, mu, phi, n)   log-density at the mean mu (full normalizer)
#   grad_mu(y, mu, phi, n)     d log f / d mu
#   working_var(mu, phi, n)    V in weight = (dmu/deta)^2 / V. NOT Var(y): for
#                              binomial the response variance is n mu (1-mu)
#                              while V is mu (1-mu) / n, and for beta V is the
#                              inverse Fisher information on mu rather than the
#                              beta variance.
#   resp_var(mu, phi, n)       Var(y | eta) on the response scale
#   resp_mean(mu, phi, n)      E[y | eta]; differs from mu for binomial (trial
#                              scaled) and lognormal (mu is the log-scale mean)
#   sample(mu, phi, n)         one draw per element
#   clamp(mu)                  admissible-range clamp, mirroring the C++ generic
#                              route's mu guards
.clamp_unit  <- function(mu) pmin(pmax(mu, 1e-10), 1 - 1e-10)
.clamp_pos   <- function(mu) pmax(mu, 1e-10)
.clamp_none  <- function(mu) mu

.FAMILY_MU <- list(
  gaussian = list(
    # phi is the residual VARIANCE here, the R-side convention (the compiled
    # kernels take the SD and are handed sqrt(phi) at the front door).
    loglik_mu   = function(y, mu, phi, n) -0.5 * ((y - mu)^2 / phi + log(2 * pi * phi)),
    grad_mu     = function(y, mu, phi, n) (y - mu) / phi,
    working_var = function(mu, phi, n) rep(phi, length(mu)),
    resp_var    = function(mu, phi, n) rep(phi, length(mu)),
    resp_mean   = function(mu, phi, n) mu,
    sample      = function(mu, phi, n) stats::rnorm(length(mu), mu, sqrt(phi)),
    clamp       = .clamp_none
  ),

  lognormal = list(
    # mu is the LOG-scale mean (the link acts on log y), phi the log-scale
    # variance. The -log(y) Jacobian is part of the density.
    loglik_mu   = function(y, mu, phi, n) {
      ly <- log(y)
      -ly - 0.5 * log(2 * pi * phi) - (ly - mu)^2 / (2 * phi)
    },
    grad_mu     = function(y, mu, phi, n) (log(y) - mu) / phi,
    working_var = function(mu, phi, n) rep(phi, length(mu)),
    resp_var    = function(mu, phi, n) (exp(phi) - 1) * exp(2 * mu + phi),
    resp_mean   = function(mu, phi, n) exp(mu + phi / 2),
    sample      = function(mu, phi, n) stats::rlnorm(length(mu), mu, sqrt(phi)),
    clamp       = .clamp_none
  ),

  binomial = list(
    loglik_mu   = function(y, mu, phi, n) lchoose(n, y) + y * log(mu) + (n - y) * log1p(-mu),
    grad_mu     = function(y, mu, phi, n) (y - n * mu) / (mu * (1 - mu)),
    working_var = function(mu, phi, n) mu * (1 - mu) / n,
    resp_var    = function(mu, phi, n) n * mu * (1 - mu),
    resp_mean   = function(mu, phi, n) n * mu,
    sample      = function(mu, phi, n) stats::rbinom(length(mu), size = n, prob = mu),
    clamp       = .clamp_unit
  ),

  poisson = list(
    loglik_mu   = function(y, mu, phi, n) y * log(mu) - mu - lgamma(y + 1),
    grad_mu     = function(y, mu, phi, n) y / mu - 1,
    working_var = function(mu, phi, n) mu,
    resp_var    = function(mu, phi, n) mu,
    resp_mean   = function(mu, phi, n) mu,
    sample      = function(mu, phi, n) stats::rpois(length(mu), mu),
    clamp       = .clamp_pos
  ),

  neg_binomial_2 = list(
    loglik_mu   = function(y, mu, phi, n) {
      lgamma(y + phi) - lgamma(phi) - lgamma(y + 1) +
        phi * (log(phi) - log(phi + mu)) + y * (log(mu) - log(mu + phi))
    },
    grad_mu     = function(y, mu, phi, n) y / mu - (y + phi) / (mu + phi),
    working_var = function(mu, phi, n) mu + mu^2 / phi,
    resp_var    = function(mu, phi, n) mu + mu^2 / phi,
    resp_mean   = function(mu, phi, n) mu,
    sample      = function(mu, phi, n) stats::rnbinom(length(mu), size = phi, mu = mu),
    clamp       = .clamp_pos
  ),

  gamma = list(
    # phi is the shape; variance mu^2 / phi.
    loglik_mu   = function(y, mu, phi, n) {
      phi * log(phi) - lgamma(phi) + (phi - 1) * log(y) - phi * log(mu) - phi * y / mu
    },
    grad_mu     = function(y, mu, phi, n) phi * (y - mu) / mu^2,
    working_var = function(mu, phi, n) mu^2 / phi,
    resp_var    = function(mu, phi, n) mu^2 / phi,
    resp_mean   = function(mu, phi, n) mu,
    sample      = function(mu, phi, n) stats::rgamma(length(mu), shape = phi, rate = phi / mu),
    clamp       = .clamp_pos
  ),

  inverse_gaussian = list(
    # phi is the dispersion; variance phi * mu^3.
    loglik_mu   = function(y, mu, phi, n) {
      -0.5 * log(2 * pi * phi * y^3) - (y - mu)^2 / (2 * phi * mu^2 * y)
    },
    grad_mu     = function(y, mu, phi, n) (y - mu) / (phi * mu^3),
    working_var = function(mu, phi, n) phi * mu^3,
    resp_var    = function(mu, phi, n) phi * mu^3,
    resp_mean   = function(mu, phi, n) mu,
    sample      = function(mu, phi, n) .rinvgauss(length(mu), mu = mu, lambda = 1 / phi),
    clamp       = .clamp_pos
  ),

  beta = list(
    # phi is the precision a + b. The working variance is the inverse Fisher
    # information on mu (Ferrari & Cribari-Neto 2004), not Var(y).
    loglik_mu   = function(y, mu, phi, n) {
      a <- mu * phi; b <- (1 - mu) * phi
      lgamma(phi) - lgamma(a) - lgamma(b) + (a - 1) * log(y) + (b - 1) * log1p(-y)
    },
    grad_mu     = function(y, mu, phi, n) {
      phi * (log(y) - log1p(-y) - digamma(mu * phi) + digamma((1 - mu) * phi))
    },
    working_var = function(mu, phi, n) {
      1 / (phi^2 * (trigamma(mu * phi) + trigamma((1 - mu) * phi)))
    },
    resp_var    = function(mu, phi, n) mu * (1 - mu) / (phi + 1),
    resp_mean   = function(mu, phi, n) mu,
    sample      = function(mu, phi, n) stats::rbeta(length(mu), mu * phi, (1 - mu) * phi),
    clamp       = function(mu) pmin(pmax(mu, 1e-7), 1 - 1e-7)
  )
)

# --- composition --------------------------------------------------------------

# Build the eta-space operation set for one (base, link) pair, in the shape
# `.FAMILY_OPS` entries have, so every consumer -- glmm_weights(), the sampler
# log-posterior, posterior_predict() -- reads a suffixed family through exactly
# the same interface as a canonical one.
.compose_family_ops <- function(base, link) {
  fm <- .FAMILY_MU[[base]]
  lk <- .LINKS[[link]]
  if (is.null(fm) || is.null(lk)) return(NULL)

  n_or_1 <- function(n_trials, k) if (!is.null(n_trials)) n_trials else rep(1, k)
  mu_of  <- function(eta) fm$clamp(lk$linkinv(eta))

  # Outside a constrained link's domain the density is -Inf, matching the
  # compiled barrier (log_lik_for_family). Without this the R log-lik would
  # report a finite value at an eta the engine refuses to visit, and the two
  # sides would disagree about where the model is defined.
  outside <- if (lk$positive_eta) function(eta) eta <= 0 else function(eta) rep(FALSE, length(eta))

  list(
    mean = function(eta, phi = 1.0, ...) mu_of(eta),

    loglik = function(eta, y, n_trials, phi) {
      n  <- n_or_1(n_trials, length(eta))
      ll <- fm$loglik_mu(y, mu_of(eta), phi, n)
      ll[outside(eta)] <- -Inf
      ll
    },

    score = function(eta, y, n_trials, phi) {
      n <- n_or_1(n_trials, length(eta))
      fm$grad_mu(y, mu_of(eta), phi, n) * lk$mu_eta(eta)
    },

    weight = function(eta, n_trials, phi) {
      n <- n_or_1(n_trials, length(eta))
      lk$mu_eta(eta)^2 / fm$working_var(mu_of(eta), phi, n)
    },

    sample = function(eta, n_trials, phi) {
      n <- n_or_1(n_trials, length(eta))
      fm$sample(mu_of(eta), phi, n)
    },

    variance = function(eta, n_trials, phi) {
      n <- n_or_1(n_trials, length(eta))
      fm$resp_var(mu_of(eta), phi, n)
    },

    response_mean = function(eta, n_trials, phi) {
      n <- n_or_1(n_trials, length(eta))
      fm$resp_mean(mu_of(eta), phi, n)
    }
  )
}

# Composed operation sets are built once per family code and reused. The closures
# are stateless, so caching is a pure allocation saving.
.family_link_cache <- new.env(parent = emptyenv())

# Operation set for a `<base>_<link>` code, or NULL when the code is not one.
# A code naming a base family's OWN canonical link resolves to the hand-written
# `.FAMILY_OPS` entry rather than the composition, so the canonical path keeps
# its simplified arithmetic and stays numerically identical.
.linked_family_ops <- function(family) {
  if (!is.null(.FAMILY_OPS[[family]])) return(.FAMILY_OPS[[family]])
  hit <- get0(family, envir = .family_link_cache, inherits = FALSE)
  if (!is.null(hit)) return(hit)

  parsed <- .parse_family_link(family)
  if (is.null(parsed)) return(NULL)
  if (identical(parsed$link, unname(.LINK_DEFAULTS[[parsed$base]]))) {
    return(.FAMILY_OPS[[parsed$base]])
  }

  ops <- .compose_family_ops(parsed$base, parsed$link)
  if (is.null(ops)) return(NULL)
  assign(family, ops, envir = .family_link_cache)
  ops
}

#' Family codes the link layer accepts beyond the bare registry names.
#'
#' The cross product of the link-capable base families and the links, minus each
#' family's own canonical form (which is already a registry name).
#' @keywords internal
linked_family_names <- function() {
  out <- unlist(lapply(names(.OK_LINKS), function(base) {
    links <- setdiff(.OK_LINKS[[base]], .LINK_DEFAULTS[[base]])
    if (!length(links)) return(character(0))
    paste0(base, "_", links)
  }), use.names = FALSE)
  sort(out)
}
