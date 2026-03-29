#' @importFrom stats rbinom rpois rgamma rexp terms rnorm rnbinom
NULL

#' Simulate data from tulpa models
#'
#' @description
#' Generate simulated datasets from prior or posterior predictive distributions.
#' Useful for prior predictive checks, simulation-based calibration, and
#' understanding model behavior.
#'
#' @details
#' This is a placeholder for the generic simulation infrastructure.
#' Model-specific simulation functions (e.g., for ratio models) are provided
#' by the model packages (numdenom, tulpaOcc, etc.).
#'
#' @name tulpa_simulate
#' @keywords internal
NULL


#' Generate random effects for simulation
#'
#' @param n_groups Number of groups
#' @param sigma Standard deviation
#' @param type Type of random effects: "iid" (default), "car", "ar1"
#'
#' @return Numeric vector of random effects
#' @keywords internal
sim_random_effects <- function(n_groups, sigma = 0.5, type = "iid") {

  if (type == "iid") {
    return(rnorm(n_groups, 0, sigma))
  }

  if (type == "ar1") {
    # Generate AR(1) process
    re <- numeric(n_groups)
    rho <- 0.7
    re[1] <- rnorm(1, 0, sigma / sqrt(1 - rho^2))
    for (i in 2:n_groups) {
      re[i] <- rho * re[i - 1] + rnorm(1, 0, sigma)
    }
    return(re)
  }

  # Default: IID
  rnorm(n_groups, 0, sigma)
}


#' Generate spatial random effects for simulation
#'
#' @param adjacency Adjacency matrix
#' @param sigma Standard deviation
#' @param type Spatial type: "icar" (default), "bym2"
#'
#' @return Numeric vector of spatial effects
#' @keywords internal
sim_spatial_effects <- function(adjacency, sigma = 0.5, type = "icar") {
  n <- nrow(adjacency)

  if (type == "icar") {
    # Generate ICAR effects via conditional simulation
    adj <- as.matrix(adjacency)
    n_neighbors <- rowSums(adj)
    Q <- diag(n_neighbors) - adj

    # Regularize for sampling
    Q_reg <- Q + 0.01 * diag(n)
    L <- chol(Q_reg)

    # Generate from precision matrix
    z <- rnorm(n)
    phi <- backsolve(L, z) * sigma

    # Center
    phi <- phi - mean(phi)
    return(phi)
  }

  # Default: IID
  rnorm(n, 0, sigma)
}


#' Generate temporal random effects for simulation
#'
#' @param n_times Number of time points
#' @param sigma Standard deviation
#' @param type Temporal type: "rw1", "rw2", "ar1"
#' @param rho Autocorrelation for AR(1). Default 0.7.
#'
#' @return Numeric vector of temporal effects
#' @keywords internal
sim_temporal_effects <- function(n_times, sigma = 0.5, type = "rw1",
                                  rho = 0.7) {

  if (type == "rw1") {
    # First-order random walk
    phi <- cumsum(rnorm(n_times, 0, sigma))
    phi <- phi - mean(phi)  # Center
    return(phi)
  }

  if (type == "rw2") {
    # Second-order random walk (smoother)
    phi <- numeric(n_times)
    phi[1] <- rnorm(1, 0, sigma)
    if (n_times >= 2) phi[2] <- phi[1] + rnorm(1, 0, sigma)
    for (t in 3:n_times) {
      phi[t] <- 2 * phi[t - 1] - phi[t - 2] + rnorm(1, 0, sigma)
    }
    phi <- phi - mean(phi)
    return(phi)
  }

  if (type == "ar1") {
    return(sim_random_effects(n_times, sigma = sigma, type = "ar1"))
  }

  # Default: IID
  rnorm(n_times, 0, sigma)
}
