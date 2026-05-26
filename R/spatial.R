#' Spatial structure specifications for tulpa
#'
#' @description
#' Functions to specify spatial random effects for tulpa models.
#' Spatial effects are shared between processes by default,
#' which helps prevent bias from spatially-structured
#' unmeasured confounders.
#'
#' @name tulpa_spatial
NULL

#' CAR spatial structure
#'
#' @description
#' Specify a conditional autoregressive (CAR) spatial random effect.
#' Supports both intrinsic CAR (ICAR) and proper CAR variants.
#'
#' @param adjacency Adjacency matrix (sparse or dense). A symmetric matrix
#'   where entry (i,j) is 1 if areas i and j are neighbors, 0 otherwise.
#' @param level Level at which spatial structure applies:
#'   - `"group"`: Spatial effect at the grouping variable level (e.g., sites).
#'     Requires `group_var` to be specified.
#'   - `"obs"`: Spatial effect at the observation level.
#' @param group_var Name of the grouping variable in data (required if
#'   `level = "group"`).
#' @param proper Logical; if FALSE (default), uses Intrinsic CAR (ICAR) with
#'   rho = 1 fixed. If TRUE, uses proper CAR with rho estimated. See Details.
#' @param shared Logical; if TRUE (default), spatial effect enters both
#'   all process linear predictors identically.
#' @param parameterization Spatial parameterization: `"standard"` (default)
#'   samples spatial effects directly; `"collapsed"` (deprecated) marginalizes
#'   spatial effects via inner Laplace approximation.
#'
#' @return A `tulpa_spatial` object
#'
#' @details
#' The CAR model specifies that:
#'
#' \deqn{\phi_i | \phi_{-i} \sim N\left(\rho \frac{\sum_{j \sim i} \phi_j}{n_i},
#'   \frac{\sigma^2}{n_i}\right)}
#'
#' where \eqn{n_i} is the number of neighbors of area i and \eqn{j \sim i}
#' denotes that j is a neighbor of i.
#'
#' This leads to a precision matrix:
#' \deqn{Q = \tau (D - \rho W)}
#'
#' where D is the diagonal matrix of neighbor counts and W is the adjacency
#' matrix.
#'
#' **ICAR (proper = FALSE, default)**
#'
#' - Sets rho = 1 (fixed)
#' - Improper prior (rank-deficient Q)
#' - Requires sum-to-zero constraint for identifiability
#' - Simpler: one fewer parameter to estimate
#' - Standard choice for disease mapping
#' - Equivalent to RW1 on a graph
#'
#' **Proper CAR (proper = TRUE)**
#'
#' - Estimates rho in (0, 1) from data
#' - Proper prior (integrates to 1)
#' - rho measures spatial autocorrelation strength
#' - rho -> 0: approaches independence (IID)
#' - rho -> 1: approaches ICAR
#' - More flexible but additional parameter to estimate
#' - Prior: rho ~ Beta(1, 1) (uniform on (0, 1))
#'
#' @examples
#' # Create adjacency matrix for 10 regions (chain structure)
#' adj <- matrix(0, 10, 10)
#' for (i in 1:9) {
#'   adj[i, i+1] <- adj[i+1, i] <- 1
#' }
#'
#' # ICAR (default, rho = 1 fixed)
#' icar <- spatial_car(adj, level = "group", group_var = "site")
#' print(icar)
#'
#' # Proper CAR (rho estimated)
#' proper_car <- spatial_car(adj, level = "group", group_var = "site",
#'                           proper = TRUE)
#' print(proper_car)
#'
#' \donttest{
#' # Generate synthetic data with spatial structure
#' set.seed(123)
#' n_sites <- 10
#' n_per_site <- 5
#' df <- data.frame(
#'   site = rep(1:n_sites, each = n_per_site),
#'   x = rnorm(n_sites * n_per_site),
#'   count = rpois(n_sites * n_per_site, 20),
#'   effort = rgamma(n_sites * n_per_site, shape = 5, rate = 1)
#' )
#'
#' # ICAR model (standard disease mapping approach)
#' fit_icar <- tulpa(
#'   count | effort ~ x,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   spatial = spatial_car(adj, level = "group", group_var = "site"),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#'
#' # Proper CAR model (estimate spatial autocorrelation)
#' fit_car <- tulpa(
#'   count | effort ~ x,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   spatial = spatial_car(adj, level = "group", group_var = "site",
#'                         proper = TRUE),
#'   backend = "hmc",
#'   iter = 200,
#'   warmup = 100,
#'   chains = 1
#' )
#'
#' # Extract rho from proper CAR
#' summary(fit_car)  # Shows rho_spatial parameter
#' }
#'
#' @seealso [spatial_bym2()] for decomposed spatial + IID effects,
#'   [spatial_gp()] for continuous spatial effects
#'
#' @export
